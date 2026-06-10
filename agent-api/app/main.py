import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent_graph import AgentGraph
from app.auth import require_api_key
from app.config import Settings, get_settings
from app.db import Database
from app.documents import DocumentService
from app.embeddings import EmbeddingError, Embeddings, OllamaEmbeddingsClient
from app.forget import ForgetService
from app.openwebui import OpenWebUIClient, OpenWebUIError
from app.repository import Repository
from app.schemas import (
    ChatRequest,
    ContextMessageRequest,
    ContextMessageResponse,
    DebugEmbeddingRequest,
    DebugEmbeddingResponse,
    DebugLlmRequest,
    DebugLlmResponse,
    DocumentUploadResponse,
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    SummaryRequest,
    SummaryResponse,
)
from app.summary import SummaryService
from app.vector_store import VectorStore, VectorStoreError


def context_debug_json(event: str, payload: dict) -> None:
    if not bool(getattr(app.state, "context_debug_json", False)):
        return
    LOGGER.info(
        "context_debug_json %s",
        json.dumps({"event": event, **payload}, default=str, ensure_ascii=False, sort_keys=True),
    )


class HealthAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in str(record.getMessage())


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname).1s %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").propagate = False
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.propagate = False
    if not any(isinstance(filter_, HealthAccessLogFilter) for filter_ in access_logger.filters):
        access_logger.addFilter(HealthAccessLogFilter())


configure_logging()
LOGGER = logging.getLogger("agent-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(settings.database_url)
    db.init_schema()

    llm = OpenWebUIClient(
        base_url=settings.openwebui_base_url,
        api_key=settings.openwebui_api_key,
        model=settings.openwebui_model,
        fallback_model=settings.openwebui_fallback_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_status_timeout_seconds=settings.ollama_status_timeout_seconds,
        timeout_seconds=settings.openwebui_timeout_seconds,
        use_web_search=settings.openwebui_use_web_search,
        embedding_model=settings.embedding_model,
    )
    embedding_provider = settings.embedding_provider or (
        "hash" if settings.use_local_hash_embeddings else "openwebui"
    )
    ollama_embeddings = None
    if embedding_provider.lower() == "ollama":
        if not settings.embedding_model:
            raise ValueError("EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=ollama")
        ollama_embeddings = OllamaEmbeddingsClient(
            base_url=settings.ollama_embeddings_base_url,
            model=settings.embedding_model,
            timeout_seconds=settings.openwebui_timeout_seconds,
        )
    embeddings = Embeddings(
        client=llm,
        ollama_client=ollama_embeddings,
        provider=embedding_provider,
        vector_size=settings.vector_size,
        use_local_hash_embeddings=settings.use_local_hash_embeddings,
    )
    vector_store = VectorStore(
        url=settings.qdrant_url,
        embeddings=embeddings,
        vector_size=settings.vector_size,
    )
    vector_store.init_collections()

    repository = Repository(db)
    app.state.db = db
    app.state.repository = repository
    app.state.vector_store = vector_store
    app.state.llm = llm
    app.state.context_debug_json = settings.context_debug_json
    app.state.agent_graph = AgentGraph(
        repository=repository,
        vector_store=vector_store,
        llm=llm,
        system_prompt=settings.system_prompt,
        recent_history_limit=settings.recent_history_limit,
        summary_every_n_messages=settings.summary_every_n_messages,
        cross_channel_memory_limit=settings.cross_channel_memory_limit,
        cross_channel_doc_limit=settings.cross_channel_doc_limit,
        cross_channel_min_score=settings.cross_channel_min_score,
        context_router_enabled=settings.context_router_enabled,
        context_router_max_feed_messages=settings.context_router_max_feed_messages,
        context_router_min_confidence=settings.context_router_min_confidence,
        context_gate_min_confidence=settings.context_gate_min_confidence,
        context_gate_ask_on_ambiguous=settings.context_gate_ask_on_ambiguous,
        external_fact_web_search_enabled=settings.external_fact_web_search_enabled,
        context_debug_json=settings.context_debug_json,
    )
    app.state.document_service = DocumentService(
        repository=repository,
        vector_store=vector_store,
        max_chars=settings.max_document_chars,
    )
    app.state.forget_service = ForgetService(
        repository=repository,
        vector_store=vector_store,
    )
    app.state.summary_service = SummaryService(repository=repository)
    LOGGER.info("Agent API initialized")
    yield


app = FastAPI(title="Agentic Codee API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    postgres = "unknown"
    status = "ok"
    try:
        postgres = app.state.db.health()
    except Exception:
        LOGGER.exception("Postgres health check failed")
        postgres = "error"
        status = "degraded"
    return HealthResponse(status=status, postgres=postgres, qdrant="ok")


@app.post("/debug/llm", response_model=DebugLlmResponse, dependencies=[Depends(require_api_key)])
async def debug_llm(request: DebugLlmRequest) -> DebugLlmResponse:
    try:
        result = await asyncio.to_thread(
            app.state.llm.chat,
            [{"role": "user", "content": request.text}],
        )
    except OpenWebUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DebugLlmResponse(reply=result.content, sources=result.sources)


@app.post(
    "/debug/embedding",
    response_model=DebugEmbeddingResponse,
    dependencies=[Depends(require_api_key)],
)
async def debug_embedding(request: DebugEmbeddingRequest) -> DebugEmbeddingResponse:
    try:
        vector = await asyncio.to_thread(app.state.vector_store.embeddings.embed, request.text)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DebugEmbeddingResponse(dimensions=len(vector), sample=vector[:5])


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    def events():
        try:
            for event in app.state.agent_graph.stream(request.model_dump()):
                yield json.dumps(event) + "\n"
        except (EmbeddingError, OpenWebUIError, VectorStoreError) as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        except Exception:
            LOGGER.exception("Streaming chat failed")
            yield json.dumps({"type": "error", "message": "Codee hit an API error while streaming."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/forget", response_model=ForgetResponse, dependencies=[Depends(require_api_key)])
async def forget(request: ForgetRequest) -> ForgetResponse:
    try:
        result = await asyncio.to_thread(
            app.state.forget_service.forget,
            telegram_user_id=request.telegram_user_id,
            telegram_chat_id=request.telegram_chat_id,
            channel=request.channel,
            days=request.days,
            scope=request.scope,
        )
    except VectorStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ForgetResponse(**result)


@app.post("/summary", response_model=SummaryResponse, dependencies=[Depends(require_api_key)])
async def summary(request: SummaryRequest) -> SummaryResponse:
    result = await asyncio.to_thread(
        app.state.summary_service.get_summary,
        telegram_user_id=request.telegram_user_id,
        telegram_chat_id=request.telegram_chat_id,
        channel=request.channel,
    )
    return SummaryResponse(**result)


@app.post(
    "/context/messages",
    response_model=ContextMessageResponse,
    dependencies=[Depends(require_api_key)],
)
async def ingest_context_message(request: ContextMessageRequest) -> ContextMessageResponse:
    repository: Repository = app.state.repository
    context_debug_json(
        "context_ingest_request",
        {
            "telegram_user_id": request.telegram_user_id,
            "telegram_chat_id": request.telegram_chat_id,
            "telegram_message_id": request.telegram_message_id,
            "channel": request.channel,
            "username": request.username,
            "first_name": request.first_name,
            "text": request.text,
        },
    )
    user = await asyncio.to_thread(
        repository.upsert_user,
        telegram_user_id=request.telegram_user_id,
        username=request.username,
        first_name=request.first_name,
    )
    try:
        point_id = await asyncio.to_thread(
            app.state.vector_store.upsert_context_message,
            user_id=user["id"],
            channel=request.channel,
            telegram_chat_id=request.telegram_chat_id,
            telegram_message_id=request.telegram_message_id,
            content=request.text,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    message = await asyncio.to_thread(
        repository.save_context_message,
        user_id=user["id"],
        channel=request.channel,
        telegram_chat_id=request.telegram_chat_id,
        telegram_message_id=request.telegram_message_id,
        content=request.text,
        qdrant_point_id=point_id,
    )
    context_debug_json(
        "context_ingest_stored",
        {
            "telegram_user_id": request.telegram_user_id,
            "db_user_id": str(user["id"]),
            "telegram_chat_id": request.telegram_chat_id,
            "telegram_message_id": request.telegram_message_id,
            "channel": request.channel,
            "context_message_id": str(message["id"]),
            "qdrant_point_id": point_id,
        },
    )
    return ContextMessageResponse(status="stored", message_id=message["id"])


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    dependencies=[Depends(require_api_key)],
)
async def upload_document(
    telegram_user_id: Annotated[str, Form()],
    telegram_chat_id: Annotated[str | None, Form()] = None,
    channel: Annotated[str | None, Form()] = None,
    username: Annotated[str | None, Form()] = None,
    first_name: Annotated[str | None, Form()] = None,
    file: UploadFile = File(),
) -> DocumentUploadResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    repository: Repository = app.state.repository
    user = await asyncio.to_thread(
        repository.upsert_user,
        telegram_user_id=telegram_user_id,
        username=username,
        first_name=first_name,
    )
    conversation = None
    if telegram_chat_id:
        if not channel:
            raise HTTPException(status_code=400, detail="channel is required when telegram_chat_id is provided")
        conversation = await asyncio.to_thread(
            repository.upsert_conversation,
            user_id=user["id"],
            telegram_chat_id=telegram_chat_id,
            channel=channel,
        )

    try:
        document_id, chunks = await asyncio.to_thread(
            app.state.document_service.ingest,
            user_id=user["id"],
            conversation_id=conversation["id"] if conversation else None,
            channel=channel,
            filename=file.filename or "upload",
            mime_type=file.content_type,
            content=content,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DocumentUploadResponse(document_id=document_id, chunks=chunks, status="stored")
