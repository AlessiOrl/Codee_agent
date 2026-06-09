from uuid import UUID

from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str | None = None
    url: str | None = None


class ChatRequest(BaseModel):
    telegram_user_id: str
    telegram_chat_id: str
    telegram_message_id: str | None = None
    username: str | None = None
    first_name: str | None = None
    text: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    conversation_id: UUID
    sources: list[Source] = Field(default_factory=list)


class ForgetRequest(BaseModel):
    telegram_user_id: str
    telegram_chat_id: str
    days: int | None = Field(default=None, ge=1)


class ForgetCounts(BaseModel):
    messages: int = 0
    summaries: int = 0
    memories: int = 0
    documents: int = 0
    document_chunks: int = 0
    tool_calls: int = 0
    conversations: int = 0


class ForgetResponse(BaseModel):
    reply: str
    counts: ForgetCounts
    scope: str


class SummaryRequest(BaseModel):
    telegram_user_id: str
    telegram_chat_id: str


class SummaryResponse(BaseModel):
    reply: str
    summary: str | None = None
    summary_created_at: str | None = None


class DebugLlmRequest(BaseModel):
    text: str = Field(min_length=1)


class DebugLlmResponse(BaseModel):
    reply: str
    sources: list[Source] = Field(default_factory=list)


class DebugEmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)


class DebugEmbeddingResponse(BaseModel):
    dimensions: int
    sample: list[float] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    chunks: int
    status: str


class HealthResponse(BaseModel):
    status: str
    postgres: str
    qdrant: str
