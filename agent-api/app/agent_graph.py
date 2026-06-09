import json
import logging
from typing import Any, Iterator, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.openwebui import LlmResult, OpenWebUIClient
from app.repository import Repository
from app.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)
PASSIVE_CONTEXT_CHANNELS = {"domotics", "unraid", "plex"}
PRIVATE_CHANNEL = "private"


def compact_text(value: Any, *, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def parse_llm_json_object(content: str) -> dict[str, Any]:
    candidates = [content.strip()]
    if candidates[0].startswith("```"):
        lines = candidates[0].splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())

    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Expected a JSON object")
        except Exception as exc:
            last_error = exc

        start = candidate.find("{")
        if start < 0:
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Expected a JSON object")
        except Exception as exc:
            last_error = exc

    raise ValueError("No JSON object found") from last_error


def is_telegram_command(text: str) -> bool:
    return text.strip().startswith("/")


def help_reply() -> str:
    return (
        "Available commands:\n"
        "/start - show the welcome message\n"
        "/help - show this command list\n"
        "/summary - show the private chat summary\n"
        "/forget - remove stored private chat context\n"
        "/forget <days> - remove private chat context from the last N days\n"
        "/forget_all - remove all stored context for your user across every channel\n"
        "/forget_all <days> - remove stored context from the last N days across every channel\n\n"
        "Send a normal text message to chat with Codee."
    )


class AgentState(TypedDict, total=False):
    telegram_user_id: str
    telegram_chat_id: str
    channel: str
    telegram_message_id: str | None
    username: str | None
    first_name: str | None
    text: str
    user: dict[str, Any]
    conversation: dict[str, Any]
    user_message: dict[str, Any]
    recent_messages: list[dict[str, Any]]
    summary: str | None
    memories: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    context_channels: list[str]
    context_router_rationale: str | None
    context_messages: list[dict[str, Any]]
    reply: str
    sources: list[dict[str, str | None]]


class AgentGraph:
    def __init__(
        self,
        *,
        repository: Repository,
        vector_store: VectorStore,
        llm: OpenWebUIClient,
        system_prompt: str,
        recent_history_limit: int,
        summary_every_n_messages: int,
        cross_channel_memory_limit: int,
        cross_channel_doc_limit: int,
        cross_channel_min_score: float,
        context_router_enabled: bool,
        context_router_max_feed_messages: int,
        context_router_min_confidence: float,
        context_debug_json: bool = False,
    ):
        self.repository = repository
        self.vector_store = vector_store
        self.llm = llm
        self.system_prompt = system_prompt
        self.recent_history_limit = recent_history_limit
        self.summary_every_n_messages = summary_every_n_messages
        self.cross_channel_memory_limit = cross_channel_memory_limit
        self.cross_channel_doc_limit = cross_channel_doc_limit
        self.cross_channel_min_score = cross_channel_min_score
        self.context_router_enabled = context_router_enabled
        self.context_router_max_feed_messages = context_router_max_feed_messages
        self.context_router_min_confidence = context_router_min_confidence
        self.context_debug_json = context_debug_json
        self.graph = self._compile()

    def invoke(self, request: dict[str, Any]) -> AgentState:
        return self.graph.invoke(request)

    def stream(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        state = self._prepare_generation_state(request)
        if is_telegram_command(state["text"]):
            state.update(self.generate_response(state))
            self.save_assistant_message(state)
            yield {
                "type": "done",
                "reply": state["reply"],
                "conversation_id": str(state["conversation"]["id"]),
                "sources": state.get("sources", []),
            }
            return

        reply_parts: list[str] = []
        sources: list[dict[str, str | None]] = []
        messages = self._build_prompt_messages(state)
        for event in self.llm.stream_chat(messages):
            if event.get("type") == "delta":
                delta = str(event.get("content") or "")
                if delta:
                    reply_parts.append(delta)
                    yield {"type": "delta", "content": delta}
            elif event.get("type") == "done":
                final_content = str(event.get("content") or "")
                if final_content and not reply_parts:
                    reply_parts.append(final_content)
                event_sources = event.get("sources")
                if isinstance(event_sources, list):
                    sources = event_sources

        state["reply"] = "".join(reply_parts)
        state["sources"] = sources
        self.maybe_extract_memory(state)
        self.maybe_update_summary(state)
        self.save_assistant_message(state)
        yield {
            "type": "done",
            "reply": state["reply"],
            "conversation_id": str(state["conversation"]["id"]),
            "sources": sources,
        }

    def _compile(self):
        graph = StateGraph(AgentState)
        graph.add_node("load_or_create_user", self.load_or_create_user)
        graph.add_node("load_or_create_conversation", self.load_or_create_conversation)
        graph.add_node("save_user_message", self.save_user_message)
        graph.add_node("choose_context_channels", self.choose_context_channels)
        graph.add_node("retrieve_context_messages", self.retrieve_context_messages)
        graph.add_node("retrieve_memories", self.retrieve_memories)
        graph.add_node("retrieve_documents", self.retrieve_documents)
        graph.add_node("load_summary", self.load_summary)
        graph.add_node("load_recent_history", self.load_recent_history)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("maybe_extract_memory", self.maybe_extract_memory)
        graph.add_node("maybe_update_summary", self.maybe_update_summary)
        graph.add_node("save_assistant_message", self.save_assistant_message)

        graph.add_edge(START, "load_or_create_user")
        graph.add_edge("load_or_create_user", "load_or_create_conversation")
        graph.add_edge("load_or_create_conversation", "save_user_message")
        graph.add_edge("save_user_message", "retrieve_memories")
        graph.add_edge("retrieve_memories", "retrieve_documents")
        graph.add_edge("retrieve_documents", "load_summary")
        graph.add_edge("load_summary", "load_recent_history")
        graph.add_edge("load_recent_history", "choose_context_channels")
        graph.add_edge("choose_context_channels", "retrieve_context_messages")
        graph.add_edge("retrieve_context_messages", "generate_response")
        graph.add_edge("generate_response", "maybe_extract_memory")
        graph.add_edge("maybe_extract_memory", "maybe_update_summary")
        graph.add_edge("maybe_update_summary", "save_assistant_message")
        graph.add_edge("save_assistant_message", END)
        return graph.compile()

    def _prepare_generation_state(self, request: dict[str, Any]) -> AgentState:
        state: AgentState = dict(request)
        for step in (
            self.load_or_create_user,
            self.load_or_create_conversation,
            self.save_user_message,
            self.retrieve_memories,
            self.retrieve_documents,
            self.load_summary,
            self.load_recent_history,
            self.choose_context_channels,
            self.retrieve_context_messages,
        ):
            state.update(step(state))
        return state

    def load_or_create_user(self, state: AgentState) -> AgentState:
        user = self.repository.upsert_user(
            telegram_user_id=state["telegram_user_id"],
            username=state.get("username"),
            first_name=state.get("first_name"),
        )
        return {"user": user}

    def load_or_create_conversation(self, state: AgentState) -> AgentState:
        conversation = self.repository.upsert_conversation(
            user_id=state["user"]["id"],
            telegram_chat_id=state["telegram_chat_id"],
            channel=state.get("channel") or PRIVATE_CHANNEL,
        )
        return {"conversation": conversation}

    def save_user_message(self, state: AgentState) -> AgentState:
        message = self.repository.save_message(
            conversation_id=state["conversation"]["id"],
            role="user",
            content=state["text"],
            telegram_message_id=state.get("telegram_message_id"),
        )
        return {"user_message": message}

    def choose_context_channels(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]) or not self.context_router_enabled:
            self._debug_json(
                "context_router_skipped",
                {
                    "reason": "command" if is_telegram_command(state["text"]) else "disabled",
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "channel": state.get("channel"),
                    "text": compact_text(state.get("text")),
                },
            )
            return {"context_channels": [], "context_router_rationale": None}
        prompt = [
            {
                "role": "system",
                "content": (
                    "You decide whether passive notification feed context is useful for answering a private chat. "
                    "Available feeds are domotics, unraid, and plex. Return strict JSON only with keys "
                    "channels, confidence, and rationale. channels must be an array containing only domotics, "
                    "unraid, and plex. Use an empty array when no feed is needed. Do not wrap the JSON in Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Latest user message:\n{state['text']}\n\n"
                    f"Private chat context already available:\n{self._router_private_context(state)}"
                ),
            },
        ]
        self._debug_json(
            "context_router_request",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "channel": state.get("channel"),
                "available_channels": sorted(PASSIVE_CONTEXT_CHANNELS),
                "text": compact_text(state.get("text")),
                "summary_present": bool(state.get("summary")),
                "recent_history_count": len(state.get("recent_messages", [])),
                "private_memory_count": len(state.get("memories", [])),
                "private_document_count": len(state.get("documents", [])),
            },
        )
        raw_content = None
        try:
            result = self.llm.chat(prompt, use_web_search=False)
            raw_content = result.content
            parsed = parse_llm_json_object(raw_content)
        except Exception as exc:
            LOGGER.info("Context router produced no valid JSON")
            self._debug_json(
                "context_router_parse_failed",
                {
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "db_user_id": str(state.get("user", {}).get("id")),
                    "error": type(exc).__name__,
                    "raw_content": compact_text(raw_content, limit=1200),
                },
            )
            return {"context_channels": [], "context_router_rationale": None}

        raw_channels = parsed.get("channels", [])
        if isinstance(raw_channels, str):
            raw_channels = [raw_channels]
        channels = [
            str(channel).lower()
            for channel in raw_channels
            if str(channel).lower() in PASSIVE_CONTEXT_CHANNELS
        ] if isinstance(raw_channels, list) else []
        channels = list(dict.fromkeys(channels))
        try:
            confidence = float(parsed.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self.context_router_min_confidence:
            channels = []
        rationale = parsed.get("rationale")
        self._debug_json(
            "context_router_decision",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "raw_json": parsed,
                "selected_channels": channels,
                "confidence": confidence,
                "min_confidence": self.context_router_min_confidence,
                "rationale": str(rationale) if rationale else None,
            },
        )
        return {
            "context_channels": channels,
            "context_router_rationale": str(rationale) if rationale else None,
        }

    def retrieve_context_messages(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            self._debug_json(
                "context_retrieval_skipped",
                {
                    "reason": "command",
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "db_user_id": str(state.get("user", {}).get("id")),
                },
            )
            return {"context_messages": []}
        channels = state.get("context_channels", [])
        if not channels:
            self._debug_json(
                "context_retrieval_skipped",
                {
                    "reason": "no_selected_channels",
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "db_user_id": str(state.get("user", {}).get("id")),
                    "query": compact_text(state.get("text")),
                },
            )
            return {"context_messages": []}
        vector_messages = self.vector_store.search_context_messages(
            user_id=state["user"]["id"],
            channels=channels,
            query=state["text"],
            limit=self.context_router_max_feed_messages,
        )
        for message in vector_messages:
            message["retrieval_source"] = "semantic"

        remaining = max(0, self.context_router_max_feed_messages - len(vector_messages))
        recent_rows = self.repository.recent_context_messages(
            user_id=state["user"]["id"],
            channels=channels,
            limit=remaining,
        )
        recent_messages = [self._context_row_to_hit(row) for row in recent_rows]
        messages = self._merge_context_hits(vector_messages, recent_messages)
        self._debug_json(
            "context_retrieval_results",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "selected_channels": channels,
                "limit": self.context_router_max_feed_messages,
                "query": compact_text(state.get("text")),
                "count": len(messages),
                "semantic_count": len(vector_messages),
                "recent_count": len(recent_messages),
                "hits": [self._hit_debug(hit) for hit in messages],
            },
        )
        return {"context_messages": messages}

    def retrieve_memories(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"memories": []}
        memories = self.vector_store.search_memories(
            user_id=state["user"]["id"],
            query=state["text"],
            channels=[PRIVATE_CHANNEL],
        )
        return {"memories": memories}

    def retrieve_documents(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"documents": []}
        documents = self.vector_store.search_documents(
            user_id=state["user"]["id"],
            query=state["text"],
            channels=[PRIVATE_CHANNEL],
        )
        return {"documents": documents}

    def load_summary(self, state: AgentState) -> AgentState:
        summary = self.repository.latest_summary(conversation_id=state["conversation"]["id"])
        return {"summary": summary["summary"] if summary else None}

    def load_recent_history(self, state: AgentState) -> AgentState:
        messages = self.repository.recent_messages(
            conversation_id=state["conversation"]["id"],
            limit=self.recent_history_limit,
        )
        return {"recent_messages": messages}

    def generate_response(self, state: AgentState) -> AgentState:
        command = state["text"].strip().lower()
        if command == "/start":
            first_name = state.get("first_name") or "there"
            return {
                "reply": (
                    f"Hi {first_name}. Codee is online. Send me a message and "
                    "I will answer with memory and document context when available."
                ),
                "sources": [],
            }
        if command == "/help":
            return {
                "reply": help_reply(),
                "sources": [],
            }
        if is_telegram_command(state["text"]):
            return {
                "reply": "Command received. Send /help for available commands, or send a normal message to chat.",
                "sources": [],
            }
        messages = self._build_prompt_messages(state)
        result = self.llm.chat(messages)
        return {"reply": result.content, "sources": result.sources}

    def maybe_extract_memory(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {}
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract only durable user facts, preferences, project details, "
                    "or recurring instructions from the latest exchange. Return JSON "
                    "with a top-level 'memories' array. Each item must have content, "
                    "memory_type, and importance 1-5. Return an empty array if none."
                ),
            },
            {
                "role": "user",
                "content": f"User: {state['text']}\nAssistant: {state.get('reply', '')}",
            },
        ]
        try:
            result = self.llm.chat(prompt, use_web_search=False)
            parsed = parse_llm_json_object(result.content)
            memories = parsed.get("memories", [])
        except Exception:
            LOGGER.info("Memory extraction produced no valid JSON")
            return {}

        if not isinstance(memories, list):
            return {}
        for memory in memories[:5]:
            if not isinstance(memory, dict):
                continue
            content = str(memory.get("content") or "").strip()
            if not content:
                continue
            memory_type = str(memory.get("memory_type") or "fact")
            importance = int(memory.get("importance") or 1)
            point_id = self.vector_store.upsert_memory(
                user_id=state["user"]["id"],
                conversation_id=state["conversation"]["id"],
                channel=PRIVATE_CHANNEL,
                content=content,
                memory_type=memory_type,
                importance=max(1, min(5, importance)),
            )
            self.repository.save_memory(
                user_id=state["user"]["id"],
                conversation_id=state["conversation"]["id"],
                channel=PRIVATE_CHANNEL,
                content=content,
                memory_type=memory_type,
                qdrant_point_id=point_id,
                importance=max(1, min(5, importance)),
            )
        return {}

    def maybe_update_summary(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {}
        count = self.repository.message_count(conversation_id=state["conversation"]["id"])
        if count == 0 or count % self.summary_every_n_messages != 0:
            return {}
        transcript = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in state.get("recent_messages", [])
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "Update the conversation summary. Keep decisions, preferences, "
                    "open tasks, names, dates, and project facts. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous summary:\n{state.get('summary') or '(none)'}\n\n"
                    f"Recent transcript:\n{transcript}"
                ),
            },
        ]
        try:
            result = self.llm.chat(prompt, use_web_search=False)
        except Exception:
            LOGGER.exception("Summary generation failed")
            return {}
        self.repository.save_summary(
            conversation_id=state["conversation"]["id"],
            summary=result.content,
        )
        return {"summary": result.content}

    def save_assistant_message(self, state: AgentState) -> AgentState:
        self.repository.save_message(
            conversation_id=state["conversation"]["id"],
            role="assistant",
            content=state["reply"],
        )
        return {}

    def _build_prompt_messages(self, state: AgentState) -> list[dict[str, str]]:
        context_blocks = []
        if (
            state.get("summary")
            or state.get("memories")
            or state.get("documents")
            or state.get("context_messages")
            or state.get("recent_messages")
        ):
            context_blocks.append(
                "Context priority rules:\n"
                "- First inspect and use the provided private chat context, recent private transcript, and selected passive feed context.\n"
                "- Resolve follow-up references from the recent private transcript before answering or searching.\n"
                "- If the user asks about something in domotics, unraid, or plex, answer from the selected passive feed context when it contains the knowledge.\n"
                "- Do not replace anything found in the selected passive feed context with web search results, public websites, prior assistant guesses, or unrelated private memories.\n"
                "- Use web search only after the provided context and recent transcript do not contain the answer, and clearly say when the answer came from web search instead of stored context.\n"
            )
        if state.get("summary"):
            context_blocks.append(f"Private chat summary:\n{state['summary']}")

        memory_lines = [
            f"- {hit['payload'].get('content')}"
            for hit in state.get("memories", [])
            if hit.get("payload", {}).get("content")
        ]
        if memory_lines:
            context_blocks.append("Relevant private memories:\n" + "\n".join(memory_lines))

        document_lines = [
            f"- {hit['payload'].get('filename')}: {hit['payload'].get('content')}"
            for hit in state.get("documents", [])
            if hit.get("payload", {}).get("content")
        ]
        if document_lines:
            context_blocks.append("Relevant private documents:\n" + "\n".join(document_lines))

        feed_lines = [
            f"- [{hit['payload'].get('channel')}] {hit['payload'].get('content')}"
            for hit in state.get("context_messages", [])
            if hit.get("payload", {}).get("content") and hit.get("payload", {}).get("channel")
        ]
        if feed_lines:
            context_blocks.append("Selected passive feed context:\n" + "\n".join(feed_lines))

        system = self.system_prompt
        if context_blocks:
            system = system + "\n\n" + "\n\n".join(context_blocks)
        self._debug_json(
            "context_prompt_context",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "conversation_id": str(state.get("conversation", {}).get("id")),
                "summary_present": bool(state.get("summary")),
                "private_memory_count": len(state.get("memories", [])),
                "private_document_count": len(state.get("documents", [])),
                "selected_feed_channels": state.get("context_channels", []),
                "feed_context_count": len(state.get("context_messages", [])),
                "recent_history_count": len(state.get("recent_messages", [])),
                "context_sections": [
                    "context_priority_rules" if block.startswith("Context priority rules") else
                    "private_summary" if block.startswith("Private chat summary") else
                    "private_memories" if block.startswith("Relevant private memories") else
                    "private_documents" if block.startswith("Relevant private documents") else
                    "passive_feed_context" if block.startswith("Selected passive feed context") else
                    "unknown"
                    for block in context_blocks
                ],
                "feed_hits": [self._hit_debug(hit) for hit in state.get("context_messages", [])],
            },
        )

        messages = [{"role": "system", "content": system}]
        for message in state.get("recent_messages", []):
            if message["role"] in {"user", "assistant"}:
                messages.append({"role": message["role"], "content": message["content"]})
        return messages

    def _router_private_context(self, state: AgentState) -> str:
        sections = []
        if state.get("summary"):
            sections.append(f"Private summary:\n{compact_text(state['summary'], limit=900)}")

        recent_lines = []
        for message in state.get("recent_messages", [])[-8:]:
            role = message.get("role")
            content = compact_text(message.get("content"), limit=300)
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        if recent_lines:
            sections.append("Recent private transcript:\n" + "\n".join(recent_lines))

        memory_lines = [
            f"- {compact_text(hit.get('payload', {}).get('content'), limit=300)}"
            for hit in state.get("memories", [])[:5]
            if hit.get("payload", {}).get("content")
        ]
        if memory_lines:
            sections.append("Relevant private memories:\n" + "\n".join(memory_lines))

        document_lines = [
            (
                f"- {hit.get('payload', {}).get('filename') or 'document'}: "
                f"{compact_text(hit.get('payload', {}).get('content'), limit=300)}"
            )
            for hit in state.get("documents", [])[:5]
            if hit.get("payload", {}).get("content")
        ]
        if document_lines:
            sections.append("Relevant private documents:\n" + "\n".join(document_lines))

        return "\n\n".join(sections) if sections else "(none)"

    def _debug_json(self, event: str, payload: dict[str, Any]) -> None:
        if not self.context_debug_json:
            return
        LOGGER.info(
            "context_debug_json %s",
            json.dumps({"event": event, **payload}, default=str, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _hit_debug(hit: dict[str, Any]) -> dict[str, Any]:
        payload = hit.get("payload", {})
        return {
            "score": hit.get("score"),
            "retrieval_source": hit.get("retrieval_source"),
            "channel": payload.get("channel"),
            "source_type": payload.get("source_type"),
            "telegram_chat_id": payload.get("telegram_chat_id"),
            "telegram_message_id": payload.get("telegram_message_id"),
            "content": compact_text(payload.get("content")),
        }

    @staticmethod
    def _context_row_to_hit(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": None,
            "retrieval_source": "recent",
            "payload": {
                "source_type": "context_message",
                "channel": row.get("channel"),
                "telegram_chat_id": row.get("telegram_chat_id"),
                "telegram_message_id": row.get("telegram_message_id"),
                "content": row.get("content"),
                "created_at": row.get("created_at"),
            },
        }

    @staticmethod
    def _merge_context_hits(
        semantic_hits: list[dict[str, Any]],
        recent_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = []
        seen = set()
        for hit in [*semantic_hits, *recent_hits]:
            payload = hit.get("payload", {})
            if payload.get("telegram_message_id"):
                key = (
                    payload.get("channel"),
                    payload.get("telegram_chat_id"),
                    payload.get("telegram_message_id"),
                )
            else:
                key = (
                    payload.get("channel"),
                    payload.get("content"),
                )
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        return merged

    @staticmethod
    def _partition_hits(hits: list[dict[str, Any]], channel: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        same_channel_hits = []
        other_channel_hits = []
        for hit in hits:
            payload = hit.get("payload", {})
            if payload.get("channel") == channel:
                same_channel_hits.append(hit)
            else:
                other_channel_hits.append(hit)
        return same_channel_hits, other_channel_hits

    def _filter_cross_channel_hits(
        self,
        *,
        same_channel_hits: list[dict[str, Any]],
        other_channel_hits: list[dict[str, Any]],
        max_items: int,
    ) -> list[dict[str, Any]]:
        if not other_channel_hits or max_items <= 0:
            return []
        if len(same_channel_hits) < 2:
            return other_channel_hits[:max_items]
        return [
            hit
            for hit in other_channel_hits
            if float(hit.get("score") or 0.0) >= self.cross_channel_min_score
        ][:max_items]
