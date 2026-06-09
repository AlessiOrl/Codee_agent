import json
import logging
from typing import Any, Iterator, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.openwebui import LlmResult, OpenWebUIClient
from app.repository import Repository
from app.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)


def is_telegram_command(text: str) -> bool:
    return text.strip().startswith("/")


def help_reply() -> str:
    return (
        "Available commands:\n"
        "/start - show the welcome message\n"
        "/help - show this command list\n"
        "/summary - show the current chat summary\n"
        "/forget - remove all stored context for your user\n"
        "/forget <days> - remove stored context from the last N days\n\n"
        "Send a normal text message to chat with Codee."
    )


class AgentState(TypedDict, total=False):
    telegram_user_id: str
    telegram_chat_id: str
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
    ):
        self.repository = repository
        self.vector_store = vector_store
        self.llm = llm
        self.system_prompt = system_prompt
        self.recent_history_limit = recent_history_limit
        self.summary_every_n_messages = summary_every_n_messages
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
        graph.add_edge("load_recent_history", "generate_response")
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

    def retrieve_memories(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"memories": []}
        memories = self.vector_store.search_memories(
            user_id=state["user"]["id"],
            query=state["text"],
        )
        return {"memories": memories}

    def retrieve_documents(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"documents": []}
        documents = self.vector_store.search_documents(
            user_id=state["user"]["id"],
            query=state["text"],
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
            result = self.llm.chat(prompt)
            parsed = json.loads(result.content)
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
                content=content,
                memory_type=memory_type,
                importance=max(1, min(5, importance)),
            )
            self.repository.save_memory(
                user_id=state["user"]["id"],
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
            result = self.llm.chat(prompt)
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
        if state.get("summary"):
            context_blocks.append(f"Conversation summary:\n{state['summary']}")
        memory_lines = [
            f"- {hit['payload'].get('content')}"
            for hit in state.get("memories", [])
            if hit.get("payload", {}).get("content")
        ]
        if memory_lines:
            context_blocks.append("Relevant memories:\n" + "\n".join(memory_lines))
        document_lines = [
            f"- {hit['payload'].get('filename')}: {hit['payload'].get('content')}"
            for hit in state.get("documents", [])
            if hit.get("payload", {}).get("content")
        ]
        if document_lines:
            context_blocks.append("Relevant documents:\n" + "\n".join(document_lines))

        system = self.system_prompt
        if context_blocks:
            system = system + "\n\n" + "\n\n".join(context_blocks)

        messages = [{"role": "system", "content": system}]
        for message in state.get("recent_messages", []):
            if message["role"] in {"user", "assistant"}:
                messages.append({"role": message["role"], "content": message["content"]})
        return messages
