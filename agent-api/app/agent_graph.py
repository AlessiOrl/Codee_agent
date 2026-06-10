import json
import logging
import re
from typing import Any, Iterator, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.openwebui import LlmResult, OpenWebUIClient
from app.repository import Repository
from app.retrieval import hybrid_rerank, row_to_hit
from app.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)
PASSIVE_CONTEXT_CHANNELS = {"domotics", "unraid", "plex"}
PRIVATE_CHANNEL = "private"
FEED_CONTEXT_TERMS = {
    "alert",
    "alerts",
    "chat",
    "event",
    "events",
    "feed",
    "feeds",
    "log",
    "logs",
    "message",
    "messages",
    "notification",
    "notifications",
    "status",
    "summary",
    "summarize",
}


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
    request_plan: dict[str, Any]
    subtasks: list[dict[str, Any]]
    retrieval_groups: list[dict[str, Any]]
    external_results: list[dict[str, Any]]
    needs_final_generation: bool
    clarification_question: str | None
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
        context_gate_min_confidence: float | None = None,
        context_gate_ask_on_ambiguous: bool = True,
        external_fact_web_search_enabled: bool = True,
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
        self.context_gate_min_confidence = (
            context_gate_min_confidence
            if context_gate_min_confidence is not None
            else context_router_min_confidence
        )
        self.context_gate_ask_on_ambiguous = context_gate_ask_on_ambiguous
        self.external_fact_web_search_enabled = external_fact_web_search_enabled
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

        if state.get("clarification_question") or state.get("needs_final_generation"):
            state.update(self.generate_response(state))
            self.maybe_extract_memory(state)
            self.maybe_update_summary(state)
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
        graph.add_node("plan_request", self.plan_request)
        graph.add_node("gate_context_per_subtask", self.gate_context_per_subtask)
        graph.add_node("retrieve_per_subtask", self.retrieve_per_subtask)
        graph.add_node("resolve_external_subtasks", self.resolve_external_subtasks)
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
        graph.add_edge("load_recent_history", "plan_request")
        graph.add_edge("plan_request", "gate_context_per_subtask")
        graph.add_edge("gate_context_per_subtask", "retrieve_per_subtask")
        graph.add_edge("retrieve_per_subtask", "resolve_external_subtasks")
        graph.add_edge("resolve_external_subtasks", "generate_response")
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
            self.plan_request,
            self.gate_context_per_subtask,
            self.retrieve_per_subtask,
            self.resolve_external_subtasks,
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

    def plan_request(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            self._debug_json(
                "request_planner_skipped",
                {
                    "reason": "command",
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "channel": state.get("channel"),
                    "text": compact_text(state.get("text")),
                },
            )
            subtask = self._subtask(
                task_id="task-1",
                description=state["text"],
                needs_feed_context=False,
                selected_channel=None,
                reason="command",
            )
            return {
                "request_plan": {"mode": "command", "risk_reasons": []},
                "subtasks": [subtask],
                "needs_final_generation": False,
            }

        text = state["text"]
        explicit_channels = self._explicit_feed_channels(text)
        all_feeds_requested = self._all_feeds_requested(text)
        if all_feeds_requested:
            subtasks = self._all_feed_subtasks(text)
            if self._has_non_feed_work(text):
                description = self._external_fact_description(text)
                subtasks = [
                    self._subtask(
                        task_id="external-task-1",
                        description=description,
                        needs_feed_context=False,
                        selected_channel=None,
                        needs_external_context=self._needs_external_context(description),
                        reason="mixed_request_external_fact",
                    ),
                    *subtasks,
                ]
                if subtasks[0].get("needs_external_context"):
                    for subtask in subtasks[1:]:
                        subtask["depends_on"] = [subtasks[0]["id"]]
            return {
                "request_plan": {
                    "mode": "generic_all_feeds",
                    "channels": sorted(PASSIVE_CONTEXT_CHANNELS),
                    "risk_reasons": ["feed_context", "multi_part"]
                    + (["external_context"] if any(subtask.get("needs_external_context") for subtask in subtasks) else []),
                },
                "subtasks": subtasks,
                "needs_final_generation": True,
            }

        if explicit_channels:
            subtasks = [
                self._subtask(
                    task_id=f"feed-{channel}",
                    description=text,
                    needs_feed_context=True,
                    selected_channel=channel,
                    reason="explicit_channel",
                )
                for channel in explicit_channels
            ]
            return {
                "request_plan": {
                    "mode": "explicit_feed",
                    "channels": explicit_channels,
                    "risk_reasons": ["feed_context"] + (["multi_part"] if len(subtasks) > 1 else []),
                },
                "subtasks": subtasks,
                "needs_final_generation": True,
            }

        if self._looks_multi_intent(text) and self.context_router_enabled:
            planned = self._llm_plan_request(state)
            if planned:
                return planned

        if self._looks_feed_related(text):
            subtask = self._subtask(
                task_id="feed-task-1",
                description=text,
                needs_feed_context=True,
                selected_channel=None,
                reason="feed_related_without_explicit_channel",
            )
            return {
                "request_plan": {"mode": "feed_gate_needed", "risk_reasons": ["feed_context"]},
                "subtasks": [subtask],
                "needs_final_generation": True,
            }

        needs_external_context = self._needs_external_context(text)
        subtask = self._subtask(
            task_id="task-1",
            description=text,
            needs_feed_context=False,
            selected_channel=None,
            needs_external_context=needs_external_context,
            reason="simple_private",
        )
        return {
            "request_plan": {
                "mode": "external_context" if needs_external_context else "direct",
                "risk_reasons": ["external_context"] if needs_external_context else [],
            },
            "subtasks": [subtask],
            "needs_final_generation": needs_external_context,
        }

    def gate_context_per_subtask(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {
                "context_channels": [],
                "context_router_rationale": None,
                "clarification_question": None,
            }

        gated_subtasks = []
        selected_channels = []
        rationales = []
        clarification_question = None

        for subtask in state.get("subtasks", []):
            gated = dict(subtask)
            if not gated.get("needs_feed_context"):
                gated["context_status"] = "not_needed"
                gated_subtasks.append(gated)
                continue

            selected = gated.get("selected_channel")
            if selected in PASSIVE_CONTEXT_CHANNELS:
                gated.update(
                    {
                        "context_status": "selected",
                        "selected_channel": selected,
                        "confidence": max(float(gated.get("confidence") or 0.0), 1.0),
                    }
                )
                selected_channels.append(selected)
                gated_subtasks.append(gated)
                continue

            if not self.context_router_enabled:
                question = self._clarifying_question(gated) if self.context_gate_ask_on_ambiguous else None
                gated.update(
                    {
                        "context_status": "ambiguous",
                        "ambiguous": True,
                        "clarifying_question": question,
                    }
                )
                clarification_question = clarification_question or question
                gated_subtasks.append(gated)
                continue

            decision = self._llm_gate_subtask(state, gated)
            channel = decision.get("selected_channel")
            confidence = self._bounded_float(
                decision.get("confidence"),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            )
            ambiguous = bool(decision.get("ambiguous"))
            rationale = str(decision.get("rationale") or "")
            if rationale:
                rationales.append(rationale)

            if (
                channel in PASSIVE_CONTEXT_CHANNELS
                and confidence >= self.context_gate_min_confidence
                and not ambiguous
            ):
                gated.update(
                    {
                        "context_status": "selected",
                        "selected_channel": channel,
                        "confidence": confidence,
                        "rationale": rationale,
                    }
                )
                selected_channels.append(channel)
            else:
                question = None
                if self.context_gate_ask_on_ambiguous:
                    question = str(decision.get("clarifying_question") or "") or self._clarifying_question(gated)
                gated.update(
                    {
                        "context_status": "ambiguous",
                        "selected_channel": None,
                        "confidence": confidence,
                        "ambiguous": True,
                        "clarifying_question": question,
                        "rationale": rationale,
                    }
                )
                clarification_question = clarification_question or question
            gated_subtasks.append(gated)

        selected_channels = list(dict.fromkeys(selected_channels))
        self._debug_json(
            "context_gate_decision",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "selected_channels": selected_channels,
                "clarification_question": clarification_question,
                "subtasks": gated_subtasks,
            },
        )
        return {
            "subtasks": gated_subtasks,
            "context_channels": selected_channels,
            "context_router_rationale": "\n".join(rationales) if rationales else None,
            "clarification_question": clarification_question,
            "needs_final_generation": bool(state.get("needs_final_generation")) or len(selected_channels) > 0,
        }

    def retrieve_per_subtask(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"retrieval_groups": [], "context_messages": []}
        if state.get("clarification_question"):
            self._debug_json(
                "context_retrieval_skipped",
                {
                    "reason": "clarification_needed",
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "db_user_id": str(state.get("user", {}).get("id")),
                    "query": compact_text(state.get("text")),
                },
            )
            return {"retrieval_groups": [], "context_messages": []}

        retrieval_groups = []
        flattened_messages = []
        for subtask in state.get("subtasks", []):
            channel = subtask.get("selected_channel")
            if not subtask.get("needs_feed_context") or channel not in PASSIVE_CONTEXT_CHANNELS:
                continue

            messages = self._retrieve_context_for_channel(
                user_id=state["user"]["id"],
                channel=channel,
                query=str(subtask.get("description") or state["text"]),
            )
            group = {
                "subtask_id": subtask.get("id"),
                "description": subtask.get("description"),
                "channel": channel,
                "messages": messages,
            }
            retrieval_groups.append(group)
            flattened_messages.extend(messages)

        self._debug_json(
            "context_retrieval_results",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "selected_channels": state.get("context_channels", []),
                "query": compact_text(state.get("text")),
                "group_count": len(retrieval_groups),
                "count": len(flattened_messages),
                "groups": [
                    {
                        "subtask_id": group.get("subtask_id"),
                        "channel": group.get("channel"),
                        "count": len(group.get("messages", [])),
                        "hits": [self._hit_debug(hit) for hit in group.get("messages", [])],
                    }
                    for group in retrieval_groups
                ],
            },
        )
        return {
            "retrieval_groups": retrieval_groups,
            "context_messages": flattened_messages,
            "needs_final_generation": bool(state.get("needs_final_generation")) or bool(flattened_messages),
        }

    def resolve_external_subtasks(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]) or state.get("clarification_question"):
            return {"external_results": []}

        subtasks = [dict(subtask) for subtask in state.get("subtasks", [])]
        external_results = []
        for subtask in subtasks:
            if not subtask.get("needs_external_context"):
                continue

            if not self.external_fact_web_search_enabled:
                subtask["external_resolution_status"] = "disabled"
                external_results.append(
                    {
                        "subtask_id": subtask.get("id"),
                        "description": subtask.get("description"),
                        "status": "disabled",
                        "content": "External fact web search is disabled.",
                        "sources": [],
                    }
                )
                continue

            prompt = self._external_resolution_messages(state, subtask, external_results)
            try:
                result = self.llm.chat(prompt, use_web_search=True)
                content = result.content.strip()
                status = "resolved_with_sources" if result.sources else "resolved_without_sources"
                if not content:
                    status = "unavailable"
                    content = "No external result was returned."
                subtask["external_resolution_status"] = status
                external_results.append(
                    {
                        "subtask_id": subtask.get("id"),
                        "description": subtask.get("description"),
                        "status": status,
                        "content": content,
                        "sources": result.sources,
                    }
                )
            except Exception as exc:
                LOGGER.info("External fact resolver failed")
                subtask["external_resolution_status"] = "unavailable"
                external_results.append(
                    {
                        "subtask_id": subtask.get("id"),
                        "description": subtask.get("description"),
                        "status": "unavailable",
                        "content": f"External fact resolution failed: {type(exc).__name__}.",
                        "sources": [],
                    }
                )

        if not external_results:
            return {"external_results": [], "subtasks": subtasks}

        self._debug_json(
            "external_resolution_results",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "result_count": len(external_results),
                "results": [
                    {
                        "subtask_id": result.get("subtask_id"),
                        "status": result.get("status"),
                        "source_count": len(result.get("sources", [])),
                        "content": compact_text(result.get("content")),
                    }
                    for result in external_results
                ],
            },
        )
        return {
            "external_results": external_results,
            "subtasks": subtasks,
            "needs_final_generation": True,
        }

    def choose_context_channels(self, state: AgentState) -> AgentState:
        state.update(self.plan_request(state))
        state.update(self.gate_context_per_subtask(state))
        return {
            "context_channels": state.get("context_channels", []),
            "context_router_rationale": state.get("context_router_rationale"),
            "subtasks": state.get("subtasks", []),
            "request_plan": state.get("request_plan", {}),
            "needs_final_generation": state.get("needs_final_generation", False),
            "clarification_question": state.get("clarification_question"),
        }

    def retrieve_context_messages(self, state: AgentState) -> AgentState:
        return self.retrieve_per_subtask(state)

    def _llm_gate_subtask(self, state: AgentState, subtask: dict[str, Any]) -> dict[str, Any]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You decide whether passive notification feed context is useful for answering a private chat. "
                    "This is a context gate: feed messages only mean the right thing inside their own feed. "
                    "Available feeds are domotics, unraid, and plex. Return strict JSON only with keys "
                    "selected_channel, confidence, ambiguous, clarifying_question, and rationale. "
                    "selected_channel must be one of domotics, unraid, plex, or null. Select exactly one feed only "
                    "when the request clearly belongs in that feed. If multiple feeds are plausible or the context "
                    "is unclear, set ambiguous=true and provide a clarifying_question. Do not wrap JSON in Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Latest user message:\n{state['text']}\n\n"
                    f"Sub-request to gate:\n{subtask.get('description')}\n\n"
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
            LOGGER.info("Context gate produced no valid JSON")
            self._debug_json(
                "context_gate_parse_failed",
                {
                    "telegram_user_id": state.get("telegram_user_id"),
                    "telegram_chat_id": state.get("telegram_chat_id"),
                    "db_user_id": str(state.get("user", {}).get("id")),
                    "error": type(exc).__name__,
                    "raw_content": compact_text(raw_content, limit=1200),
                },
            )
            return {
                "selected_channel": None,
                "confidence": 0.0,
                "ambiguous": True,
                "clarifying_question": self._clarifying_question(subtask),
                "rationale": "Context gate did not return valid JSON.",
            }

        raw_channels = parsed.get("channels", [])
        if isinstance(raw_channels, str):
            raw_channels = [raw_channels]
        channels = [
            str(channel).lower()
            for channel in raw_channels
            if str(channel).lower() in PASSIVE_CONTEXT_CHANNELS
        ] if isinstance(raw_channels, list) else []
        channels = list(dict.fromkeys(channels))
        selected_channel = str(parsed.get("selected_channel") or "").lower()
        if selected_channel not in PASSIVE_CONTEXT_CHANNELS and len(channels) == 1:
            selected_channel = channels[0]
        if selected_channel not in PASSIVE_CONTEXT_CHANNELS:
            selected_channel = None
        try:
            confidence = float(parsed.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        ambiguous = bool(parsed.get("ambiguous")) or len(channels) > 1
        rationale = parsed.get("rationale")
        self._debug_json(
            "context_gate_llm_decision",
            {
                "telegram_user_id": state.get("telegram_user_id"),
                "telegram_chat_id": state.get("telegram_chat_id"),
                "db_user_id": str(state.get("user", {}).get("id")),
                "raw_json": parsed,
                "selected_channel": selected_channel,
                "confidence": confidence,
                "min_confidence": self.context_gate_min_confidence,
                "ambiguous": ambiguous,
                "rationale": str(rationale) if rationale else None,
            },
        )
        return {
            "selected_channel": selected_channel,
            "confidence": confidence,
            "ambiguous": ambiguous,
            "clarifying_question": parsed.get("clarifying_question"),
            "rationale": str(rationale) if rationale else None,
        }

    def _retrieve_context_for_channel(self, *, user_id: UUID, channel: str, query: str) -> list[dict[str, Any]]:
        limit = self.context_router_max_feed_messages
        candidate_limit = max(limit * 3, limit)
        vector_messages = self.vector_store.search_context_messages(
            user_id=user_id,
            channels=[channel],
            query=query,
            limit=candidate_limit,
        )
        for message in vector_messages:
            message["retrieval_source"] = "semantic"
            message["retrieval_sources"] = ["semantic"]
            message.setdefault("payload", {}).setdefault("source_type", "context_message")

        keyword_messages = self._keyword_context_hits(
            user_id=user_id,
            channels=[channel],
            query=query,
            limit=candidate_limit,
        )
        recent_rows = self.repository.recent_context_messages(
            user_id=user_id,
            channels=[channel],
            limit=limit,
        )
        recent_messages = [self._context_row_to_hit(row) for row in recent_rows]
        return hybrid_rerank(
            query=query,
            hits=[*vector_messages, *keyword_messages, *recent_messages],
            limit=limit,
            allowed_channels={channel},
            allowed_source_types={"context_message"},
        )

    def retrieve_memories(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"memories": []}
        limit = 8
        candidate_limit = limit * 3
        memories = self.vector_store.search_memories(
            user_id=state["user"]["id"],
            query=state["text"],
            channels=[PRIVATE_CHANNEL],
            limit=candidate_limit,
        )
        for memory in memories:
            memory["retrieval_source"] = "semantic"
            memory["retrieval_sources"] = ["semantic"]
        keyword_memories = self._keyword_memory_hits(
            user_id=state["user"]["id"],
            query=state["text"],
            channels=[PRIVATE_CHANNEL],
            limit=candidate_limit,
        )
        memories = hybrid_rerank(
            query=state["text"],
            hits=[*memories, *keyword_memories],
            limit=limit,
            allowed_channels={PRIVATE_CHANNEL},
        )
        return {"memories": memories}

    def retrieve_documents(self, state: AgentState) -> AgentState:
        if is_telegram_command(state["text"]):
            return {"documents": []}
        limit = 8
        candidate_limit = limit * 3
        documents = self.vector_store.search_documents(
            user_id=state["user"]["id"],
            query=state["text"],
            channels=[PRIVATE_CHANNEL],
            limit=candidate_limit,
        )
        for document in documents:
            document["retrieval_source"] = "semantic"
            document["retrieval_sources"] = ["semantic"]
            document.setdefault("payload", {}).setdefault("source_type", "document_chunk")
        keyword_documents = self._keyword_document_hits(
            user_id=state["user"]["id"],
            query=state["text"],
            channels=[PRIVATE_CHANNEL],
            limit=candidate_limit,
        )
        documents = hybrid_rerank(
            query=state["text"],
            hits=[*documents, *keyword_documents],
            limit=limit,
            allowed_channels={PRIVATE_CHANNEL},
            allowed_source_types={"document_chunk"},
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
        if state.get("clarification_question"):
            return {"reply": str(state["clarification_question"]), "sources": []}
        messages = self._build_prompt_messages(state)
        result = self.llm.chat(messages)
        return {"reply": result.content, "sources": self._combined_sources(state, result.sources)}

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
            confidence = self._bounded_float(memory.get("confidence"), default=1.0, minimum=0.0, maximum=1.0)
            importance = max(1, min(5, importance))
            if self._memory_exists(
                user_id=state["user"]["id"],
                channel=PRIVATE_CHANNEL,
                content=content,
            ):
                continue
            source_text = f"User: {state['text']}\nAssistant: {state.get('reply', '')}"
            point_id = self.vector_store.upsert_memory(
                user_id=state["user"]["id"],
                conversation_id=state["conversation"]["id"],
                channel=PRIVATE_CHANNEL,
                content=content,
                memory_type=memory_type,
                importance=importance,
                confidence=confidence,
                source_user_message_id=state.get("user_message", {}).get("id"),
                source_text=source_text,
            )
            self.repository.save_memory(
                user_id=state["user"]["id"],
                conversation_id=state["conversation"]["id"],
                channel=PRIVATE_CHANNEL,
                content=content,
                memory_type=memory_type,
                qdrant_point_id=point_id,
                importance=importance,
                confidence=confidence,
                source_user_message_id=state.get("user_message", {}).get("id"),
                source_text=source_text,
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

    def _llm_plan_request(self, state: AgentState) -> AgentState | None:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Decompose a private assistant request only when it contains multiple distinct tasks. "
                    "Return strict JSON with keys subtasks and risk_reasons. Each subtask must have id, "
                    "description, needs_feed_context, needs_external_context, selected_channel, and depends_on. "
                    "selected_channel may be domotics, unraid, plex, private, or null. Mark "
                    "needs_external_context true for outside-world facts that need current or public knowledge. "
                    "Treat generic references to feed chats as a request to check all configured passive feeds "
                    "separately. Preserve dependencies between subtasks: if one part establishes a value, event, "
                    "or condition that another part uses, keep those as separate ordered subtasks and reference "
                    "the earlier id in depends_on. Do not invent feed context when the request is ordinary chat."
                ),
            },
            {"role": "user", "content": state["text"]},
        ]
        try:
            result = self.llm.chat(prompt, use_web_search=False)
            parsed = parse_llm_json_object(result.content)
        except Exception:
            LOGGER.info("Request planner produced no valid JSON")
            return None

        raw_subtasks = parsed.get("subtasks")
        if not isinstance(raw_subtasks, list):
            return None
        subtasks = []
        for index, raw in enumerate(raw_subtasks[:5], start=1):
            if not isinstance(raw, dict):
                continue
            channel = str(raw.get("selected_channel") or "").lower()
            if channel not in PASSIVE_CONTEXT_CHANNELS:
                channel = None
            needs_feed_context = bool(raw.get("needs_feed_context")) or channel in PASSIVE_CONTEXT_CHANNELS
            description = str(raw.get("description") or state["text"])
            depends_on = raw.get("depends_on")
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            elif not isinstance(depends_on, list):
                depends_on = []
            subtasks.append(
                self._subtask(
                    task_id=str(raw.get("id") or f"task-{index}"),
                    description=description,
                    needs_feed_context=needs_feed_context,
                    selected_channel=channel,
                    needs_external_context=bool(raw.get("needs_external_context"))
                    or (not needs_feed_context and self._needs_external_context(description)),
                    depends_on=[str(item) for item in depends_on if str(item)],
                    reason="llm_planner",
                    confidence=self._bounded_float(
                        raw.get("confidence"),
                        default=1.0 if channel else 0.0,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                )
            )
        if not subtasks:
            return None
        risk_reasons = parsed.get("risk_reasons")
        if not isinstance(risk_reasons, list):
            risk_reasons = []
        needs_final_generation = (
            len(subtasks) > 1
            or any(subtask.get("needs_feed_context") for subtask in subtasks)
            or any(subtask.get("needs_external_context") for subtask in subtasks)
            or any(str(reason) for reason in risk_reasons)
        )
        return {
            "request_plan": {
                "mode": "llm_planned",
                "risk_reasons": [str(reason) for reason in risk_reasons if str(reason)],
            },
            "subtasks": subtasks,
            "needs_final_generation": needs_final_generation,
        }

    @staticmethod
    def _subtask(
        *,
        task_id: str,
        description: str,
        needs_feed_context: bool,
        selected_channel: str | None,
        reason: str,
        confidence: float = 1.0,
        needs_external_context: bool = False,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": task_id,
            "description": description,
            "needs_feed_context": needs_feed_context,
            "needs_external_context": needs_external_context,
            "selected_channel": selected_channel,
            "depends_on": depends_on or [],
            "reason": reason,
            "confidence": confidence,
        }

    @staticmethod
    def _explicit_feed_channels(text: str) -> list[str]:
        lowered = text.lower()
        channels = [
            channel
            for channel in sorted(PASSIVE_CONTEXT_CHANNELS)
            if re.search(rf"\b{re.escape(channel)}\b", lowered)
        ]
        return list(dict.fromkeys(channels))

    @staticmethod
    def _all_feeds_requested(text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(r"\b(each|all|every)\s+(feed|feeds|chat|chats)\b", lowered)
            or re.search(r"\b(feed|feeds|chat|chats)\s+(each|all|every)\b", lowered)
            or re.search(r"\b(the\s+)?feed\s+chats?\b", lowered)
            or re.search(r"\bthe\s+feeds?\b", lowered)
            or re.search(r"\bnotification\s+chats?\b", lowered)
            or re.search(r"\bstored\s+feeds?\b", lowered)
        )

    @staticmethod
    def _all_feed_subtasks(text: str) -> list[dict[str, Any]]:
        return [
            AgentGraph._subtask(
                task_id=f"feed-{channel}",
                description=f"Check {channel} feed evidence for this request: {text}",
                needs_feed_context=True,
                selected_channel=channel,
                reason="generic_all_feeds",
            )
            for channel in sorted(PASSIVE_CONTEXT_CHANNELS)
        ]

    @staticmethod
    def _has_non_feed_work(text: str) -> bool:
        lowered = text.lower()
        generic_feed = re.search(
            r"\b((the\s+)?feed\s+chats?|the\s+feeds?|notification\s+chats?|stored\s+feeds?)\b",
            lowered,
        )
        if generic_feed:
            prefix = lowered[: generic_feed.start()]
            return bool(
                re.search(r"\b(what|who|which|when|where|find|tell|lookup|look up|check|compare|is|are)\b", prefix)
                and len(prefix.strip(" ,.;")) > 8
            )
        return bool(
            re.search(r"\b(what|who|which|when|where|find|tell|lookup|look up)\b", lowered)
            and re.search(r"\b(and|then|also)\b", lowered)
        )

    @staticmethod
    def _external_fact_description(text: str) -> str:
        parts = re.split(r"\b(?:and|then|also)\b", text, maxsplit=1, flags=re.IGNORECASE)
        description = parts[0].strip(" ,.;")
        if description:
            return description
        return "Resolve the non-feed factual part of the user request."

    @staticmethod
    def _needs_external_context(text: str) -> bool:
        lowered = text.lower()
        current_terms = (
            r"\b(latest|current|currently|today|tonight|yesterday|tomorrow|"
            r"recent|newest|live|breaking|news|price|stock|weather|score|"
            r"winner|won|grand prix|exchange rate)\b"
        )
        return bool(re.search(current_terms, lowered))

    def _external_resolution_messages(
        self,
        state: AgentState,
        subtask: dict[str, Any],
        prior_results: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        prior_text = "\n".join(
            f"- {result.get('subtask_id')}: {compact_text(result.get('content'), limit=500)}"
            for result in prior_results
            if result.get("content")
        )
        user_content = (
            f"Original user request:\n{state['text']}\n\n"
            f"Subtask to resolve:\n{subtask.get('description')}\n\n"
            f"Depends on subtasks:\n{json.dumps(subtask.get('depends_on') or [], default=str)}\n\n"
            f"Earlier external results:\n{prior_text or '(none)'}"
        )
        return [
            {
                "role": "system",
                "content": (
                    "Resolve only the outside-world fact needed for this subtask. Use web search when available. "
                    "Return a concise answer with the smallest useful value or fact needed by later subtasks. "
                    "Do not use passive feed context here. If the current/public fact cannot be verified, say that "
                    "plainly and do not guess."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _looks_feed_related(text: str) -> bool:
        tokens = set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))
        return bool(tokens & FEED_CONTEXT_TERMS)

    @staticmethod
    def _looks_multi_intent(text: str) -> bool:
        lowered = text.lower()
        action_hits = len(re.findall(r"\b(check|summarize|compare|find|tell|list|explain|show)\b", lowered))
        return action_hits > 1 or bool(re.search(r"\b(and then|also|as well as)\b", lowered))

    @staticmethod
    def _clarifying_question(subtask: dict[str, Any]) -> str:
        description = compact_text(subtask.get("description"), limit=180) or "that request"
        return (
            f"Which feed should I use for {description}: domotics, unraid, or plex?"
        )

    def _combined_sources(
        self,
        state: AgentState,
        result_sources: list[dict[str, str | None]] | None,
    ) -> list[dict[str, str | None]]:
        combined: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str | None]] = set()

        def add(source: Any) -> None:
            if not isinstance(source, dict):
                return
            title = source.get("title")
            url = source.get("url")
            normalized_title = str(title).strip() if title is not None and str(title).strip() else None
            normalized_url = str(url).strip() if url is not None and str(url).strip() else None
            if normalized_title is None and normalized_url is None:
                return
            key = (normalized_url, normalized_title)
            if key in seen:
                return
            seen.add(key)
            combined.append({"title": normalized_title, "url": normalized_url})

        for result in state.get("external_results", []):
            for source in result.get("sources") or []:
                add(source)
        for source in result_sources or []:
            add(source)
        return combined

    def _grouped_evidence_text(self, state: AgentState) -> str:
        lines = []
        for result in state.get("external_results", []):
            lines.append(
                f"External subtask {result.get('subtask_id')} [{result.get('status')}]: "
                f"{result.get('description')}"
            )
            content = compact_text(result.get("content"), limit=900)
            if content:
                lines.append(f"- {content}")
            sources = result.get("sources") or []
            for source in sources[:5]:
                title = source.get("title") or "source"
                url = source.get("url") or ""
                lines.append(f"- source: {title} {url}".strip())
        for group in state.get("retrieval_groups", []):
            lines.append(
                f"Subtask {group.get('subtask_id')} [{group.get('channel')}]: {group.get('description')}"
            )
            messages = group.get("messages") or []
            if not messages:
                lines.append("- No retrieved feed messages.")
                continue
            for hit in messages:
                payload = hit.get("payload", {})
                content = compact_text(payload.get("content"), limit=900)
                if content:
                    lines.append(f"- [{payload.get('channel')}] {content}")
        return "\n".join(lines) if lines else "(no grouped feed evidence)"

    def _build_prompt_messages(self, state: AgentState) -> list[dict[str, str]]:
        context_blocks = []
        if (
            state.get("summary")
            or state.get("memories")
            or state.get("documents")
            or state.get("retrieval_groups")
            or state.get("external_results")
            or state.get("context_messages")
            or state.get("recent_messages")
        ):
            context_blocks.append(
                "Context priority rules:\n"
                "- First inspect and use the provided private chat context, recent private transcript, and grouped passive feed context.\n"
                "- Answer the user's requested information directly; do not restate the full retrieval plan or evidence inventory.\n"
                "- Resolve follow-up references from the recent private transcript before answering or searching.\n"
                "- If the user asks about something in domotics, unraid, or plex, use only the grouped evidence for that same feed.\n"
                "- Do not transfer meaning across domotics, unraid, and plex; the feed name is part of the context.\n"
                "- Use resolved external context only for outside-world premises; do not treat it as feed evidence.\n"
                "- Cite external sources with [1], [2], etc. when source numbers are provided; do not list source names in prose when a citation is enough.\n"
                "- Summarize empty feed results compactly, for example no match in domotics or plex, instead of listing every empty group.\n"
                "- Do not replace anything found in grouped passive feed context with web search results, public websites, prior assistant guesses, or unrelated private memories.\n"
                "- Use web search only after the provided context and recent transcript do not contain the answer, and clearly say when the answer came from web search instead of stored context.\n"
            )
        if state.get("summary"):
            context_blocks.append(f"Private chat summary:\n{state['summary']}")

        subtask_lines = [
            (
                f"- {subtask.get('id')}: {subtask.get('description')} "
                f"(scope: {subtask.get('selected_channel') or 'private/non-feed'})"
            )
            for subtask in state.get("subtasks", [])
            if subtask.get("description")
        ]
        if subtask_lines:
            context_blocks.append(
                "Request plan:\n"
                + "\n".join(subtask_lines)
                + "\nUse the plan to preserve dependencies between subtasks. If a later check depends on an "
                "earlier premise, establish or caveat that premise before drawing the dependent conclusion. "
                "Use resolved external context for outside-world premises and grouped feed context for feed claims; "
                "when they are insufficient, state the uncertainty instead of guessing. Keep the final reply concise "
                "and shaped around the user's question, not the internal subtask structure."
            )

        followup_rules = self._followup_reference_rules(state)
        if followup_rules:
            context_blocks.append(followup_rules)

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

        external_blocks = []
        external_source_number = 1
        for result in state.get("external_results", []):
            source_lines = []
            for source in (result.get("sources") or [])[:5]:
                title = source.get("title") or "source"
                url = source.get("url")
                source_lines.append(f"- [{external_source_number}] source: {title}{' - ' + url if url else ''}")
                external_source_number += 1
            block_lines = [
                f"Subtask {result.get('subtask_id')} [{result.get('status')}]: {result.get('description')}",
                str(result.get("content") or "").strip() or "No external result was returned.",
            ]
            block_lines.extend(source_lines)
            external_blocks.append("\n".join(block_lines))
        if external_blocks:
            context_blocks.append(
                "External resolved context:\n"
                + "\n\n".join(external_blocks)
                + "\nIf this section is disabled, unavailable, or unsourced, answer the dependent part as partial "
                "or uncertain instead of claiming a definite match."
            )

        grouped_feed_blocks = []
        for group in state.get("retrieval_groups", []):
            channel = group.get("channel")
            description = group.get("description")
            feed_lines = [
                f"- [{hit['payload'].get('channel')}] {hit['payload'].get('content')}"
                for hit in group.get("messages", [])
                if hit.get("payload", {}).get("content") and hit.get("payload", {}).get("channel")
            ]
            if feed_lines:
                grouped_feed_blocks.append(
                    f"Subtask {group.get('subtask_id')} [{channel}]: {description}\n" + "\n".join(feed_lines)
                )
        if grouped_feed_blocks:
            context_blocks.append("Grouped passive feed context:\n" + "\n\n".join(grouped_feed_blocks))

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
                "retrieval_group_count": len(state.get("retrieval_groups", [])),
                "external_result_count": len(state.get("external_results", [])),
                "needs_final_generation": bool(state.get("needs_final_generation")),
                "recent_history_count": len(state.get("recent_messages", [])),
                "context_sections": [
                    "context_priority_rules" if block.startswith("Context priority rules") else
                    "private_summary" if block.startswith("Private chat summary") else
                    "request_plan" if block.startswith("Request plan") else
                    "private_memories" if block.startswith("Relevant private memories") else
                    "private_documents" if block.startswith("Relevant private documents") else
                    "external_resolved_context" if block.startswith("External resolved context") else
                    "passive_feed_context" if block.startswith("Grouped passive feed context") else
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

    @staticmethod
    def _followup_reference_rules(state: AgentState) -> str | None:
        text = str(state.get("text") or "").lower()
        if "this number" not in text and "that number" not in text and "with this number" not in text:
            return None
        for message in reversed(state.get("recent_messages", [])):
            if message.get("role") != "assistant":
                continue
            numbers = re.findall(r"\b\d+\b", str(message.get("content") or ""))
            if not numbers:
                continue
            number = numbers[-1]
            return (
                "Follow-up reference resolution:\n"
                f"- The recent private transcript indicates to treat the number as {number}.\n"
                "- The user's follow-up grants permission to search if the requested answer is not in stored context.\n"
                "- If web search is available, use that value as the web search subject."
            )
        return None

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
            "keyword_score": hit.get("keyword_score"),
            "hybrid_score": hit.get("hybrid_score"),
            "retrieval_source": hit.get("retrieval_source"),
            "retrieval_sources": hit.get("retrieval_sources"),
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
            "retrieval_sources": ["recent"],
            "payload": {
                "source_type": "context_message",
                "channel": row.get("channel"),
                "telegram_chat_id": row.get("telegram_chat_id"),
                "telegram_message_id": row.get("telegram_message_id"),
                "content": row.get("content"),
                "created_at": row.get("created_at"),
            },
        }

    def _keyword_memory_hits(
        self,
        *,
        user_id: UUID,
        query: str,
        channels: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        search = getattr(self.repository, "search_memories_keyword", None)
        if search is None:
            return []
        rows = search(user_id=user_id, query=query, channels=channels, limit=limit)
        return [row_to_hit(row, source_type="memory") for row in rows]

    def _keyword_document_hits(
        self,
        *,
        user_id: UUID,
        query: str,
        channels: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        search = getattr(self.repository, "search_document_chunks_keyword", None)
        if search is None:
            return []
        rows = search(user_id=user_id, query=query, channels=channels, limit=limit)
        return [row_to_hit(row, source_type="document_chunk") for row in rows]

    def _keyword_context_hits(
        self,
        *,
        user_id: UUID,
        channels: list[str],
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        search = getattr(self.repository, "search_context_messages_keyword", None)
        if search is None:
            return []
        rows = search(user_id=user_id, channels=channels, query=query, limit=limit)
        return [row_to_hit(row, source_type="context_message") for row in rows]

    def _memory_exists(self, *, user_id: UUID, channel: str, content: str) -> bool:
        lookup = getattr(self.repository, "get_memory_by_content", None)
        if lookup is None:
            return False
        return bool(lookup(user_id=user_id, channel=channel, content=content))

    @staticmethod
    def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

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
