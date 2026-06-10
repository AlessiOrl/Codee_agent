from uuid import uuid4

from app.agent_graph import AgentGraph
from app.openwebui import LlmResult


class FakeRepo:
    def __init__(self) -> None:
        self.user_id = uuid4()
        self.conversation_id = uuid4()
        self.saved = []
        self.memories = []

    def upsert_user(self, **kwargs):
        return {"id": self.user_id, **kwargs}

    def upsert_conversation(self, **kwargs):
        return {"id": self.conversation_id, **kwargs}

    def save_message(self, **kwargs):
        row = {"id": uuid4(), **kwargs}
        self.saved.append(row)
        return row

    def recent_messages(self, **kwargs):
        return [{"role": "user", "content": "hello"}]

    def latest_summary(self, **kwargs):
        return {"summary": "User is testing Codee."}

    def message_count(self, **kwargs):
        return 1

    def save_memory(self, **kwargs):
        row = {"id": uuid4(), **kwargs}
        self.memories.append(row)
        return row

    def get_memory_by_content(self, **kwargs):
        for memory in self.memories:
            if (
                memory["user_id"] == kwargs["user_id"]
                and memory["channel"] == kwargs["channel"]
                and memory["content"].lower() == kwargs["content"].lower()
            ):
                return memory
        return None

    def save_summary(self, **kwargs):
        return {"id": uuid4(), **kwargs}

    def search_memories_keyword(self, **kwargs):
        return []

    def search_document_chunks_keyword(self, **kwargs):
        return []

    def search_context_messages_keyword(self, **kwargs):
        return []

    def recent_context_messages(self, **kwargs):
        return [
            {
                "id": uuid4(),
                "channel": "unraid",
                "telegram_chat_id": "22",
                "telegram_message_id": "101",
                "content": "The stored number is 42.",
                "qdrant_point_id": "context-point-1",
                "created_at": "2026-06-09T20:00:00Z",
            }
        ]


class FollowupRepo(FakeRepo):
    def recent_messages(self, **kwargs):
        return [
            {"role": "user", "content": "There is a value stored in the Domotics notification chat, what is that number"},
            {"role": "assistant", "content": "The number stored in the Domotics Debug test is 55."},
            {"role": "user", "content": "Nice, now i want to know which f1 drivers with this number"},
        ]

    def latest_summary(self, **kwargs):
        return None


class FakeVector:
    def __init__(self) -> None:
        self.search_calls = 0
        self.context_search_calls = []

    def search_memories(self, **kwargs):
        self.search_calls += 1
        return [
            {"payload": {"channel": "private", "content": "User likes concise replies."}, "score": 0.95},
        ]

    def search_documents(self, **kwargs):
        self.search_calls += 1
        return [
            {"payload": {"channel": "private", "filename": "notes.txt", "content": "Local project note."}, "score": 0.9},
        ]

    def search_context_messages(self, **kwargs):
        self.search_calls += 1
        self.context_search_calls.append(kwargs)
        return [
            {"payload": {"channel": "domotics", "content": "Front door opened."}, "score": 0.88},
        ]

    def upsert_memory(self, **kwargs):
        return "point-1"


class MultiFeedVector(FakeVector):
    def search_context_messages(self, **kwargs):
        self.search_calls += 1
        self.context_search_calls.append(kwargs)
        channel = kwargs["channels"][0]
        return [
            {
                "payload": {
                    "source_type": "context_message",
                    "channel": channel,
                    "content": f"{channel} semantic context.",
                    "created_at": "2026-06-09T20:00:00Z",
                },
                "score": 0.88,
            },
        ]


class EmptyContextVector(FakeVector):
    def search_context_messages(self, **kwargs):
        self.search_calls += 1
        self.context_search_calls.append(kwargs)
        return []


class NoPrivateContextVector(EmptyContextVector):
    def search_memories(self, **kwargs):
        self.search_calls += 1
        return []

    def search_documents(self, **kwargs):
        self.search_calls += 1
        return []


class WrongChannelContextVector(FakeVector):
    def search_context_messages(self, **kwargs):
        self.search_calls += 1
        self.context_search_calls.append(kwargs)
        return [
            {
                "payload": {
                    "source_type": "context_message",
                    "channel": "plex",
                    "content": "This should not cross into domotics.",
                },
                "score": 0.99,
            }
        ]


class KeywordContextRepo(FakeRepo):
    def search_context_messages_keyword(self, **kwargs):
        return [
            {
                "id": uuid4(),
                "channel": "domotics",
                "telegram_chat_id": "11",
                "telegram_message_id": "201",
                "content": "Garage sensor battery is 9 percent.",
                "qdrant_point_id": "keyword-context-1",
                "created_at": "2026-06-09T20:00:00Z",
                "keyword_score": 1.0,
            },
            {
                "id": uuid4(),
                "channel": "plex",
                "telegram_chat_id": "33",
                "telegram_message_id": "301",
                "content": "Plex keyword hit must not be injected.",
                "qdrant_point_id": "keyword-context-2",
                "created_at": "2026-06-09T20:00:00Z",
                "keyword_score": 1.0,
            },
        ]


class MultiFeedRepo(FakeRepo):
    def recent_context_messages(self, **kwargs):
        channel = kwargs["channels"][0]
        return [
            {
                "id": uuid4(),
                "channel": channel,
                "telegram_chat_id": str({"unraid": 22, "plex": 33, "domotics": 11}.get(channel, 0)),
                "telegram_message_id": f"{channel}-101",
                "content": f"{channel} recent context.",
                "qdrant_point_id": f"{channel}-context-point",
                "created_at": "2026-06-09T20:00:00Z",
            }
        ]


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []
        self.call_options = []

    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        if "You decide whether passive notification feed context" in messages[0]["content"]:
            return LlmResult(content='{"channels": ["domotics"], "confidence": 0.8, "rationale": "home event question"}')
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content='{"memories": []}')
        return LlmResult(content="final answer")

    def stream_chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        yield {"type": "delta", "content": "streamed "}
        yield {"type": "delta", "content": "answer"}
        yield {"type": "done", "content": "streamed answer", "sources": []}


class MemoryExtractingLlm(FakeLlm):
    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        if "You decide whether passive notification feed context" in messages[0]["content"]:
            return LlmResult(content='{"channels": [], "confidence": 0.0, "rationale": "private memory"}')
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(
                content=(
                    '{"memories": ['
                    '{"content": "User is evaluating hybrid retrieval.", '
                    '"memory_type": "project", "importance": 4, "confidence": 0.7}'
                    "]}"
                )
            )
        return LlmResult(content="final answer")


class FencedRouterLlm(FakeLlm):
    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        if "You decide whether passive notification feed context" in messages[0]["content"]:
            return LlmResult(
                content=(
                    "```json\n"
                    "{\n"
                    '  "channels": ["unraid"],\n'
                    '  "confidence": 0.95,\n'
                    '  "rationale": "The user explicitly asks for the unraid chat."\n'
                    "}\n"
                    "```"
                )
            )
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content="```json\n{\"memories\": []}\n```")
        return LlmResult(content="final answer")


class AmbiguousGateLlm(FakeLlm):
    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        if "You decide whether passive notification feed context" in messages[0]["content"]:
            return LlmResult(
                content=(
                    '{"selected_channel": null, "confidence": 0.2, "ambiguous": true, '
                    '"clarifying_question": "Which feed should I use: domotics, unraid, or plex?", '
                    '"rationale": "The alert could belong to multiple feeds."}'
                )
            )
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content='{"memories": []}')
        return LlmResult(content="final answer")


class WebResolvingLlm(FakeLlm):
    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        if "Resolve only the outside-world fact" in messages[0]["content"]:
            return LlmResult(
                content="The current outside-world value needed for the comparison is car number 4.",
                sources=[{"title": "Race result", "url": "https://example.test/race"}],
            )
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content='{"memories": []}')
        return LlmResult(content="The resolved car number is 4 [1]. I found no matching feed evidence.")


class FailingExternalResolverLlm(FakeLlm):
    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        self.call_options.append(use_web_search)
        if "Resolve only the outside-world fact" in messages[0]["content"]:
            raise RuntimeError("web unavailable")
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content='{"memories": []}')
        return LlmResult(content="I could not verify the outside value, so this is only a separate feed check.")


def test_graph_invocation_saves_user_and_assistant_messages() -> None:
    repo = FakeRepo()
    llm = FakeLlm()
    graph = AgentGraph(
        repository=repo,
        vector_store=FakeVector(),
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=False,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "What changed?",
        }
    )

    assert state["reply"] == "final answer"
    assert len(repo.saved) == 2
    assert repo.saved[0]["role"] == "user"
    assert repo.saved[1]["role"] == "assistant"
    assert state["request_plan"]["mode"] == "direct"
    assert state["needs_final_generation"] is False
    first_prompt = llm.calls[0][0]["content"]
    assert "Context priority rules" in first_prompt
    assert "Private chat summary" in first_prompt
    assert "Relevant private memories" in first_prompt
    assert "Relevant private documents" in first_prompt
    assert "Grouped passive feed context" not in first_prompt


def test_context_router_accepts_fenced_json() -> None:
    repo = FakeRepo()
    llm = FencedRouterLlm()
    vector = FakeVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "I want the number stored in the unraid chat.",
        }
    )

    assert state["context_channels"] == ["unraid"]
    assert vector.context_search_calls[0]["channels"] == ["unraid"]
    assert state["reply"] == "final answer"
    assert state["needs_final_generation"] is True
    assert state["external_results"] == []
    assert True not in llm.call_options


def test_selected_feed_uses_recent_messages_when_semantic_search_is_empty() -> None:
    repo = FakeRepo()
    llm = FencedRouterLlm()
    vector = EmptyContextVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "Retrieve the number I asked you from the Unraid chat.",
        }
    )

    assert state["context_channels"] == ["unraid"]
    assert state["context_messages"][0]["retrieval_source"] == "recent"
    assert state["context_messages"][0]["payload"]["content"] == "The stored number is 42."
    answer_prompt = next(call[0]["content"] for call in llm.calls if "Context priority rules" in call[0]["content"])
    assert "Context priority rules" in answer_prompt
    assert "Use web search only after the provided context and recent transcript do not contain the answer" in answer_prompt
    assert "Grouped passive feed context" in answer_prompt
    assert "The stored number is 42." in answer_prompt


def test_keyword_feed_retrieval_stays_inside_router_selected_channel() -> None:
    repo = KeywordContextRepo()
    llm = FakeLlm()
    vector = WrongChannelContextVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "What is the garage sensor battery in domotics?",
        }
    )

    assert state["context_channels"] == ["domotics"]
    assert vector.context_search_calls[0]["channels"] == ["domotics"]
    context_contents = [hit["payload"]["content"] for hit in state["context_messages"]]
    assert "Garage sensor battery is 9 percent." in context_contents
    assert "Plex keyword hit must not be injected." not in context_contents
    assert "This should not cross into domotics." not in context_contents


def test_ambiguous_feed_question_asks_clarification_without_retrieval() -> None:
    repo = FakeRepo()
    llm = AmbiguousGateLlm()
    vector = FakeVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "What was the alert?",
        }
    )

    assert state["context_channels"] == []
    assert state["context_messages"] == []
    assert vector.context_search_calls == []
    assert state["reply"] == "Which feed should I use: domotics, unraid, or plex?"


def test_multi_feed_request_retrieves_each_feed_separately() -> None:
    repo = MultiFeedRepo()
    llm = FakeLlm()
    vector = MultiFeedVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "Check unraid status and summarize plex.",
        }
    )

    assert state["context_channels"] == ["plex", "unraid"]
    assert [call["channels"] for call in vector.context_search_calls] == [["plex"], ["unraid"]]
    assert [group["channel"] for group in state["retrieval_groups"]] == ["plex", "unraid"]
    answer_prompt = next(call[0]["content"] for call in llm.calls if "Grouped passive feed context" in call[0]["content"])
    assert "Subtask feed-plex [plex]" in answer_prompt
    assert "Subtask feed-unraid [unraid]" in answer_prompt


def test_generic_feed_chat_request_checks_all_feeds_without_clarification() -> None:
    repo = MultiFeedRepo()
    llm = FakeLlm()
    vector = MultiFeedVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "Is there any number in the feed chats equal to 4?",
        }
    )

    assert state.get("clarification_question") is None
    assert state["request_plan"]["mode"] == "generic_all_feeds"
    assert state["context_channels"] == ["domotics", "plex", "unraid"]
    assert [call["channels"] for call in vector.context_search_calls] == [["domotics"], ["plex"], ["unraid"]]
    assert [group["channel"] for group in state["retrieval_groups"]] == ["domotics", "plex", "unraid"]
    assert state["external_results"] == []
    assert True not in llm.call_options


def test_mixed_external_fact_and_feed_chat_request_checks_all_feeds() -> None:
    repo = MultiFeedRepo()
    llm = WebResolvingLlm()
    vector = MultiFeedVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": (
                "What is the car number of the latest f1 grand prix winner and, "
                "is there any number in the feed chat that is equal to that car number?"
            ),
        }
    )

    assert state.get("clarification_question") is None
    assert state["request_plan"]["mode"] == "generic_all_feeds"
    assert [subtask["id"] for subtask in state["subtasks"]] == [
        "external-task-1",
        "feed-domotics",
        "feed-plex",
        "feed-unraid",
    ]
    assert state["subtasks"][0]["needs_feed_context"] is False
    assert state["context_channels"] == ["domotics", "plex", "unraid"]
    assert [call["channels"] for call in vector.context_search_calls] == [["domotics"], ["plex"], ["unraid"]]
    assert state["external_results"][0]["subtask_id"] == "external-task-1"
    assert state["external_results"][0]["status"] == "resolved_with_sources"
    assert state["reply"] == "The resolved car number is 4 [1]. I found no matching feed evidence."
    assert state["sources"] == [{"title": "Race result", "url": "https://example.test/race"}]
    assert True in llm.call_options
    assert llm.call_options.count(True) == 1
    answer_prompt = next(call[0]["content"] for call in llm.calls if "Request plan" in call[0]["content"])
    assert "external-task-1" in answer_prompt
    assert "feed-domotics" in answer_prompt
    assert "External resolved context" in answer_prompt
    assert "car number 4" in answer_prompt
    assert "[1] source: Race result - https://example.test/race" in answer_prompt
    assert "Answer the user's requested information directly" in answer_prompt
    assert "preserve dependencies between subtasks" in answer_prompt
    assert "state the uncertainty instead of guessing" in answer_prompt
    assert len(llm.calls) == 3


def test_external_resolver_failure_still_allows_partial_feed_answer() -> None:
    repo = MultiFeedRepo()
    llm = FailingExternalResolverLlm()
    vector = MultiFeedVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": (
                "What is the car number of the latest f1 grand prix winner and, "
                "is there any number in the feed chat that is equal to that car number?"
            ),
        }
    )

    assert state["external_results"][0]["status"] == "unavailable"
    assert llm.call_options.count(True) == 1
    assert [call["channels"] for call in vector.context_search_calls] == [["domotics"], ["plex"], ["unraid"]]
    answer_prompt = next(call[0]["content"] for call in llm.calls if "External resolved context" in call[0]["content"])
    assert "External fact resolution failed" in answer_prompt
    assert "answer the dependent part as partial" in answer_prompt
    assert "could not verify" in state["reply"]
    assert "[1]" not in state["reply"]
    assert state["sources"] == []
    assert len(llm.calls) == 3


def test_followup_web_question_resolves_number_from_recent_transcript() -> None:
    repo = FollowupRepo()
    llm = FakeLlm()
    graph = AgentGraph(
        repository=repo,
        vector_store=NoPrivateContextVector(),
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=False,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "Nice, now i want to know which f1 drivers with this number",
        }
    )

    answer_prompt = llm.calls[0][0]["content"]
    assert state["reply"] == "final answer"
    assert "Context priority rules" in answer_prompt
    assert "Resolve follow-up references" in answer_prompt
    assert "treat the number as 55" in answer_prompt
    assert "grants permission to search" in answer_prompt
    assert "use that value as the web search subject" in answer_prompt


def test_telegram_command_does_not_call_llm_or_vector_search() -> None:
    repo = FakeRepo()
    llm = FakeLlm()
    vector = FakeVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=1,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "4",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "/unknown",
        }
    )

    assert state["reply"].startswith("Command received.")
    assert llm.calls == []
    assert vector.search_calls == 0
    assert len(repo.saved) == 2


def test_memory_extraction_stores_source_and_confidence() -> None:
    repo = FakeRepo()
    llm = MemoryExtractingLlm()
    vector = FakeVector()
    graph = AgentGraph(
        repository=repo,
        vector_store=vector,
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "username": "orlando",
            "first_name": "Orlando",
            "text": "Remember that I am evaluating hybrid retrieval.",
        }
    )

    assert len(repo.memories) == 1
    memory = repo.memories[0]
    assert memory["content"] == "User is evaluating hybrid retrieval."
    assert memory["confidence"] == 0.7
    assert memory["source_user_message_id"] == repo.saved[0]["id"]
    assert "Remember that I am evaluating hybrid retrieval." in memory["source_text"]


def test_stream_invocation_saves_final_assistant_message() -> None:
    repo = FakeRepo()
    llm = FakeLlm()
    graph = AgentGraph(
        repository=repo,
        vector_store=FakeVector(),
        llm=llm,
        system_prompt="System prompt.",
        recent_history_limit=20,
        summary_every_n_messages=12,
        cross_channel_memory_limit=3,
        cross_channel_doc_limit=2,
        cross_channel_min_score=0.6,
        context_router_enabled=True,
        context_router_max_feed_messages=6,
        context_router_min_confidence=0.4,
        context_debug_json=False,
    )

    events = list(
        graph.stream(
            {
                "telegram_user_id": "1",
                "telegram_chat_id": "2",
                "channel": "private",
                "telegram_message_id": "3",
                "username": "orlando",
                "first_name": "Orlando",
                "text": "Stream this.",
            }
        )
    )

    assert events[0] == {"type": "delta", "content": "streamed "}
    assert events[1] == {"type": "delta", "content": "answer"}
    assert events[-1]["type"] == "done"
    assert events[-1]["reply"] == "streamed answer"
    assert repo.saved[-1]["role"] == "assistant"
    assert repo.saved[-1]["content"] == "streamed answer"
