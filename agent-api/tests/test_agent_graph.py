from uuid import uuid4

from app.agent_graph import AgentGraph
from app.openwebui import LlmResult


class FakeRepo:
    def __init__(self) -> None:
        self.user_id = uuid4()
        self.conversation_id = uuid4()
        self.saved = []

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
        return {"id": uuid4(), **kwargs}

    def save_summary(self, **kwargs):
        return {"id": uuid4(), **kwargs}

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


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        if "You decide whether passive notification feed context" in messages[0]["content"]:
            return LlmResult(content='{"channels": ["domotics"], "confidence": 0.8, "rationale": "home event question"}')
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content='{"memories": []}')
        return LlmResult(content="final answer")

    def stream_chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
        yield {"type": "delta", "content": "streamed "}
        yield {"type": "delta", "content": "answer"}
        yield {"type": "done", "content": "streamed answer", "sources": []}


class FencedRouterLlm(FakeLlm):
    def chat(self, messages, *, use_web_search=None):
        self.calls.append(messages)
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
    router_prompt = llm.calls[0][1]["content"]
    assert "Private chat context already available" in router_prompt
    assert "Private summary" in router_prompt
    assert "Recent private transcript" in router_prompt
    assert "Relevant private memories" in router_prompt
    assert "Relevant private documents" in router_prompt
    first_prompt = llm.calls[1][0]["content"]
    assert "Context priority rules" in first_prompt
    assert "Private chat summary" in first_prompt
    assert "Relevant private memories" in first_prompt
    assert "Relevant private documents" in first_prompt
    assert "Selected passive feed context" in first_prompt


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
    answer_prompt = llm.calls[1][0]["content"]
    assert "Context priority rules" in answer_prompt
    assert "Use web search only after the provided context does not contain the answer" in answer_prompt
    assert "Selected passive feed context" in answer_prompt
    assert "The stored number is 42." in answer_prompt


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
