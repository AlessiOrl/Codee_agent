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


class FakeVector:
    def __init__(self) -> None:
        self.search_calls = 0

    def search_memories(self, **kwargs):
        self.search_calls += 1
        return [{"payload": {"content": "User likes concise replies."}, "score": 0.9}]

    def search_documents(self, **kwargs):
        self.search_calls += 1
        return [{"payload": {"filename": "notes.txt", "content": "Project note."}, "score": 0.8}]

    def upsert_memory(self, **kwargs):
        return "point-1"


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        if "Extract only durable" in messages[0]["content"]:
            return LlmResult(content='{"memories": []}')
        return LlmResult(content="final answer")

    def stream_chat(self, messages):
        self.calls.append(messages)
        yield {"type": "delta", "content": "streamed "}
        yield {"type": "delta", "content": "answer"}
        yield {"type": "done", "content": "streamed answer", "sources": []}


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
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
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
    first_prompt = llm.calls[0][0]["content"]
    assert "Conversation summary" in first_prompt
    assert "Relevant memories" in first_prompt
    assert "Relevant documents" in first_prompt


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
    )

    state = graph.invoke(
        {
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
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
    )

    events = list(
        graph.stream(
            {
                "telegram_user_id": "1",
                "telegram_chat_id": "2",
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
