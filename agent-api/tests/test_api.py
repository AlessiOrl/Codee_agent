from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.openwebui import OpenWebUIError


def configure_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://agent:pw@postgres:5432/agent_db")
    monkeypatch.setenv("OPENWEBUI_BASE_URL", "http://openwebui.test")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "secret")
    monkeypatch.setenv("OPENWEBUI_MODEL", "model-a")
    get_settings.cache_clear()


class FakeGraph:
    def stream(self, payload):
        conversation_id = uuid4()
        yield {"type": "delta", "content": "echo: "}
        yield {"type": "delta", "content": payload["text"]}
        yield {
            "type": "done",
            "reply": f"echo: {payload['text']}",
            "conversation_id": str(conversation_id),
            "sources": [{"title": "Docs", "url": "https://example.test"}],
        }


class FakeDb:
    def health(self):
        return "ok"


class FakeVector:
    def __init__(self) -> None:
        self.context_calls = []

    def health(self):
        return "ok"

    def upsert_context_message(self, **kwargs):
        self.context_calls.append(kwargs)
        return "context-point-1"


class FakeRepository:
    def __init__(self) -> None:
        self.user_id = uuid4()
        self.context_calls = []

    def upsert_user(self, **kwargs):
        return {"id": self.user_id, **kwargs}

    def save_context_message(self, **kwargs):
        self.context_calls.append(kwargs)
        return {"id": uuid4(), **kwargs}


class FakeForgetService:
    def __init__(self) -> None:
        self.calls = []

    def forget(self, **kwargs):
        self.calls.append(kwargs)
        scope = kwargs.get("scope", "channel")
        return {
            "reply": "Forgot stored data.",
            "scope": "recent_all" if scope == "all" and kwargs["days"] is not None else (
                "all" if scope == "all" else ("recent" if kwargs["days"] is not None else "channel")
            ),
            "counts": {
                "messages": 1,
                "summaries": 2,
                "memories": 3,
                "context_messages": 4,
                "documents": 4,
                "document_chunks": 5,
                "tool_calls": 6,
                "conversations": 7,
            },
        }


class FakeSummaryService:
    def __init__(self) -> None:
        self.calls = []

    def get_summary(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "reply": "Current chat summary:\nUser is testing Codee.",
            "summary": "User is testing Codee.",
            "summary_created_at": "2026-06-09T10:00:00+00:00",
        }


class RedirectingLlm:
    def chat(self, messages):
        raise OpenWebUIError(
            "Open WebUI returned HTTP 302 and redirected to http://openwebui.test/login"
        )


def test_chat_stream_requires_bearer_token(monkeypatch) -> None:
    configure_env(monkeypatch)
    app.state.agent_graph = FakeGraph()
    client = TestClient(app)

    response = client.post(
        "/chat/stream",
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "text": "hello",
        },
    )

    assert response.status_code == 401


def test_chat_stream_returns_ndjson_events(monkeypatch) -> None:
    configure_env(monkeypatch)
    app.state.agent_graph = FakeGraph()
    client = TestClient(app)

    response = client.post(
        "/chat/stream",
        headers={"Authorization": "Bearer test-key"},
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "telegram_message_id": "3",
            "text": "hello",
        },
    )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line]
    assert '"type": "delta"' in lines[0]
    assert '"type": "done"' in lines[-1]
    assert '"reply": "echo: hello"' in lines[-1]


def test_debug_llm_returns_bad_gateway_for_openwebui_error(monkeypatch) -> None:
    configure_env(monkeypatch)
    app.state.llm = RedirectingLlm()
    client = TestClient(app)

    response = client.post(
        "/debug/llm",
        headers={"Authorization": "Bearer test-key"},
        json={"text": "hello"},
    )

    assert response.status_code == 502
    assert "Open WebUI returned HTTP 302" in response.json()["detail"]


def test_health_uses_state_services() -> None:
    app.state.db = FakeDb()
    app.state.vector_store = FakeVector()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "postgres": "ok", "qdrant": "ok"}


def test_forget_requires_bearer_token(monkeypatch) -> None:
    configure_env(monkeypatch)
    app.state.forget_service = FakeForgetService()
    client = TestClient(app)

    response = client.post(
        "/forget",
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
        },
    )

    assert response.status_code == 401


def test_forget_full_reset_returns_service_response(monkeypatch) -> None:
    configure_env(monkeypatch)
    service = FakeForgetService()
    app.state.forget_service = service
    client = TestClient(app)

    response = client.post(
        "/forget",
        headers={"Authorization": "Bearer test-key"},
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
        },
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "Forgot stored data."
    assert response.json()["scope"] == "channel"
    assert service.calls == [{"telegram_user_id": "1", "telegram_chat_id": "2", "channel": "private", "scope": "channel", "days": None}]


def test_forget_recent_delete_passes_days(monkeypatch) -> None:
    configure_env(monkeypatch)
    service = FakeForgetService()
    app.state.forget_service = service
    client = TestClient(app)

    response = client.post(
        "/forget",
        headers={"Authorization": "Bearer test-key"},
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
            "days": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["scope"] == "recent"
    assert service.calls == [{"telegram_user_id": "1", "telegram_chat_id": "2", "channel": "private", "scope": "channel", "days": 3}]


def test_summary_requires_bearer_token(monkeypatch) -> None:
    configure_env(monkeypatch)
    app.state.summary_service = FakeSummaryService()
    client = TestClient(app)

    response = client.post(
        "/summary",
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
        },
    )

    assert response.status_code == 401


def test_summary_returns_service_response(monkeypatch) -> None:
    configure_env(monkeypatch)
    service = FakeSummaryService()
    app.state.summary_service = service
    client = TestClient(app)

    response = client.post(
        "/summary",
        headers={"Authorization": "Bearer test-key"},
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "2",
            "channel": "private",
        },
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "User is testing Codee."
    assert service.calls == [{"telegram_user_id": "1", "telegram_chat_id": "2", "channel": "private"}]


def test_context_message_ingest_stores_passive_feed(monkeypatch) -> None:
    configure_env(monkeypatch)
    repo = FakeRepository()
    vector = FakeVector()
    app.state.repository = repo
    app.state.vector_store = vector
    client = TestClient(app)

    response = client.post(
        "/context/messages",
        headers={"Authorization": "Bearer test-key"},
        json={
            "telegram_user_id": "1",
            "telegram_chat_id": "-100",
            "telegram_message_id": "55",
            "channel": "domotics",
            "text": "Garage door opened.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "stored"
    assert vector.context_calls[0]["channel"] == "domotics"
    assert repo.context_calls[0]["qdrant_point_id"] == "context-point-1"
