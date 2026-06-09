from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.forget import ForgetService


class FakeRepository:
    def __init__(self, user_exists: bool = True) -> None:
        self.user_exists = user_exists
        self.calls = []
        self.user_id = uuid4()
        self.result = {
            "counts": {
                "messages": 2,
                "summaries": 1,
                "memories": 3,
                "documents": 1,
                "document_chunks": 4,
                "tool_calls": 2,
                "conversations": 1,
            },
            "memory_point_ids": ["mem-1", "mem-2"],
            "document_point_ids": ["doc-1"],
        }

    def get_user_by_telegram_user_id(self, **kwargs):
        self.calls.append(("get_user", kwargs))
        if not self.user_exists:
            return None
        return {"id": self.user_id}

    def forget_user_data(self, **kwargs):
        self.calls.append(("forget_user_data", kwargs))
        return self.result


class FakeVectorStore:
    def __init__(self) -> None:
        self.memory_deletes = []
        self.document_deletes = []

    def delete_memory_points(self, point_ids):
        self.memory_deletes.append(point_ids)

    def delete_document_points(self, point_ids):
        self.document_deletes.append(point_ids)


def test_forget_full_reset_deletes_all_user_data() -> None:
    repo = FakeRepository()
    vector = FakeVectorStore()
    service = ForgetService(repository=repo, vector_store=vector)

    result = service.forget(telegram_user_id="1", telegram_chat_id="2", days=None)

    assert result["scope"] == "all"
    assert result["counts"]["messages"] == 2
    assert "Forgot all stored data" in result["reply"]
    assert repo.calls[1][0] == "forget_user_data"
    assert repo.calls[1][1]["user_id"] == repo.user_id
    assert repo.calls[1][1]["cutoff"] is None
    assert vector.memory_deletes == [["mem-1", "mem-2"]]
    assert vector.document_deletes == [["doc-1"]]


def test_forget_recent_delete_uses_cutoff_and_reports_recent_scope() -> None:
    repo = FakeRepository()
    vector = FakeVectorStore()
    service = ForgetService(repository=repo, vector_store=vector)

    before = datetime.now(timezone.utc) - timedelta(days=2, seconds=5)
    result = service.forget(telegram_user_id="1", telegram_chat_id="2", days=2)
    after = datetime.now(timezone.utc) - timedelta(days=2) + timedelta(seconds=5)

    cutoff = repo.calls[1][1]["cutoff"]
    assert result["scope"] == "recent"
    assert "last 2 day(s)" in result["reply"]
    assert before <= cutoff <= after


def test_forget_unknown_user_is_safe_noop() -> None:
    repo = FakeRepository(user_exists=False)
    vector = FakeVectorStore()
    service = ForgetService(repository=repo, vector_store=vector)

    result = service.forget(telegram_user_id="1", telegram_chat_id="2", days=3)

    assert result["counts"] == {
        "messages": 0,
        "summaries": 0,
        "memories": 0,
        "documents": 0,
        "document_chunks": 0,
        "tool_calls": 0,
        "conversations": 0,
    }
    assert "nothing was removed" in result["reply"].lower()
    assert vector.memory_deletes == []
    assert vector.document_deletes == []
