from datetime import datetime, timezone
from uuid import uuid4

from app.summary import SummaryService


class FakeRepository:
    def __init__(self, *, has_user: bool = True, has_conversation: bool = True, has_summary: bool = True):
        self.has_user = has_user
        self.has_conversation = has_conversation
        self.has_summary = has_summary
        self.user_id = uuid4()
        self.conversation_id = uuid4()

    def get_user_by_telegram_user_id(self, **kwargs):
        if not self.has_user:
            return None
        return {"id": self.user_id}

    def get_conversation(self, **kwargs):
        if not self.has_conversation:
            return None
        return {"id": self.conversation_id}

    def latest_summary(self, **kwargs):
        if not self.has_summary:
            return None
        return {
            "summary": "User prefers concise answers.",
            "created_at": datetime(2026, 6, 9, 10, 0, tzinfo=timezone.utc),
        }


def test_summary_returns_current_summary_and_explains_storage() -> None:
    service = SummaryService(repository=FakeRepository())

    result = service.get_summary(telegram_user_id="1", telegram_chat_id="2")

    assert result["summary"] == "User prefers concise answers."
    assert "plain text stored in Postgres" in result["reply"]
    assert result["summary_created_at"] == "2026-06-09T10:00:00+00:00"


def test_summary_handles_missing_summary_cleanly() -> None:
    service = SummaryService(repository=FakeRepository(has_summary=False))

    result = service.get_summary(telegram_user_id="1", telegram_chat_id="2")

    assert result["summary"] is None
    assert "not embeddings" in result["reply"]
