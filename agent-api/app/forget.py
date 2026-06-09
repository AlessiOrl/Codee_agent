from datetime import datetime, timedelta, timezone

from app.repository import Repository
from app.vector_store import VectorStore


class ForgetService:
    def __init__(self, *, repository: Repository, vector_store: VectorStore):
        self.repository = repository
        self.vector_store = vector_store

    def forget(self, *, telegram_user_id: str, telegram_chat_id: str, days: int | None) -> dict:
        del telegram_chat_id

        user = self.repository.get_user_by_telegram_user_id(telegram_user_id=telegram_user_id)
        if not user:
            return {
                "reply": self._reply(days=days, counts=self._empty_counts(), had_data=False),
                "counts": self._empty_counts(),
                "scope": self._scope(days),
            }

        cutoff = None
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        result = self.repository.forget_user_data(user_id=user["id"], cutoff=cutoff)
        self.vector_store.delete_memory_points(result["memory_point_ids"])
        self.vector_store.delete_document_points(result["document_point_ids"])

        counts = result["counts"]
        return {
            "reply": self._reply(days=days, counts=counts, had_data=any(counts.values())),
            "counts": counts,
            "scope": self._scope(days),
        }

    @staticmethod
    def _scope(days: int | None) -> str:
        return "recent" if days is not None else "all"

    @staticmethod
    def _empty_counts() -> dict[str, int]:
        return {
            "messages": 0,
            "summaries": 0,
            "memories": 0,
            "documents": 0,
            "document_chunks": 0,
            "tool_calls": 0,
            "conversations": 0,
        }

    def _reply(self, *, days: int | None, counts: dict[str, int], had_data: bool) -> str:
        if not had_data:
            if days is None:
                return "Nothing was stored for this user, so there was nothing to forget."
            return f"No stored data from the last {days} day(s) was found, so nothing was removed."

        if days is None:
            prefix = "Forgot all stored data for this user."
        else:
            prefix = f"Forgot stored data from the last {days} day(s) for this user."

        details = (
            f"Messages: {counts['messages']}, summaries: {counts['summaries']}, "
            f"memories: {counts['memories']}, documents: {counts['documents']}, "
            f"document chunks: {counts['document_chunks']}, tool calls: {counts['tool_calls']}, "
            f"conversations removed: {counts['conversations']}."
        )
        return f"{prefix}\n{details}"
