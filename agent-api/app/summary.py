from app.repository import Repository


class SummaryService:
    def __init__(self, *, repository: Repository):
        self.repository = repository

    def get_summary(self, *, telegram_user_id: str, telegram_chat_id: str, channel: str) -> dict:
        del telegram_chat_id
        user = self.repository.get_user_by_telegram_user_id(telegram_user_id=telegram_user_id)
        if not user:
            return {
                "reply": (
                    "No summary is stored yet for this chat.\n\n"
                    "Summaries are plain text stored in Postgres, not embeddings."
                ),
                "summary": None,
                "summary_created_at": None,
            }

        conversation = self.repository.get_conversation(
            user_id=user["id"],
            channel=channel,
        )
        if not conversation:
            return {
                "reply": (
                    "No summary is stored yet for this chat.\n\n"
                    "Summaries are plain text stored in Postgres, not embeddings."
                ),
                "summary": None,
                "summary_created_at": None,
            }

        latest = self.repository.latest_summary(conversation_id=conversation["id"])
        if not latest:
            return {
                "reply": (
                    "No summary is stored yet for this chat.\n\n"
                    "Summaries are plain text stored in Postgres, not embeddings."
                ),
                "summary": None,
                "summary_created_at": None,
            }

        created_at = latest.get("created_at")
        created_text = created_at.isoformat() if created_at else None
        return {
            "reply": (
                "Current chat summary:\n"
                f"{latest['summary']}\n\n"
                "This is plain text stored in Postgres, not embeddings."
            ),
            "summary": latest["summary"],
            "summary_created_at": created_text,
        }
