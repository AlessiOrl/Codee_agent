from datetime import datetime
from typing import Any
from uuid import UUID

from app.db import Database


class Repository:
    def __init__(self, db: Database):
        self.db = db

    def upsert_user(
        self,
        *,
        telegram_user_id: str,
        username: str | None,
        first_name: str | None,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO users (telegram_user_id, username, first_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (telegram_user_id) DO UPDATE
                SET username = EXCLUDED.username,
                    first_name = EXCLUDED.first_name,
                    updated_at = now()
                RETURNING *
                """,
                (telegram_user_id, username, first_name),
            ).fetchone()
            return dict(row)

    def get_user_by_telegram_user_id(self, *, telegram_user_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE telegram_user_id = %s",
                (telegram_user_id,),
            ).fetchone()
            return dict(row) if row else None

    def upsert_conversation(self, *, user_id: UUID, telegram_chat_id: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO conversations (user_id, telegram_chat_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, telegram_chat_id) DO UPDATE
                SET updated_at = now()
                RETURNING *
                """,
                (user_id, telegram_chat_id),
            ).fetchone()
            return dict(row)

    def get_conversation(self, *, user_id: UUID, telegram_chat_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM conversations
                WHERE user_id = %s AND telegram_chat_id = %s
                """,
                (user_id, telegram_chat_id),
            ).fetchone()
            return dict(row) if row else None

    def save_message(
        self,
        *,
        conversation_id: UUID,
        role: str,
        content: str,
        telegram_message_id: str | None = None,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, telegram_message_id)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (conversation_id, role, content, telegram_message_id),
            ).fetchone()
            return dict(row)

    def recent_messages(self, *, conversation_id: UUID, limit: int) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, telegram_message_id, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def latest_summary(self, *, conversation_id: UUID) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM conversation_summaries
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            return dict(row) if row else None

    def message_count(self, *, conversation_id: UUID) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT count(*) AS count FROM messages WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
            return int(row["count"])

    def save_summary(
        self,
        *,
        conversation_id: UUID,
        summary: str,
        from_message_id: UUID | None = None,
        to_message_id: UUID | None = None,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO conversation_summaries (
                    conversation_id, summary, from_message_id, to_message_id
                )
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (conversation_id, summary, from_message_id, to_message_id),
            ).fetchone()
            return dict(row)

    def save_memory(
        self,
        *,
        user_id: UUID,
        content: str,
        memory_type: str,
        qdrant_point_id: str,
        importance: int = 1,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO memories (
                    user_id, memory_type, content, qdrant_point_id, importance
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (user_id, memory_type, content, qdrant_point_id, importance),
            ).fetchone()
            return dict(row)

    def create_document(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID | None,
        filename: str,
        mime_type: str | None,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO documents (user_id, conversation_id, filename, mime_type)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (user_id, conversation_id, filename, mime_type),
            ).fetchone()
            return dict(row)

    def save_document_chunk(
        self,
        *,
        document_id: UUID,
        chunk_index: int,
        content: str,
        qdrant_point_id: str,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO document_chunks (
                    document_id, chunk_index, content, qdrant_point_id
                )
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (document_id, chunk_index, content, qdrant_point_id),
            ).fetchone()
            return dict(row)

    def update_document_chunk_point(self, *, chunk_id: UUID, qdrant_point_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE document_chunks SET qdrant_point_id = %s WHERE id = %s",
                (qdrant_point_id, chunk_id),
            )

    def forget_user_data(
        self,
        *,
        user_id: UUID,
        cutoff: datetime | None = None,
    ) -> dict[str, Any]:
        with self.db.connect() as conn:
            conversation_rows = conn.execute(
                "SELECT id FROM conversations WHERE user_id = %s",
                (user_id,),
            ).fetchall()
            conversation_ids = [row["id"] for row in conversation_rows]

            memory_rows = conn.execute(
                f"""
                SELECT id, qdrant_point_id
                FROM memories
                WHERE user_id = %s{self._cutoff_clause('created_at', cutoff)}
                """,
                self._cutoff_params((user_id,), cutoff),
            ).fetchall()
            memory_ids = [row["id"] for row in memory_rows]
            memory_point_ids = [
                row["qdrant_point_id"]
                for row in memory_rows
                if row.get("qdrant_point_id")
            ]

            document_rows = conn.execute(
                f"""
                SELECT id
                FROM documents
                WHERE user_id = %s{self._cutoff_clause('created_at', cutoff)}
                """,
                self._cutoff_params((user_id,), cutoff),
            ).fetchall()
            document_ids = [row["id"] for row in document_rows]

            document_chunk_rows = []
            if document_ids:
                document_chunk_rows = conn.execute(
                    """
                    SELECT id, qdrant_point_id
                    FROM document_chunks
                    WHERE document_id = ANY(%s)
                    """,
                    (document_ids,),
                ).fetchall()
            document_chunk_ids = [row["id"] for row in document_chunk_rows]
            document_point_ids = [
                row["qdrant_point_id"]
                for row in document_chunk_rows
                if row.get("qdrant_point_id")
            ]

            message_count = 0
            summary_count = 0
            tool_call_count = 0
            deleted_conversations = 0
            if conversation_ids:
                message_row = conn.execute(
                    f"""
                    SELECT count(*) AS count
                    FROM messages
                    WHERE conversation_id = ANY(%s){self._cutoff_clause('created_at', cutoff)}
                    """,
                    self._cutoff_params((conversation_ids,), cutoff),
                ).fetchone()
                summary_row = conn.execute(
                    f"""
                    SELECT count(*) AS count
                    FROM conversation_summaries
                    WHERE conversation_id = ANY(%s){self._cutoff_clause('created_at', cutoff)}
                    """,
                    self._cutoff_params((conversation_ids,), cutoff),
                ).fetchone()
                tool_call_row = conn.execute(
                    f"""
                    SELECT count(*) AS count
                    FROM tool_calls
                    WHERE conversation_id = ANY(%s){self._cutoff_clause('created_at', cutoff)}
                    """,
                    self._cutoff_params((conversation_ids,), cutoff),
                ).fetchone()
                message_count = int(message_row["count"])
                summary_count = int(summary_row["count"])
                tool_call_count = int(tool_call_row["count"])

                conn.execute(
                    f"""
                    DELETE FROM messages
                    WHERE conversation_id = ANY(%s){self._cutoff_clause('created_at', cutoff)}
                    """,
                    self._cutoff_params((conversation_ids,), cutoff),
                )
                conn.execute(
                    f"""
                    DELETE FROM conversation_summaries
                    WHERE conversation_id = ANY(%s){self._cutoff_clause('created_at', cutoff)}
                    """,
                    self._cutoff_params((conversation_ids,), cutoff),
                )
                conn.execute(
                    f"""
                    DELETE FROM tool_calls
                    WHERE conversation_id = ANY(%s){self._cutoff_clause('created_at', cutoff)}
                    """,
                    self._cutoff_params((conversation_ids,), cutoff),
                )

            if memory_ids:
                conn.execute("DELETE FROM memories WHERE id = ANY(%s)", (memory_ids,))
            if document_ids:
                conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (document_ids,))

            if cutoff is None:
                conversation_delete_row = conn.execute(
                    "DELETE FROM conversations WHERE user_id = %s RETURNING id",
                    (user_id,),
                ).fetchall()
                deleted_conversations = len(conversation_delete_row)
            else:
                orphan_rows = conn.execute(
                    """
                    DELETE FROM conversations c
                    WHERE c.user_id = %s
                      AND NOT EXISTS (
                          SELECT 1 FROM messages m WHERE m.conversation_id = c.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM conversation_summaries s WHERE s.conversation_id = c.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM documents d WHERE d.conversation_id = c.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM tool_calls t WHERE t.conversation_id = c.id
                      )
                    RETURNING id
                    """,
                    (user_id,),
                ).fetchall()
                deleted_conversations = len(orphan_rows)

            return {
                "counts": {
                    "messages": message_count,
                    "summaries": summary_count,
                    "memories": len(memory_ids),
                    "documents": len(document_ids),
                    "document_chunks": len(document_chunk_ids),
                    "tool_calls": tool_call_count,
                    "conversations": deleted_conversations,
                },
                "memory_point_ids": memory_point_ids,
                "document_point_ids": document_point_ids,
            }

    @staticmethod
    def _cutoff_clause(column: str, cutoff: datetime | None) -> str:
        if cutoff is None:
            return ""
        return f" AND {column} >= %s"

    @staticmethod
    def _cutoff_params(params: tuple[Any, ...], cutoff: datetime | None) -> tuple[Any, ...]:
        if cutoff is None:
            return params
        return params + (cutoff,)
