from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    telegram_user_id TEXT UNIQUE NOT NULL,
    username TEXT,
    first_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    telegram_chat_id TEXT NOT NULL,
    channel TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, telegram_chat_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    telegram_message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
ON messages (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS context_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    channel TEXT NOT NULL,
    telegram_chat_id TEXT NOT NULL,
    telegram_message_id TEXT,
    content TEXT NOT NULL,
    qdrant_point_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_context_messages_user_channel_created
ON context_messages (user_id, channel, created_at);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    from_message_id UUID,
    to_message_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    channel TEXT,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    qdrant_point_id TEXT,
    importance INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    filename TEXT NOT NULL,
    mime_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    qdrant_point_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    tool_input JSONB,
    tool_output JSONB,
    status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS channel TEXT;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS channel TEXT;
UPDATE conversations
SET channel = telegram_chat_id
WHERE channel IS NULL;
UPDATE memories AS m
SET conversation_id = c.id,
    channel = c.channel
FROM conversations AS c
WHERE m.conversation_id IS NULL
  AND m.channel IS NULL
  AND c.user_id = m.user_id;
CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_user_channel
ON conversations (user_id, channel);
CREATE INDEX IF NOT EXISTS idx_conversations_channel
ON conversations (channel);
CREATE INDEX IF NOT EXISTS idx_memories_user_channel
ON memories (user_id, channel);
"""


class Database:
    def __init__(self, database_url: str):
        self.database_url = database_url

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            yield conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA_SQL)

    def health(self) -> str:
        with self.connect() as conn:
            conn.execute("SELECT 1")
        return "ok"
