# Agentic Codee Stack

This directory contains the side-by-side replacement stack for Codee:

```text
Telegram -> telegram-bot -> agent-api -> LangGraph -> Open WebUI native API
                                  |-> Postgres
                                  |-> Qdrant
```

The existing root `codee.py` polling bot is not modified by this implementation.

## Quick Start

1. Copy `.env.example` to `.env`.
2. Fill in the values in `.env`.
3. Start the stack:

```powershell
docker compose up --build
```

4. Smoke test:

```powershell
curl.exe http://localhost:8000/health
$body = @{ text = "Reply with pong." } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/debug/llm" `
  -Headers @{ Authorization = "Bearer change_me_agent_api_key" } `
  -ContentType "application/json" `
  -Body $body
```

## Environment Variables

Docker Compose reads values from the root `.env` file automatically. Start by copying `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and replace the placeholder values:

```dotenv
POSTGRES_USER=agent
POSTGRES_PASSWORD=your_database_password
POSTGRES_DB=agent_db
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

AGENT_API_KEY=your_random_agent_api_key
AGENT_API_URL=http://agent-api:8000
DATABASE_URL=postgresql://agent:your_database_password@postgres:5432/agent_db
QDRANT_URL=http://qdrant:6333

OPENWEBUI_BASE_URL=http://host.docker.internal:6969
OPENWEBUI_API_KEY=your_openwebui_api_token
OPENWEBUI_MODEL=your_model_id
OPENWEBUI_FALLBACK_MODEL=optional_backup_model_id
OPENWEBUI_USE_WEB_SEARCH=false

EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_EMBEDDINGS_BASE_URL=http://host.docker.internal:11434
OLLAMA_STATUS_TIMEOUT_SECONDS=4
EMBEDDING_MODEL=qwen3-embedding:4b
VECTOR_SIZE=2560
```

How these values are used:

- `POSTGRES_USER` and `POSTGRES_PASSWORD` create the Postgres login inside the `postgres` container.
- `DATABASE_URL` must use the same Postgres username and password so `agent-api` can connect to `postgres` over the Compose network.
- `AGENT_API_KEY` protects the `agent-api` HTTP endpoints. Use this same value in your `curl` calls and in the `telegram-bot` service.
- `AGENT_API_URL` is the URL the Telegram bot uses to call `agent-api`; inside Compose, `http://agent-api:8000` is correct.
- `OPENWEBUI_API_KEY` is the token from your external Open WebUI instance.
- `OPENWEBUI_BASE_URL` is the fixed address that the `agent-api` container uses to reach Open WebUI for chat. If Open WebUI runs on your Windows host, `http://host.docker.internal:<port>` is correct.
- `EMBEDDING_PROVIDER` selects `hash`, `openwebui`, or `ollama`. Use `ollama` to call Ollama directly for embeddings while keeping Open WebUI for chat.
- `OPENWEBUI_FALLBACK_MODEL` is optional. If set, Codee can use this model on the same `OPENWEBUI_BASE_URL` when the primary Ollama server is unavailable.
- `OLLAMA_STATUS_TIMEOUT_SECONDS` controls the fast Ollama status check before generation. If the primary Ollama server does not respond within this short timeout, Codee sends the same Open WebUI chat request with `OPENWEBUI_FALLBACK_MODEL` instead of `OPENWEBUI_MODEL`.
- `OLLAMA_BASE_URL` is used only for the fast primary Ollama status check before chat generation.
- `OLLAMA_EMBEDDINGS_BASE_URL` is the address that the `agent-api` container uses to reach Ollama. For your Qwen3 embedding model, set `EMBEDDING_MODEL=qwen3-embedding:4b` and `VECTOR_SIZE=2560`.
- `TELEGRAM_BOT_TOKEN` lets the Python Telegram bot call Telegram `getUpdates` and `sendMessage` without a public HTTPS webhook.

Notes:

- Change `POSTGRES_PASSWORD`, `AGENT_API_KEY`, `OPENWEBUI_API_KEY`, and `TELEGRAM_BOT_TOKEN` from the placeholder values before starting the stack.
- If you change `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB`, update `DATABASE_URL` to match.
- `OPENWEBUI_MODEL` must be an actual model id exposed by your Open WebUI instance.
- If `/debug/llm` reports that Open WebUI redirected to `/login`, check that `OPENWEBUI_BASE_URL` includes the correct host and port for Open WebUI, that `OPENWEBUI_API_KEY` is a valid token, and that `OPENWEBUI_MODEL` matches an available model id.

## Services

- `agent-api`: FastAPI app with LangGraph orchestration.
- `telegram-bot`: Python polling bot that receives Telegram messages and sends replies.
- `postgres`: durable state for users, conversations, messages, summaries, memories, documents, and tool-call logs.
- `qdrant`: vector store for user memories and document chunks.

Open WebUI remains external. The agent calls its native API:

```text
POST /api/v1/chats/new
POST /api/chat/completions
DELETE /api/v1/chats/{id}
```

Embeddings default to local hash embeddings so the stack can run without an embedding service. Set `EMBEDDING_PROVIDER=ollama`, `OLLAMA_EMBEDDINGS_BASE_URL=...`, `EMBEDDING_MODEL=...`, and `VECTOR_SIZE=...` to call Ollama directly via `POST /api/embed`. Set `EMBEDDING_PROVIDER=openwebui` to use Open WebUI's `POST /api/embeddings` endpoint instead.

If existing Qdrant collections were created with a different vector size, the app keeps them intact and creates dimension-specific collections such as `user_memories_2560` and `documents_2560`.

## Endpoints

- `GET /health`
- `POST /debug/llm`
- `POST /debug/embedding`
- `POST /chat`
- `POST /chat/stream`
- `POST /forget`
- `POST /summary`
- `POST /documents/upload`

All non-health endpoints require:

```text
Authorization: Bearer ${AGENT_API_KEY}
```

## Telegram Bot

The `telegram-bot` service polls Telegram every few seconds, so it works on a local network without HTTPS.

Configure:

- `TELEGRAM_BOT_TOKEN` in `.env`.
- `AGENT_API_KEY` in `.env`; it must match the key used by `agent-api`.
- `AGENT_API_URL=http://agent-api:8000` when running in Compose.

Telegram commands are handled locally by `telegram-bot` and do not call `agent-api`:

- `/start`
- `/help`

The `/forget` command is forwarded to `agent-api` and deletes stored user data across chats:

- `/summary`
- `/forget`
- `/forget <days>`

Any other slash command is handled locally.

Normal text messages are forwarded to `POST /chat/stream`. The bot sends a typing action, creates a placeholder message, and edits that message as streamed text arrives. If streaming cannot start, it falls back to `POST /chat`. Final replies are sent with `parse_mode=MarkdownV2` so markdown-style replies, code fences, and clickable citations render correctly. If Telegram rejects the formatting, the bot automatically retries the final reply as plain text. Non-text messages receive a local unsupported-message reply.

Smoke test in Telegram:

```text
/start
/help
/summary
/forget
/forget 1
Reply with pong.
```

Document upload support is implemented in `agent-api`, but the Python Telegram bot currently handles text only.

## Logs

Use service-specific logs when debugging:

```powershell
docker compose logs --tail 40 telegram-bot
docker compose logs --tail 40 agent-api
```

Normal useful bot logs look like:

```text
Telegram bot starting agent_api_url=http://agent-api:8000 offset_loaded=True
Handling command locally update_id=... command=/start
Forwarding text update_id=... chat_id=... to agent-api
```

Routine `/health` access logs and HTTP client request logs are silenced by default. Set `LOG_LEVEL=DEBUG` in `.env` only when you need deeper debugging.
