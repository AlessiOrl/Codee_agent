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
curl.exe http://localhost:8002/health
$body = @{ text = "Reply with pong." } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8002/debug/llm" `
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
CONTEXT_OWNER_TELEGRAM_USER_ID=your_telegram_user_id
DOMOTICS_CHAT_ID=your_domotics_chat_id
UNRAID_CHAT_ID=your_unraid_chat_id
PLEX_CHAT_ID=your_plex_chat_id

LOG_LEVEL=INFO
DOCKER_LOG_MAX_SIZE=10m
DOCKER_LOG_MAX_FILE=3
POSTGRES_LOG_MIN_MESSAGES=warning
POSTGRES_LOG_MIN_ERROR_STATEMENT=error
QDRANT_LOG_LEVEL=warn

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
CROSS_CHANNEL_MEMORY_LIMIT=3
CROSS_CHANNEL_DOC_LIMIT=2
CROSS_CHANNEL_MIN_SCORE=0.6
CONTEXT_ROUTER_ENABLED=true
CONTEXT_ROUTER_MAX_FEED_MESSAGES=6
CONTEXT_ROUTER_MIN_CONFIDENCE=0.4
CONTEXT_DEBUG_JSON=false
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
- `CONTEXT_OWNER_TELEGRAM_USER_ID` attaches passive feed messages to your private user context.
- `DOMOTICS_CHAT_ID`, `UNRAID_CHAT_ID`, and `PLEX_CHAT_ID` map passive Telegram feeds to the fixed logical channels used for storage and context routing.
- `LOG_LEVEL` controls Python service verbosity with compact one-line logs. `DOCKER_LOG_MAX_SIZE` and `DOCKER_LOG_MAX_FILE` keep every container log bounded.
- `POSTGRES_LOG_MIN_MESSAGES`, `POSTGRES_LOG_MIN_ERROR_STATEMENT`, and `QDRANT_LOG_LEVEL` reduce routine infrastructure noise.
- `CONTEXT_ROUTER_ENABLED`, `CONTEXT_ROUTER_MAX_FEED_MESSAGES`, and `CONTEXT_ROUTER_MIN_CONFIDENCE` control the LangGraph router that decides whether passive feed context is useful for a private chat question.
- `CONTEXT_DEBUG_JSON=true` prints JSON-shaped debug events for Telegram feed routing, context ingestion, LangGraph router decisions, feed retrieval, and prompt context assembly. These logs include message text, so leave it off except while debugging.

Notes:

- Change `POSTGRES_PASSWORD`, `AGENT_API_KEY`, `OPENWEBUI_API_KEY`, and `TELEGRAM_BOT_TOKEN` from the placeholder values before starting the stack.
- If you change `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB`, update `DATABASE_URL` to match.
- `OPENWEBUI_MODEL` must be an actual model id exposed by your Open WebUI instance.
- If `/debug/llm` reports that Open WebUI redirected to `/login`, check that `OPENWEBUI_BASE_URL` includes the correct host and port for Open WebUI, that `OPENWEBUI_API_KEY` is a valid token, and that `OPENWEBUI_MODEL` matches an available model id.

## Services

- `agent-api`: FastAPI app with LangGraph orchestration.
- `telegram-bot`: Python polling bot that receives Telegram messages and sends replies.
- `postgres`: durable state for users, conversations, messages, summaries, source-backed memories, documents, passive feed messages, and tool-call logs.
- `qdrant`: vector store for user memories, document chunks, and passive feed messages.

Open WebUI remains external. The agent calls its native API:

```text
POST /api/v1/chats/new
POST /api/chat/completions
DELETE /api/v1/chats/{id}
```

Embeddings default to local hash embeddings so the stack can run without an embedding service. Set `EMBEDDING_PROVIDER=ollama`, `OLLAMA_EMBEDDINGS_BASE_URL=...`, `EMBEDDING_MODEL=...`, and `VECTOR_SIZE=...` to call Ollama directly via `POST /api/embed`. Set `EMBEDDING_PROVIDER=openwebui` to use Open WebUI's `POST /api/embeddings` endpoint instead.

Retrieval uses a hybrid local pipeline. Qdrant provides dense semantic candidates, Postgres full-text search provides keyword candidates, and `agent-api` reranks them with a deterministic score that combines semantic similarity, keyword overlap, recency, and memory importance. Passive feed retrieval still only happens after the LLM router selects `domotics`, `unraid`, or `plex`; keyword search never bypasses that router decision.

If existing Qdrant collections were created with a different vector size, the app keeps them intact and creates dimension-specific collections such as `user_memories_2560` and `documents_2560`.

## Endpoints

- `GET /health`
- `POST /debug/llm`
- `POST /debug/embedding`
- `POST /chat/stream`
- `POST /context/messages`
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
- `CONTEXT_OWNER_TELEGRAM_USER_ID` in `.env`.
- `DOMOTICS_CHAT_ID`, `UNRAID_CHAT_ID`, and `PLEX_CHAT_ID` in `.env`.
- `AGENT_API_KEY` in `.env`; it must match the key used by `agent-api`.
- `AGENT_API_URL=http://agent-api:8000` when running in Compose.

Telegram commands are handled locally by `telegram-bot` and do not call `agent-api`:

- `/start`
- `/help`
- `/summary`
- `/forget`
- `/forget <days>`
- `/forget_all`
- `/forget_all <days>`

The `/summary` and `/forget` commands are scoped to the private chat. `/forget_all` removes private chat data plus passive feed data for that user.

Any other slash command is handled locally.

Normal private-chat text messages are forwarded to `POST /chat/stream`. Configured `domotics`, `unraid`, and `plex` chats are passive feeds only: the bot stores their messages through `POST /context/messages` and never replies there. For private chats, LangGraph first asks an LLM router whether any passive feed is relevant. Only the router-selected feeds are searched, then semantic, keyword, and recent-feed candidates are reranked into the prompt context. The bot sends a typing action, creates a placeholder message, and edits that message as streamed text arrives. Chat responses use the streaming endpoint only. Final replies are sent with `parse_mode=MarkdownV2` so markdown-style replies, code fences, and clickable citations render correctly. If Telegram rejects the formatting, the bot automatically retries the final reply as plain text. Non-text messages in private chat receive a local unsupported-message reply.

Smoke test in Telegram:

```text
/start
/help
/summary
/forget
/forget 1
/forget_all
Reply with pong.
```

Document upload support is implemented in `agent-api`, but the Python Telegram bot currently handles text only. Uploaded text and PDFs are chunked with structure-aware rules for headings, paragraphs, and log-style timestamps before falling back to overlapping fixed-size windows.

## Retrieval Evaluation

A small dependency-light retrieval evaluator is included for quick checks of hybrid ranking behavior:

```powershell
python agent-api\scripts\evaluate_retrieval.py
```

It prints JSON with `recall_at_k`, `mrr`, `context_precision`, and average ranking latency. You can pass a custom case file with:

```powershell
python agent-api\scripts\evaluate_retrieval.py --cases path\to\cases.json
```

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

Routine `/health` access logs, HTTP client request logs, Postgres info logs, and Qdrant info logs are silenced by default. Python services use compact logs like `I agent-api | Agent API initialized`. Set `CONTEXT_DEBUG_JSON=true` when diagnosing passive feed routing or retrieval; set `LOG_LEVEL=DEBUG` or `QDRANT_LOG_LEVEL=info` only when you need deeper infrastructure debugging.
