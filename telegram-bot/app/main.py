import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


LOGGER = logging.getLogger("telegram-bot")
TELEGRAM_MARKDOWN_V2_SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!"
INTERNAL_PLACEHOLDER_RE = re.compile(r"@@(?:BOLD|CODEBLOCK|INLINECODE|MARKDOWNLINK)\d+@@")
SUPERSCRIPT_DIGITS = str.maketrans(
    {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    agent_api_url: str
    agent_api_key: str
    poll_interval_seconds: float
    telegram_timeout_seconds: float
    agent_timeout_seconds: float
    state_path: Path


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r; using default %s", name, value, default)
        return default


def load_settings() -> Settings:
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    agent_api_key = os.getenv("AGENT_API_KEY", "").strip()
    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    if not agent_api_key:
        raise RuntimeError("AGENT_API_KEY is required")

    return Settings(
        telegram_bot_token=telegram_bot_token,
        agent_api_url=os.getenv("AGENT_API_URL", "http://agent-api:8000").rstrip("/"),
        agent_api_key=agent_api_key,
        poll_interval_seconds=env_float("TELEGRAM_POLL_INTERVAL_SECONDS", 3.0),
        telegram_timeout_seconds=env_float("TELEGRAM_TIMEOUT_SECONDS", 30.0),
        agent_timeout_seconds=env_float("AGENT_API_TIMEOUT_SECONDS", 180.0),
        state_path=Path(os.getenv("TELEGRAM_STATE_PATH", "/data/telegram_offset.json")),
    )


class OffsetStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> int | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Could not read Telegram offset state: %s", exc)
            return None

        offset = data.get("offset")
        return int(offset) if isinstance(offset, int) else None

    def save(self, offset: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"offset": offset}), encoding="utf-8")
        tmp_path.replace(self.path)


class TelegramBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.offset_store = OffsetStore(settings.state_path)
        self.telegram_base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        self.offset = self.offset_store.load()

    def run_forever(self) -> None:
        LOGGER.info(
            "Telegram bot starting agent_api_url=%s offset_loaded=%s",
            self.settings.agent_api_url,
            self.offset is not None,
        )
        while True:
            try:
                self.poll_once()
            except Exception:
                LOGGER.exception("Unexpected polling error")
            time.sleep(self.settings.poll_interval_seconds)

    def poll_once(self) -> None:
        updates = self.get_updates()
        for update in updates:
            update_id = update.get("update_id")
            if not isinstance(update_id, int):
                LOGGER.warning("Skipping update without numeric update_id")
                continue

            try:
                self.handle_update(update)
            except Exception:
                LOGGER.exception("Failed to process update_id=%s", update_id)
            finally:
                self.offset = update_id + 1
                self.offset_store.save(self.offset)

    def get_updates(self) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": 0,
            "allowed_updates": json.dumps(["message"]),
        }
        if self.offset is not None:
            params["offset"] = self.offset

        try:
            with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                response = client.get(f"{self.telegram_base_url}/getUpdates", params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            LOGGER.warning("Telegram getUpdates failed: %s", exc)
            return []

        if not data.get("ok"):
            LOGGER.warning("Telegram getUpdates returned ok=false: %s", data)
            return []

        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def handle_update(self, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        message = update.get("message")
        if not isinstance(message, dict):
            LOGGER.info("Ignoring non-message update_id=%s", update_id)
            return

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            LOGGER.warning("Ignoring message without chat id update_id=%s", update_id)
            return

        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            LOGGER.info("Unsupported non-text message update_id=%s chat_id=%s", update_id, chat_id)
            self.send_message(chat_id, "I can handle text messages for now.")
            return

        if is_telegram_command(text):
            command = command_name(text)
            LOGGER.info("Handling command update_id=%s command=%s", update_id, command)
            if command == "/forget":
                self.send_message(chat_id, self.call_forget(message))
            elif command == "/summary":
                self.send_message(chat_id, self.call_summary(message))
            else:
                self.send_message(chat_id, command_reply(text, message))
            return

        LOGGER.info("Forwarding text update_id=%s chat_id=%s to agent-api", update_id, chat_id)
        if self.stream_agent_reply(message):
            return
        reply = self.call_agent(message)
        self.send_message(chat_id, reply)

    def call_agent(self, message: dict[str, Any]) -> str:
        payload = agent_payload(message)
        headers = {"Authorization": f"Bearer {self.settings.agent_api_key}"}
        try:
            with httpx.Client(timeout=self.settings.agent_timeout_seconds) as client:
                response = client.post(
                    f"{self.settings.agent_api_url}/chat",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "agent-api returned HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return "Codee hit an API error while answering. Please try again."
        except httpx.HTTPError as exc:
            LOGGER.warning("agent-api request failed: %s", exc)
            return "Codee could not reach the agent service. Please try again."
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("agent-api returned an invalid response: %s", exc)
            return "Codee received an invalid agent response. Please try again."

        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            LOGGER.warning("agent-api response missing reply")
            return "Codee did not return a reply. Please try again."
        sources = data.get("sources")
        return inject_source_citations(reply, sources if isinstance(sources, list) else [])

    def stream_agent_reply(self, message: dict[str, Any]) -> bool:
        user = message.get("from") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return False

        self.send_chat_action(chat_id)
        message_id = self.send_placeholder_message(chat_id, "Codee is thinking...")
        if message_id is None:
            return False

        payload = agent_payload(message)
        headers = {"Authorization": f"Bearer {self.settings.agent_api_key}"}
        accumulated = ""
        sources: list[dict[str, Any]] = []
        last_edit_at = 0.0
        last_typing_at = time.monotonic()
        last_edited_length = 0
        try:
            with httpx.Client(timeout=self.settings.agent_timeout_seconds) as client:
                with client.stream(
                    "POST",
                    f"{self.settings.agent_api_url}/chat/stream",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        now = time.monotonic()
                        if now - last_typing_at >= 4:
                            self.send_chat_action(chat_id)
                            last_typing_at = now
                        if not line:
                            continue
                        data_line = line.decode("utf-8") if isinstance(line, bytes) else line
                        try:
                            event = json.loads(data_line)
                        except json.JSONDecodeError:
                            LOGGER.debug("Skipping unparsable agent stream line")
                            continue

                        event_type = event.get("type")
                        if event_type == "delta":
                            delta = event.get("content")
                            if isinstance(delta, str):
                                accumulated += delta
                            enough_text = len(accumulated) - last_edited_length >= 80
                            enough_time = now - last_edit_at >= 1
                            if accumulated and enough_text and enough_time:
                                self.edit_message(chat_id, message_id, accumulated, markdown=False)
                                last_edit_at = now
                                last_edited_length = len(accumulated)
                        elif event_type == "done":
                            final_reply = event.get("reply")
                            if isinstance(final_reply, str) and final_reply.strip():
                                accumulated = final_reply
                            event_sources = event.get("sources")
                            if isinstance(event_sources, list):
                                sources = event_sources
                            final_text = inject_source_citations(accumulated, sources)
                            self.edit_message(chat_id, message_id, final_text, markdown=True)
                            return True
                        elif event_type == "error":
                            message_text = event.get("message")
                            LOGGER.warning("agent-api streaming error: %s", message_text)
                            if not accumulated:
                                fallback = self.call_agent(message)
                                self.edit_message(chat_id, message_id, fallback, markdown=True)
                                return True
                            self.edit_message(
                                chat_id,
                                message_id,
                                f"{accumulated}\n\nCodee stopped streaming before the answer fully finished.",
                                markdown=False,
                            )
                            return True
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "agent-api /chat/stream returned HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
        except httpx.HTTPError as exc:
            LOGGER.warning("agent-api /chat/stream request failed: %s", exc)

        if accumulated:
            self.edit_message(
                chat_id,
                message_id,
                f"{accumulated}\n\nCodee stopped streaming before the answer fully finished.",
                markdown=False,
            )
            return True
        fallback = self.call_agent(message)
        self.edit_message(chat_id, message_id, fallback, markdown=True)
        return True

    def call_forget(self, message: dict[str, Any]) -> str:
        parsed = parse_forget_command(message["text"])
        if isinstance(parsed, str):
            return parsed

        user = message.get("from") or {}
        chat = message.get("chat") or {}
        payload = {
            "telegram_user_id": str(user.get("id")),
            "telegram_chat_id": str(chat.get("id")),
            "days": parsed,
        }
        headers = {"Authorization": f"Bearer {self.settings.agent_api_key}"}
        try:
            with httpx.Client(timeout=self.settings.agent_timeout_seconds) as client:
                response = client.post(
                    f"{self.settings.agent_api_url}/forget",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "agent-api /forget returned HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return "Codee hit an API error while forgetting data. Please try again."
        except httpx.HTTPError as exc:
            LOGGER.warning("agent-api /forget request failed: %s", exc)
            return "Codee could not reach the agent service to forget data. Please try again."
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("agent-api /forget returned an invalid response: %s", exc)
            return "Codee received an invalid forget response. Please try again."

        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            LOGGER.warning("agent-api /forget response missing reply")
            return "Codee did not confirm the forget request. Please try again."
        return reply

    def call_summary(self, message: dict[str, Any]) -> str:
        user = message.get("from") or {}
        chat = message.get("chat") or {}
        payload = {
            "telegram_user_id": str(user.get("id")),
            "telegram_chat_id": str(chat.get("id")),
        }
        headers = {"Authorization": f"Bearer {self.settings.agent_api_key}"}
        try:
            with httpx.Client(timeout=self.settings.agent_timeout_seconds) as client:
                response = client.post(
                    f"{self.settings.agent_api_url}/summary",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "agent-api /summary returned HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return "Codee hit an API error while loading the summary. Please try again."
        except httpx.HTTPError as exc:
            LOGGER.warning("agent-api /summary request failed: %s", exc)
            return "Codee could not reach the agent service to load the summary. Please try again."
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("agent-api /summary returned an invalid response: %s", exc)
            return "Codee received an invalid summary response. Please try again."

        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            LOGGER.warning("agent-api /summary response missing reply")
            return "Codee did not return a summary. Please try again."
        return reply

    def send_message(self, chat_id: Any, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": render_telegram_markdown(text),
            "disable_web_page_preview": True,
            "parse_mode": "MarkdownV2",
        }
        try:
            with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                response = client.post(f"{self.telegram_base_url}/sendMessage", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "Telegram sendMessage rejected MarkdownV2 chat_id=%s status=%s: %s",
                chat_id,
                exc.response.status_code,
                exc.response.text[:500],
            )
            fallback_payload = {
                "chat_id": chat_id,
                "text": render_telegram_plain_text(text),
                "disable_web_page_preview": True,
            }
            try:
                with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                    response = client.post(f"{self.telegram_base_url}/sendMessage", json=fallback_payload)
                    response.raise_for_status()
            except httpx.HTTPError as fallback_exc:
                LOGGER.warning("Telegram fallback sendMessage failed chat_id=%s: %s", chat_id, fallback_exc)
        except httpx.HTTPError as exc:
            LOGGER.warning("Telegram sendMessage failed chat_id=%s: %s", chat_id, exc)

    def send_chat_action(self, chat_id: Any) -> None:
        payload = {"chat_id": chat_id, "action": "typing"}
        try:
            with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                response = client.post(f"{self.telegram_base_url}/sendChatAction", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Telegram sendChatAction failed chat_id=%s: %s", chat_id, exc)

    def send_placeholder_message(self, chat_id: Any, text: str) -> int | None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                response = client.post(f"{self.telegram_base_url}/sendMessage", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("Telegram placeholder sendMessage failed chat_id=%s: %s", chat_id, exc)
            return None

        result = data.get("result") if isinstance(data, dict) else None
        message_id = result.get("message_id") if isinstance(result, dict) else None
        return int(message_id) if isinstance(message_id, int) else None

    def edit_message(self, chat_id: Any, message_id: int, text: str, *, markdown: bool) -> bool:
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": render_telegram_markdown(text) if markdown else render_telegram_plain_text(text),
            "disable_web_page_preview": True,
        }
        if markdown:
            payload["parse_mode"] = "MarkdownV2"
        try:
            with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                response = client.post(f"{self.telegram_base_url}/editMessageText", json=payload)
                response.raise_for_status()
                return True
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "Telegram editMessageText failed chat_id=%s status=%s: %s",
                chat_id,
                exc.response.status_code,
                exc.response.text[:500],
            )
            if not markdown:
                return False
            fallback_payload = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": render_telegram_plain_text(text),
                "disable_web_page_preview": True,
            }
            try:
                with httpx.Client(timeout=self.settings.telegram_timeout_seconds) as client:
                    response = client.post(f"{self.telegram_base_url}/editMessageText", json=fallback_payload)
                    response.raise_for_status()
                    return True
            except httpx.HTTPError as fallback_exc:
                LOGGER.warning("Telegram fallback editMessageText failed chat_id=%s: %s", chat_id, fallback_exc)
                return False
        except httpx.HTTPError as exc:
            LOGGER.warning("Telegram editMessageText failed chat_id=%s: %s", chat_id, exc)
            return False


def is_telegram_command(text: str) -> bool:
    return text.strip().startswith("/")


def command_name(text: str) -> str:
    command = text.strip().split(maxsplit=1)[0].lower()
    return command.split("@", 1)[0]


def agent_payload(message: dict[str, Any]) -> dict[str, Any]:
    user = message.get("from") or {}
    chat = message.get("chat") or {}
    return {
        "telegram_user_id": str(user.get("id")),
        "telegram_chat_id": str(chat.get("id")),
        "telegram_message_id": str(message.get("message_id")) if message.get("message_id") is not None else None,
        "username": user.get("username"),
        "first_name": user.get("first_name"),
        "text": message["text"],
    }


def command_reply(text: str, message: dict[str, Any]) -> str:
    command = command_name(text)
    if command == "/start":
        user = message.get("from") or {}
        first_name = user.get("first_name") or "there"
        return (
            f"Hi {first_name}. Codee is online. Send me a message and "
            "I will answer with memory and document context when available."
        )
    if command == "/help":
        return help_reply()
    return "Command received. Send /help for available commands, or send a normal message to chat."


def help_reply() -> str:
    return (
        "Available commands:\n"
        "/start - show the welcome message\n"
        "/help - show this command list\n"
        "/summary - show the current chat summary\n"
        "/forget - remove all stored context for your user\n"
        "/forget <days> - remove stored context from the last N days\n\n"
        "Send a normal text message to chat with Codee."
    )


def parse_forget_command(text: str) -> int | None | str:
    parts = text.strip().split()
    if not parts:
        return "Usage: /forget or /forget <days>."

    command = command_name(parts[0])
    if command != "/forget":
        return "Usage: /forget or /forget <days>."

    if len(parts) == 1:
        return None
    if len(parts) != 2:
        return "Usage: /forget or /forget <days>."
    try:
        days = int(parts[1])
    except ValueError:
        return "Usage: /forget <days>, where <days> is a positive integer."
    if days < 1:
        return "Usage: /forget <days>, where <days> is a positive integer."
    return days


def sanitize_llm_text_for_telegram(text: str) -> str:
    if not text:
        return text

    rendered = text.replace("\r\n", "\n")
    rendered = INTERNAL_PLACEHOLDER_RE.sub("", rendered)
    rendered = _rewrite_visible_citation_urls(rendered)
    rendered = re.sub(r"(?m)^[ \t]*(?:\*{3,}|-{3,}|_{3,})[ \t]*$", "", rendered)
    rendered = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", r"**\1**", rendered)

    def replace_bullet(match: re.Match[str]) -> str:
        return f"{match.group(1)}\u2022 "

    rendered = re.sub(r"(?m)^([ \t]*)[*-][ \t]+(?=\S)", replace_bullet, rendered)
    return rendered.strip()


def render_telegram_plain_text(text: str) -> str:
    if not text:
        return text

    rendered = sanitize_llm_text_for_telegram(text)
    rendered = re.sub(
        r"```([^\n`]*)\n?(.*?)```",
        lambda match: match.group(2).strip("\n"),
        rendered,
        flags=re.DOTALL,
    )
    rendered = re.sub(r"`([^`\n]+)`", r"\1", rendered)
    rendered = re.sub(r"\[((?:\\.|[^\]\n])*)\]\(([^)\n]+)\)", r"\1", rendered)
    rendered = re.sub(r"\*\*([^\n*]+?)\*\*", r"\1", rendered)
    rendered = re.sub(r"(?<!\*)\*([^\n*]+?)\*(?!\*)", r"\1", rendered)
    rendered = rendered.replace("*", "")
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    rendered = re.sub(r" +([.,!?;:])", r"\1", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip()


def render_telegram_markdown(text: str) -> str:
    if not text:
        return text

    placeholders: dict[str, str] = {}
    rendered = sanitize_llm_text_for_telegram(text)
    rendered = _extract_and_render_code_blocks(rendered, placeholders)
    rendered = _extract_and_render_inline_code(rendered, placeholders)
    rendered = _extract_and_render_bold(rendered, placeholders)
    rendered = _extract_and_render_markdown_links(rendered, placeholders)
    rendered = escape_markdown_v2(rendered)

    for key, value in placeholders.items():
        rendered = rendered.replace(key, value)

    return rendered


def _extract_and_render_code_blocks(text: str, placeholders: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = _placeholder_token("CODEBLOCK", placeholders)
        language = (match.group(1) or "").strip()
        code = match.group(2).replace("\\", "\\\\").replace("`", "\\`")
        opening = f"```{language}\n" if language else "```\n"
        placeholders[token] = f"{opening}{code}```"
        return token

    return re.sub(r"```([^\n`]*)\n?(.*?)```", replace, text, flags=re.DOTALL)


def _extract_and_render_inline_code(text: str, placeholders: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = _placeholder_token("INLINECODE", placeholders)
        code = match.group(1).replace("\\", "\\\\").replace("`", "\\`")
        placeholders[token] = f"`{code}`"
        return token

    return re.sub(r"`([^`\n]+)`", replace, text)


def _extract_and_render_bold(text: str, placeholders: dict[str, str]) -> str:
    def replace_double_star(match: re.Match[str]) -> str:
        token = _placeholder_token("BOLD", placeholders)
        content = escape_markdown_v2(match.group(1).strip())
        placeholders[token] = f"*{content}*"
        return token

    def replace_single_star(match: re.Match[str]) -> str:
        token = _placeholder_token("BOLD", placeholders)
        content = escape_markdown_v2(match.group(1).strip())
        placeholders[token] = f"*{content}*"
        return token

    rendered = re.sub(r"\*\*([^\n*][^\n]*?[^\n*]?)\*\*", replace_double_star, text)
    return re.sub(r"(?<!\*)\*([^\n*][^\n]*?[^\n*]?)\*(?!\*)", replace_single_star, rendered)


def _extract_and_render_markdown_links(text: str, placeholders: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = _placeholder_token("MARKDOWNLINK", placeholders)
        label = match.group(1)
        url = _escape_markdown_link_url(match.group(2))
        placeholders[token] = f"[{label}]({url})"
        return token

    return re.sub(r"\[((?:\\.|[^\]\n])*)\]\(([^)\n]+)\)", replace, text)


def _placeholder_token(kind: str, placeholders: dict[str, str]) -> str:
    return f"\uE000{kind}{len(placeholders)}\uE001"


def _escape_markdown_link_url(url: str) -> str:
    return url.replace("\\", "\\\\").replace(")", "\\)")


def inject_source_citations(text: str, sources: list[dict[str, Any]]) -> str:
    if not text:
        return text
    text = _rewrite_visible_citation_urls(text)
    if not sources:
        return text

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(sources):
            return match.group(0)
        source = sources[index]
        url = source.get("url") if isinstance(source, dict) else None
        if not isinstance(url, str) or not url.strip():
            return match.group(0)
        label = superscript_citation_label(match.group(1))
        return f"[{label}]({url.strip()})"

    return re.sub(r"\[(\d+)\]", replace, text)


def _rewrite_visible_citation_urls(text: str) -> str:
    superscript_pattern = "(\u207d[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+\u207e)\\s*\\((https?://[^)\\s]+)\\)"
    text = re.sub(superscript_pattern, r"[\1](\2)", text)

    def replace_numbered(match: re.Match[str]) -> str:
        return f"[{superscript_citation_label(match.group(1))}]({match.group(2)})"

    return re.sub(r"\[(\d+)\]\s*\((https?://[^)\s]+)\)", replace_numbered, text)


def superscript_citation_label(number: str) -> str:
    return f"⁽{number.translate(SUPERSCRIPT_DIGITS)}⁾"


def escape_markdown_v2(text: str) -> str:
    return "".join(
        f"\\{char}" if char in TELEGRAM_MARKDOWN_V2_SPECIAL_CHARS else char
        for char in text
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    settings = load_settings()
    TelegramBot(settings).run_forever()


if __name__ == "__main__":
    main()
