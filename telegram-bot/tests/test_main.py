import unittest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from app.main import (
    Settings,
    TelegramBot,
    command_name,
    command_reply,
    help_reply,
    inject_source_citations,
    is_telegram_command,
    parse_forget_command,
    render_telegram_markdown,
    render_telegram_plain_text,
    sanitize_llm_text_for_telegram,
    superscript_citation_label,
)


def json_line(data: dict) -> str:
    return json.dumps(data)


def make_settings() -> Settings:
    return Settings(
        telegram_bot_token="token",
        agent_api_url="http://agent-api:8000",
        agent_api_key="key",
        poll_interval_seconds=3.0,
        telegram_timeout_seconds=30.0,
        agent_timeout_seconds=180.0,
        state_path=Path("telegram-offset.json"),
        domotics_chat_id="2",
        unraid_chat_id="22",
        plex_chat_id="222",
        context_owner_telegram_user_id="1",
        context_debug_json=False,
    )


class TelegramBotHelpersTest(unittest.TestCase):
    def test_command_name_strips_bot_suffix(self) -> None:
        self.assertEqual(command_name("/start@CodeeBot hello"), "/start")

    def test_command_replies_are_local(self) -> None:
        message = {"from": {"first_name": "Alessio"}}

        self.assertTrue(command_reply("/start", message).startswith("Hi Alessio."))
        self.assertEqual(command_reply("/help", message), help_reply())
        self.assertIn("/summary", command_reply("/help", message))
        self.assertIn("/forget <days>", command_reply("/help", message))
        self.assertTrue(command_reply("/unknown", message).startswith("Command received."))

    def test_is_telegram_command(self) -> None:
        self.assertTrue(is_telegram_command("/start"))
        self.assertFalse(is_telegram_command("hello"))

    def test_parse_forget_command(self) -> None:
        self.assertIsNone(parse_forget_command("/forget"))
        self.assertEqual(parse_forget_command("/forget 3"), 3)
        self.assertIn("positive integer", parse_forget_command("/forget abc"))
        self.assertIn("positive integer", parse_forget_command("/forget -1"))
        self.assertIsNone(parse_forget_command("/forget_all", expected_command="/forget_all"))

    def test_render_telegram_markdown_escapes_plain_text(self) -> None:
        rendered = render_telegram_markdown("Hello - world. Use [brackets] safely!")

        self.assertEqual(rendered, r"Hello \- world\. Use \[brackets\] safely\!")

    def test_render_telegram_markdown_preserves_code(self) -> None:
        rendered = render_telegram_markdown("Use `pip install`.\n```python\nprint('ok')\n```")

        self.assertEqual(
            rendered,
            "Use `pip install`\\.\n```python\nprint('ok')\n```",
        )

    def test_render_telegram_markdown_preserves_inline_links(self) -> None:
        rendered = render_telegram_markdown("See [⁽¹⁾](https://example.test/docs?a=1&b=2).")

        self.assertEqual(
            rendered,
            "See [⁽¹⁾](https://example.test/docs?a=1&b=2)\\.",
        )

    def test_render_telegram_markdown_preserves_bold(self) -> None:
        rendered = render_telegram_markdown("The story of *Pokemon Unbound* has **Ancient War:** lore.")

        self.assertEqual(
            rendered,
            "The story of *Pokemon Unbound* has *Ancient War:* lore\\.",
        )

    def test_render_telegram_markdown_preserves_bold_inside_bullet_text(self) -> None:
        rendered = render_telegram_markdown("*   **Ancient War:** 3000 years ago.")

        self.assertEqual(
            rendered,
            "\u2022 *Ancient War:* 3000 years ago\\.",
        )

    def test_sanitize_llm_text_for_telegram_normalizes_common_markdown(self) -> None:
        sanitized = sanitize_llm_text_for_telegram(
            "***\n\n"
            "### Topic 1: Formula 1\n\n"
            "*   **Ancient War:** 3000 years ago.\n"
            "Pokemon Unboundis based on thePokemon FireRed* game @@BOLD7@@.\n"
            "Source \u207d\u00b9\u207e (https://example.test/source)."
        )

        self.assertNotIn("@@BOLD", sanitized)
        self.assertNotIn("###", sanitized)
        self.assertNotIn("***", sanitized)
        self.assertIn("**Topic 1: Formula 1**", sanitized)
        self.assertIn("\u2022 **Ancient War:** 3000 years ago.", sanitized)
        self.assertIn("[\u207d\u00b9\u207e](https://example.test/source)", sanitized)

    def test_render_telegram_markdown_handles_bad_llm_markdown_without_placeholder_leak(self) -> None:
        rendered = render_telegram_markdown(
            "***\n\n"
            "### Topic 1: Formula 1\n\n"
            "*   **Ancient War:** 3000 years ago.\n"
            "Pokemon Unboundis based on thePokemon FireRed* game @@BOLD7@@."
        )

        self.assertNotIn("@@BOLD", rendered)
        self.assertNotIn("\\#\\#\\#", rendered)
        self.assertNotIn("\\*\\*\\*", rendered)
        self.assertIn("*Topic 1: Formula 1*", rendered)
        self.assertIn("\u2022 *Ancient War:* 3000 years ago\\.", rendered)
        self.assertIn("FireRed\\* game", rendered)

    def test_render_telegram_plain_text_strips_markdown_and_hides_links(self) -> None:
        rendered = render_telegram_plain_text(
            "### Topic\n"
            "Answer [\u207d\u00b9\u207e](https://example.test/source) with **bold** text @@BOLD7@@."
        )

        self.assertEqual(rendered, "Topic\nAnswer \u207d\u00b9\u207e with bold text.")

    def test_inject_source_citations_rewrites_numbered_references(self) -> None:
        rendered = inject_source_citations(
            "Answer with citations [1][2][3].",
            [
                {"title": "Docs", "url": "https://example.test/docs"},
                {"title": "Guide", "url": "https://example.test/guide"},
            ],
        )

        self.assertEqual(
            rendered,
            "Answer with citations [⁽¹⁾](https://example.test/docs)[⁽²⁾](https://example.test/guide)[3].",
        )

    def test_inject_source_citations_hides_visible_source_urls(self) -> None:
        rendered = inject_source_citations(
            "The story of *Pokemon Unbound* is set in Borrius ⁽¹⁾ (https://example.test/source).",
            [],
        )

        self.assertEqual(
            rendered,
            "The story of *Pokemon Unbound* is set in Borrius [⁽¹⁾](https://example.test/source).",
        )

    def test_superscript_citation_label_supports_multiple_digits(self) -> None:
        self.assertEqual(superscript_citation_label("1"), "⁽¹⁾")
        self.assertEqual(superscript_citation_label("12"), "⁽¹²⁾")

    @patch("app.main.httpx.Client")
    def test_send_message_retries_plain_text_after_markdown_failure(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        first_response = MagicMock()
        first_response.status_code = 400
        first_response.text = "Bad Request: can't parse entities"
        first_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad markdown",
            request=httpx.Request("POST", "https://api.telegram.org/bot/sendMessage"),
            response=httpx.Response(400, request=httpx.Request("POST", "https://api.telegram.org/bot/sendMessage")),
        )

        second_response = MagicMock()
        second_response.raise_for_status.return_value = None

        client = MagicMock()
        client.post.side_effect = [first_response, second_response]
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        bot.send_message(123, "Hello - world")

        first_payload = client.post.call_args_list[0].kwargs["json"]
        second_payload = client.post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["parse_mode"], "MarkdownV2")
        self.assertEqual(first_payload["text"], r"Hello \- world")
        self.assertNotIn("parse_mode", second_payload)
        self.assertEqual(second_payload["text"], "Hello - world")

    @patch("app.main.httpx.Client")
    def test_call_forget_posts_full_reset_to_agent_api(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"reply": "Forgot all stored data."}

        client = MagicMock()
        client.post.return_value = response
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        reply = bot.call_forget(
            {
                "text": "/forget",
                "from": {"id": 1},
                "chat": {"id": 2},
            },
            channel="private",
            scope="channel",
        )

        self.assertEqual(reply, "Forgot all stored data.")
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["telegram_user_id"], "1")
        self.assertEqual(payload["telegram_chat_id"], "2")
        self.assertEqual(payload["channel"], "private")
        self.assertEqual(payload["scope"], "channel")
        self.assertIsNone(payload["days"])

    @patch("app.main.httpx.Client")
    def test_call_forget_posts_recent_delete_to_agent_api(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"reply": "Forgot stored data from the last 3 day(s)."}

        client = MagicMock()
        client.post.return_value = response
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        reply = bot.call_forget(
            {
                "text": "/forget 3",
                "from": {"id": 1},
                "chat": {"id": 2},
            },
            channel="private",
            scope="channel",
        )

        self.assertEqual(reply, "Forgot stored data from the last 3 day(s).")
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["days"], 3)

    @patch("app.main.httpx.Client")
    def test_call_summary_posts_to_agent_api(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"reply": "Current chat summary:\nUser is testing Codee."}

        client = MagicMock()
        client.post.return_value = response
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        reply = bot.call_summary(
            {
                "text": "/summary",
                "from": {"id": 1},
                "chat": {"id": 2},
            },
            channel="private",
        )

        self.assertEqual(reply, "Current chat summary:\nUser is testing Codee.")
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["telegram_user_id"], "1")
        self.assertEqual(payload["telegram_chat_id"], "2")
        self.assertEqual(payload["channel"], "private")

    def test_inject_source_citations_formats_multiple_sources(self) -> None:
        reply = inject_source_citations(
            "Kimi Antonelli won [1][2].",
            [
                {"title": "Race recap", "url": "https://example.test/race"},
                {"title": "Monaco report", "url": "https://example.test/monaco"},
            ],
        )

        self.assertEqual(
            reply,
            "Kimi Antonelli won [⁽¹⁾](https://example.test/race)[⁽²⁾](https://example.test/monaco).",
        )

    @patch("app.main.httpx.Client")
    def test_stream_agent_reply_edits_placeholder_and_final_citations(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)
        bot.send_chat_action = MagicMock()
        bot.send_placeholder_message = MagicMock(return_value=44)
        bot.edit_message = MagicMock(return_value=True)

        long_delta = "A" * 90
        stream_response = MagicMock()
        stream_response.raise_for_status.return_value = None
        stream_response.iter_lines.return_value = [
            json_line({"type": "delta", "content": long_delta}),
            json_line(
                {
                    "type": "done",
                    "reply": "Kimi Antonelli won [1].",
                    "sources": [{"title": "Race recap", "url": "https://example.test/race"}],
                }
            ),
        ]
        stream_context = MagicMock()
        stream_context.__enter__.return_value = stream_response
        stream_context.__exit__.return_value = None

        client = MagicMock()
        client.stream.return_value = stream_context
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        handled = bot.stream_agent_reply(
            {
                "text": "Who won?",
                "message_id": 3,
                "from": {"id": 1, "username": "orlando", "first_name": "Orlando"},
                "chat": {"id": 2},
            },
            channel="private",
        )

        self.assertTrue(handled)
        bot.send_chat_action.assert_called_with(2)
        bot.send_placeholder_message.assert_called_with(2, "Codee is thinking...")
        self.assertEqual(client.stream.call_args.args[1], "http://agent-api:8000/chat/stream")
        partial_edit = bot.edit_message.call_args_list[0]
        final_edit = bot.edit_message.call_args_list[-1]
        self.assertEqual(partial_edit.args[:3], (2, 44, long_delta))
        self.assertFalse(partial_edit.kwargs["markdown"])
        self.assertEqual(
            final_edit.args[:3],
            (2, 44, f"Kimi Antonelli won [{superscript_citation_label('1')}](https://example.test/race)."),
        )
        self.assertTrue(final_edit.kwargs["markdown"])

    def test_handle_update_reports_when_streaming_does_not_start(self) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)
        bot.stream_agent_reply = MagicMock(return_value=False)
        bot.send_message = MagicMock()

        bot.handle_update(
            {
                "update_id": 10,
                "message": {
                    "text": "hello",
                    "message_id": 3,
                    "from": {"id": 1},
                    "chat": {"id": 999, "type": "private"},
                },
            }
        )

        bot.stream_agent_reply.assert_called()
        bot.send_message.assert_called_with(999, "Codee could not start a streaming reply. Please try again.")

    def test_handle_update_ignores_unmapped_non_private_chat(self) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)
        bot.send_message = MagicMock()
        bot.stream_agent_reply = MagicMock()

        bot.handle_update(
            {
                "update_id": 11,
                "message": {
                    "text": "hello",
                    "message_id": 3,
                    "from": {"id": 1},
                    "chat": {"id": 999, "type": "group"},
                },
            }
        )

        bot.stream_agent_reply.assert_not_called()
        bot.send_message.assert_not_called()

    def test_handle_update_stores_feed_message_without_reply(self) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)
        bot.call_context_ingest = MagicMock(return_value=True)
        bot.send_message = MagicMock()
        bot.stream_agent_reply = MagicMock()

        bot.handle_update(
            {
                "update_id": 12,
                "channel_post": {
                    "text": "Garage door opened.",
                    "message_id": 4,
                    "chat": {"id": 2, "type": "channel"},
                },
            }
        )

        bot.call_context_ingest.assert_called_once()
        self.assertEqual(bot.call_context_ingest.call_args.kwargs["channel"], "domotics")
        bot.stream_agent_reply.assert_not_called()
        bot.send_message.assert_not_called()

    @patch("app.main.httpx.Client")
    def test_call_context_ingest_posts_passive_message(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        response = MagicMock()
        response.raise_for_status.return_value = None

        client = MagicMock()
        client.post.return_value = response
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        ok = bot.call_context_ingest(
            {
                "text": "Array parity check started.",
                "message_id": 8,
                "chat": {"id": 22, "type": "channel"},
            },
            channel="unraid",
        )

        self.assertTrue(ok)
        self.assertEqual(client.post.call_args.args[0], "http://agent-api:8000/context/messages")
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["telegram_user_id"], "1")
        self.assertEqual(payload["telegram_chat_id"], "22")
        self.assertEqual(payload["channel"], "unraid")

    @patch("app.main.httpx.Client")
    def test_edit_message_retries_plain_text_after_markdown_failure(self, client_cls: MagicMock) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        first_response = MagicMock()
        first_response.status_code = 400
        first_response.text = "Bad Request: can't parse entities"
        first_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad markdown",
            request=httpx.Request("POST", "https://api.telegram.org/bot/editMessageText"),
            response=httpx.Response(400, request=httpx.Request("POST", "https://api.telegram.org/bot/editMessageText")),
        )
        second_response = MagicMock()
        second_response.raise_for_status.return_value = None

        client = MagicMock()
        client.post.side_effect = [first_response, second_response]
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        client_cls.return_value = client

        ok = bot.edit_message(123, 44, "Hello - world", markdown=True)

        self.assertTrue(ok)
        first_payload = client.post.call_args_list[0].kwargs["json"]
        second_payload = client.post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["parse_mode"], "MarkdownV2")
        self.assertEqual(first_payload["text"], r"Hello \- world")
        self.assertNotIn("parse_mode", second_payload)
        self.assertEqual(second_payload["text"], "Hello - world")

    def test_call_forget_rejects_invalid_syntax_locally(self) -> None:
        settings = make_settings()
        bot = TelegramBot(settings)

        reply = bot.call_forget(
            {
                "text": "/forget nope",
                "from": {"id": 1},
                "chat": {"id": 2},
            },
            channel="private",
            scope="channel",
        )

        self.assertIn("positive integer", reply)


if __name__ == "__main__":
    unittest.main()
