import json

import respx
from httpx import Response

from app.openwebui import OpenWebUIClient, extract_content, extract_sources, extract_stream_delta


def test_extract_content_and_sources() -> None:
    data = {
        "choices": [{"message": {"content": "hello [1]"}}],
        "sources": [
            {
                "document": [
                    '{"title": "Docs", "url": "https://example.test/docs"}',
                    {"name": "Guide", "url": "https://example.test/guide"},
                ]
            }
        ],
    }

    assert extract_content(data) == "hello [1]"
    assert extract_sources(data) == [
        {"title": "Docs", "url": "https://example.test/docs"},
        {"title": "Guide", "url": "https://example.test/guide"},
    ]


def test_extract_stream_delta() -> None:
    assert extract_stream_delta({"choices": [{"delta": {"content": "hello"}}]}) == "hello"


@respx.mock
def test_openwebui_client_creates_calls_and_deletes_chat() -> None:
    base_url = "http://openwebui.test"
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        return_value=Response(200, json={"id": "chat-1"})
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        return_value=Response(
            200,
            json={"choices": [{"message": {"content": "pong"}}]},
        )
    )
    delete = respx.delete(f"{base_url}/api/v1/chats/chat-1").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="model-a",
        fallback_model=None,
        ollama_base_url=None,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=True,
        embedding_model=None,
    )

    result = client.chat([{"role": "user", "content": "ping"}])

    assert result.content == "pong"
    assert completion.called
    sent = completion.calls.last.request.content.decode("utf-8")
    assert '"chat_id":"chat-1"' in sent
    assert '"tool_ids":["web_search"]' in sent
    assert delete.called


@respx.mock
def test_openwebui_client_stream_chat_parses_deltas() -> None:
    base_url = "http://openwebui.test"
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        return_value=Response(200, json={"id": "chat-1"})
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        return_value=Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"hel"}}]}\n'
                'data: {"choices":[{"delta":{"content":"lo"}}]}\n'
                "data: [DONE]\n"
            ),
        )
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-1").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="model-a",
        fallback_model=None,
        ollama_base_url=None,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=False,
        embedding_model=None,
    )

    events = list(client.stream_chat([{"role": "user", "content": "ping"}]))

    assert events == [
        {"type": "delta", "content": "hel"},
        {"type": "delta", "content": "lo"},
        {"type": "done", "content": "hello", "sources": []},
    ]
    sent = completion.calls.last.request.content.decode("utf-8")
    assert '"stream":true' in sent


@respx.mock
def test_gemma_chat_moves_system_messages_into_first_user_message() -> None:
    base_url = "http://openwebui.test"
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        return_value=Response(200, json={"id": "chat-1"})
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        return_value=Response(
            200,
            json={"choices": [{"message": {"content": "pong"}}]},
        )
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-1").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="gemma4:e4b",
        fallback_model=None,
        ollama_base_url=None,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=False,
        embedding_model=None,
    )

    client.chat(
        [
            {"role": "system", "content": "System A."},
            {"role": "system", "content": "System B."},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "next question"},
        ]
    )

    payload = json.loads(completion.calls.last.request.content.decode("utf-8"))
    assert all(message["role"] != "system" for message in payload["messages"])
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"].startswith("Instructions and context:\nSystem A.\n\nSystem B.")
    assert payload["messages"][0]["content"].endswith("User message:\nping")
    assert payload["messages"][1:] == [
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "next question"},
    ]


@respx.mock
def test_non_gemma_chat_leaves_system_messages_unchanged() -> None:
    base_url = "http://openwebui.test"
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        return_value=Response(200, json={"id": "chat-1"})
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        return_value=Response(
            200,
            json={"choices": [{"message": {"content": "pong"}}]},
        )
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-1").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="model-a",
        fallback_model=None,
        ollama_base_url=None,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=False,
        embedding_model=None,
    )
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "ping"},
    ]

    client.chat(messages)

    payload = json.loads(completion.calls.last.request.content.decode("utf-8"))
    assert payload["messages"] == messages


@respx.mock
def test_gemma_stream_chat_moves_system_messages_into_first_user_message() -> None:
    base_url = "http://openwebui.test"
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        return_value=Response(200, json={"id": "chat-1"})
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        return_value=Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
                "data: [DONE]\n"
            ),
        )
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-1").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="my-gemma-4-e4b",
        fallback_model=None,
        ollama_base_url=None,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=False,
        embedding_model=None,
    )

    events = list(
        client.stream_chat(
            [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "ping"},
            ]
        )
    )

    payload = json.loads(completion.calls.last.request.content.decode("utf-8"))
    assert events[-1] == {"type": "done", "content": "ok", "sources": []}
    assert all(message["role"] != "system" for message in payload["messages"])
    assert payload["messages"][0]["content"] == (
        "Instructions and context:\nSystem prompt.\n\nUser message:\nping"
    )


@respx.mock
def test_openwebui_client_chat_falls_back_to_backup_model() -> None:
    base_url = "http://openwebui.test"
    ollama_url = "http://ollama.test"
    respx.get(f"{ollama_url}/api/tags").mock(
        return_value=Response(200, json={"models": []})
    )
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        side_effect=[
            Response(200, json={"id": "chat-primary"}),
            Response(200, json={"id": "chat-fallback"}),
        ]
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        side_effect=[
            Response(500, json={"detail": "primary failed"}),
            Response(200, json={"choices": [{"message": {"content": "fallback pong"}}]}),
        ]
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-primary").mock(
        return_value=Response(200, json={"ok": True})
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-fallback").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="model-a",
        fallback_model="model-b",
        ollama_base_url=ollama_url,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=False,
        embedding_model=None,
    )

    result = client.chat([{"role": "user", "content": "ping"}])

    assert result.content == "fallback pong"
    first_sent = completion.calls[0].request.content.decode("utf-8")
    second_sent = completion.calls[1].request.content.decode("utf-8")
    assert '"model":"model-a"' in first_sent
    assert '"model":"model-b"' in second_sent


@respx.mock
def test_openwebui_client_uses_backup_when_ollama_status_fails() -> None:
    base_url = "http://openwebui.test"
    ollama_url = "http://ollama.test"
    respx.get(f"{ollama_url}/api/tags").mock(
        return_value=Response(504, json={"detail": "ollama too slow"})
    )
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        return_value=Response(200, json={"id": "chat-fallback"})
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        return_value=Response(200, json={"choices": [{"message": {"content": "fallback pong"}}]})
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-fallback").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="model-a",
        fallback_model="model-b",
        ollama_base_url=ollama_url,
        ollama_status_timeout_seconds=1,
        timeout_seconds=120,
        use_web_search=False,
        embedding_model=None,
    )

    result = client.chat([{"role": "user", "content": "ping"}])

    assert result.content == "fallback pong"
    generation_sent = completion.calls[0].request.content.decode("utf-8")
    assert str(completion.calls[0].request.url).startswith(base_url)
    assert '"model":"model-b"' in generation_sent


@respx.mock
def test_openwebui_client_stream_chat_falls_back_before_deltas() -> None:
    base_url = "http://openwebui.test"
    ollama_url = "http://ollama.test"
    respx.get(f"{ollama_url}/api/tags").mock(
        return_value=Response(200, json={"models": []})
    )
    respx.post(f"{base_url}/api/v1/chats/new").mock(
        side_effect=[
            Response(200, json={"id": "chat-primary"}),
            Response(200, json={"id": "chat-fallback"}),
        ]
    )
    completion = respx.post(f"{base_url}/api/chat/completions").mock(
        side_effect=[
            Response(500, json={"detail": "primary failed"}),
            Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
                    "data: [DONE]\n"
                ),
            ),
        ]
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-primary").mock(
        return_value=Response(200, json={"ok": True})
    )
    respx.delete(f"{base_url}/api/v1/chats/chat-fallback").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = OpenWebUIClient(
        base_url=base_url,
        api_key="secret",
        model="model-a",
        fallback_model="model-b",
        ollama_base_url=ollama_url,
        ollama_status_timeout_seconds=1,
        timeout_seconds=5,
        use_web_search=False,
        embedding_model=None,
    )

    events = list(client.stream_chat([{"role": "user", "content": "ping"}]))

    assert events == [
        {"type": "delta", "content": "ok"},
        {"type": "done", "content": "ok", "sources": []},
    ]
    first_sent = completion.calls[0].request.content.decode("utf-8")
    second_sent = completion.calls[1].request.content.decode("utf-8")
    assert '"model":"model-a"' in first_sent
    assert '"model":"model-b"' in second_sent
