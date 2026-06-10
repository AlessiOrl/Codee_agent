import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

LOGGER = logging.getLogger(__name__)


@dataclass
class LlmResult:
    content: str
    sources: list[dict[str, str | None]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class OpenWebUIError(RuntimeError):
    pass


def _openwebui_error(exc: httpx.HTTPError) -> OpenWebUIError:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        location = response.headers.get("location")
        detail = f"Open WebUI returned HTTP {response.status_code}"
        if location:
            detail += f" and redirected to {location}"
        if response.status_code in {301, 302, 303, 307, 308}:
            detail += ". Check OPENWEBUI_BASE_URL, OPENWEBUI_API_KEY, and OPENWEBUI_MODEL."
        return OpenWebUIError(detail)
    return OpenWebUIError(f"Open WebUI request failed: {exc}")


def _error_detail(exc: BaseException) -> str:
    return str(exc).replace("\n", " ")[:300]


def _source_documents(data: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for source in data.get("sources", []):
        if not isinstance(source, dict):
            continue
        for raw_document in source.get("document", []):
            parsed = raw_document
            if isinstance(raw_document, str):
                try:
                    parsed = json.loads(raw_document)
                except json.JSONDecodeError:
                    LOGGER.debug("Skipping unparsable Open WebUI source document")
                    continue
            if isinstance(parsed, dict):
                documents.append(parsed)
            elif isinstance(parsed, list):
                documents.extend(item for item in parsed if isinstance(item, dict))
    return documents


def extract_sources(data: dict[str, Any]) -> list[dict[str, str | None]]:
    seen: set[tuple[str | None, str | None]] = set()
    sources: list[dict[str, str | None]] = []
    for document in _source_documents(data):
        title = document.get("title") or document.get("name")
        url = document.get("url")
        source = (str(title) if title else None, str(url) if url else None)
        if source in seen or (source[0] is None and source[1] is None):
            continue
        seen.add(source)
        sources.append({"title": source[0], "url": source[1]})
    return sources


def extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
    content = data.get("content") or data.get("response")
    if isinstance(content, str):
        return content
    raise ValueError("Could not extract assistant content from Open WebUI response")


def extract_stream_delta(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta", {})
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            return content
        text = choices[0].get("text")
        if isinstance(text, str):
            return text
    content = data.get("content") or data.get("response")
    return content if isinstance(content, str) else ""


def is_gemma_model(model: str) -> bool:
    return "gemma" in model.lower()


def normalize_messages_for_model(messages: list[dict[str, str]], *, model: str) -> list[dict[str, str]]:
    if not is_gemma_model(model):
        return [dict(message) for message in messages]

    system_parts = [
        message.get("content", "").strip()
        for message in messages
        if message.get("role") == "system" and message.get("content", "").strip()
    ]
    normalized = [dict(message) for message in messages if message.get("role") != "system"]
    if not system_parts:
        return normalized

    system_text = "\n\n".join(system_parts)
    prefix = f"Instructions and context:\n{system_text}\n\nUser message:\n"
    for message in normalized:
        if message.get("role") == "user":
            message["content"] = prefix + message.get("content", "")
            return normalized

    return [{"role": "user", "content": system_text}, *normalized]


class OpenWebUIClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        ollama_base_url: str | None = None,
        ollama_status_timeout_seconds: float = 4.0,
        timeout_seconds: int,
        use_web_search: bool,
        embedding_model: str | None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model.strip() if fallback_model and fallback_model.strip() else None
        self.ollama_base_url = ollama_base_url.rstrip("/") if ollama_base_url else None
        self.ollama_status_timeout_seconds = ollama_status_timeout_seconds
        self.timeout_seconds = timeout_seconds
        self.use_web_search = use_web_search
        self.embedding_model = embedding_model or model

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def chat(self, messages: list[dict[str, str]], *, use_web_search: bool | None = None) -> LlmResult:
        last_error: OpenWebUIError | None = None
        routes = self._chat_routes_for_request()
        for index, route in enumerate(routes):
            try:
                return self._chat_with_model(
                    messages,
                    base_url=route["base_url"],
                    model=route["model"],
                    use_web_search=self.use_web_search if use_web_search is None else use_web_search,
                )
            except OpenWebUIError as exc:
                last_error = exc
                if index < len(routes) - 1:
                    LOGGER.warning(
                        "Open WebUI chat failed route=%s model=%s url=%s; trying next route: %s",
                        index + 1,
                        route["model"],
                        route["base_url"],
                        _error_detail(exc),
                    )
                    continue
                raise
        if last_error:
            raise last_error
        raise OpenWebUIError("No Open WebUI model configured")

    def _chat_with_model(
        self,
        messages: list[dict[str, str]],
        *,
        base_url: str,
        model: str,
        use_web_search: bool,
    ) -> LlmResult:
        chat_id: str | None = None
        client = httpx.Client(timeout=self.timeout_seconds, headers=self.headers)
        try:
            try:
                chat_id = self._create_chat(client, base_url=base_url, model=model)
            except OpenWebUIError as exc:
                LOGGER.info(
                    "Open WebUI chat creation failed url=%s model=%s; continuing without chat_id: %s",
                    base_url,
                    model,
                    _error_detail(exc),
                )

            payload: dict[str, Any] = {
                "model": model,
                "messages": normalize_messages_for_model(messages, model=model),
                "stream": False,
            }
            if chat_id:
                payload["chat_id"] = chat_id
                payload["parent_id"] = None
            if use_web_search:
                payload["tool_ids"] = ["web_search"]

            try:
                response = client.post(f"{base_url}/api/chat/completions", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                error = _openwebui_error(exc)
                LOGGER.warning(
                    "Open WebUI chat completion request failed url=%s model=%s: %s",
                    base_url,
                    model,
                    _error_detail(error),
                )
                raise error from exc
            data = response.json()
            return LlmResult(
                content=extract_content(data),
                sources=extract_sources(data),
                raw=data,
            )
        finally:
            client.close()
            if chat_id:
                self._delete_chat(base_url=base_url, chat_id=chat_id)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        use_web_search: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        last_error: OpenWebUIError | None = None
        routes = self._chat_routes_for_request()
        for index, route in enumerate(routes):
            emitted_delta = False
            try:
                for event in self._stream_chat_with_model(
                    messages,
                    base_url=route["base_url"],
                    model=route["model"],
                    use_web_search=self.use_web_search if use_web_search is None else use_web_search,
                ):
                    if event.get("type") == "delta":
                        emitted_delta = True
                    yield event
                return
            except OpenWebUIError as exc:
                last_error = exc
                if index < len(routes) - 1 and not emitted_delta:
                    LOGGER.warning(
                        "Open WebUI stream failed before content route=%s model=%s url=%s; trying next route: %s",
                        index + 1,
                        route["model"],
                        route["base_url"],
                        _error_detail(exc),
                    )
                    continue
                raise
        if last_error:
            raise last_error
        raise OpenWebUIError("No Open WebUI model configured")

    def _stream_chat_with_model(
        self,
        messages: list[dict[str, str]],
        *,
        base_url: str,
        model: str,
        use_web_search: bool,
    ) -> Iterator[dict[str, Any]]:
        chat_id: str | None = None
        client = httpx.Client(timeout=self.timeout_seconds, headers=self.headers)
        content_parts: list[str] = []
        sources: list[dict[str, str | None]] = []
        try:
            try:
                chat_id = self._create_chat(client, base_url=base_url, model=model)
            except OpenWebUIError as exc:
                LOGGER.info(
                    "Open WebUI chat creation failed url=%s model=%s; continuing without chat_id: %s",
                    base_url,
                    model,
                    _error_detail(exc),
                )

            payload: dict[str, Any] = {
                "model": model,
                "messages": normalize_messages_for_model(messages, model=model),
                "stream": True,
            }
            if chat_id:
                payload["chat_id"] = chat_id
                payload["parent_id"] = None
            if use_web_search:
                payload["tool_ids"] = ["web_search"]

            try:
                with client.stream(
                    "POST",
                    f"{base_url}/api/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data_line = line.decode("utf-8") if isinstance(line, bytes) else line
                        if data_line.startswith("data:"):
                            data_line = data_line.removeprefix("data:").strip()
                        if data_line == "[DONE]":
                            break
                        try:
                            data = json.loads(data_line)
                        except json.JSONDecodeError:
                            LOGGER.debug("Skipping unparsable Open WebUI stream line")
                            continue

                        delta = extract_stream_delta(data)
                        if delta:
                            content_parts.append(delta)
                            yield {"type": "delta", "content": delta}
                        extracted_sources = extract_sources(data)
                        if extracted_sources:
                            sources = extracted_sources
            except httpx.HTTPError as exc:
                error = _openwebui_error(exc)
                LOGGER.warning(
                    "Open WebUI stream request failed url=%s model=%s: %s",
                    base_url,
                    model,
                    _error_detail(error),
                )
                raise error from exc

            yield {
                "type": "done",
                "content": "".join(content_parts),
                "sources": sources,
            }
        finally:
            client.close()
            if chat_id:
                self._delete_chat(base_url=base_url, chat_id=chat_id)

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "input": text}
        try:
            with httpx.Client(timeout=self.timeout_seconds, headers=self.headers) as client:
                response = client.post(f"{self.base_url}/api/embeddings", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            error = _openwebui_error(exc)
            LOGGER.warning(
                "Open WebUI embedding request failed url=%s model=%s: %s",
                self.base_url,
                self.embedding_model,
                _error_detail(error),
            )
            raise error from exc
        if isinstance(data.get("embedding"), list):
            return [float(value) for value in data["embedding"]]
        if isinstance(data.get("data"), list) and data["data"]:
            embedding = data["data"][0].get("embedding")
            if isinstance(embedding, list):
                return [float(value) for value in embedding]
        raise ValueError("Could not extract embedding from Open WebUI response")

    def _chat_routes(self) -> list[dict[str, Any]]:
        routes = [{"base_url": self.base_url, "model": self.model, "is_primary": True}]
        if self._has_fallback_route():
            routes.append(
                {
                    "base_url": self.base_url,
                    "model": self.fallback_model or self.model,
                    "is_primary": False,
                }
            )
        return routes

    def _chat_routes_for_request(self) -> list[dict[str, Any]]:
        routes = self._chat_routes()
        if len(routes) == 1:
            return routes
        if self._ollama_status_succeeds():
            return routes
        LOGGER.warning(
            "Primary Ollama did not pass fast status check url=%s; trying fallback model=%s before primary model=%s",
            self.ollama_base_url,
            routes[1]["model"],
            routes[0]["model"],
        )
        return [routes[1], routes[0]]

    def _has_fallback_route(self) -> bool:
        return bool(self.fallback_model and self.fallback_model != self.model)

    def _ollama_status_succeeds(self) -> bool:
        if not self.ollama_base_url:
            return True
        try:
            with httpx.Client(
                timeout=self.ollama_status_timeout_seconds,
            ) as client:
                response = client.get(f"{self.ollama_base_url}/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Ollama status check failed url=%s endpoint=/api/tags: %s", self.ollama_base_url, exc)
            return False
        return True

    def _create_chat(self, client: httpx.Client, *, base_url: str, model: str) -> str:
        payload = {"chat": {"model": model, "title": "Codee API chat", "messages": []}}
        try:
            response = client.post(f"{base_url}/api/v1/chats/new", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise _openwebui_error(exc) from exc
        return str(response.json()["id"])

    def _delete_chat(self, *, base_url: str, chat_id: str) -> None:
        try:
            with httpx.Client(timeout=self.timeout_seconds, headers=self.headers) as client:
                response = client.delete(f"{base_url}/api/v1/chats/{chat_id}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "Failed to delete temporary Open WebUI chat url=%s chat_id=%s: %s",
                base_url,
                chat_id,
                exc,
            )
