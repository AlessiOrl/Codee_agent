import respx
from httpx import Response

from app.embeddings import Embeddings, OllamaEmbeddingsClient, extract_ollama_embedding


class FakeOpenWebUIClient:
    def embed(self, text: str) -> list[float]:
        return [9.0]


def test_extract_ollama_embedding_from_embed_response() -> None:
    data = {"embeddings": [[0.1, 0.2, 0.3]]}

    assert extract_ollama_embedding(data) == [0.1, 0.2, 0.3]


@respx.mock
def test_ollama_embeddings_client_calls_api_embed() -> None:
    base_url = "http://ollama.test"
    route = respx.post(f"{base_url}/api/embed").mock(
        return_value=Response(200, json={"embeddings": [[0.1, 0.2]]})
    )
    client = OllamaEmbeddingsClient(
        base_url=base_url,
        model="qwen3-embedding:4b",
        timeout_seconds=5,
    )

    result = client.embed("hello")

    assert result == [0.1, 0.2]
    assert route.called
    sent = route.calls.last.request.content.decode("utf-8")
    assert '"model":"qwen3-embedding:4b"' in sent
    assert '"input":"hello"' in sent


def test_explicit_ollama_provider_ignores_local_hash_flag() -> None:
    ollama = OllamaEmbeddingsClient(
        base_url="http://ollama.test",
        model="qwen3-embedding:4b",
        timeout_seconds=5,
    )
    ollama.embed = lambda text: [1.0, 2.0]  # type: ignore[method-assign]
    embeddings = Embeddings(
        client=FakeOpenWebUIClient(),  # type: ignore[arg-type]
        ollama_client=ollama,
        provider="ollama",
        vector_size=384,
        use_local_hash_embeddings=True,
    )

    assert embeddings.embed("hello") == [1.0, 2.0]
