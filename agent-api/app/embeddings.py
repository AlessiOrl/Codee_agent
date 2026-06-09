import hashlib
import math
from typing import Any

import httpx

from app.openwebui import OpenWebUIClient


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbeddingsClient:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

        return extract_ollama_embedding(data)


def extract_ollama_embedding(data: dict[str, Any]) -> list[float]:
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, list):
            return [float(value) for value in first]

    embedding = data.get("embedding")
    if isinstance(embedding, list):
        return [float(value) for value in embedding]

    raise EmbeddingError("Could not extract embedding from Ollama response")


class Embeddings:
    def __init__(
        self,
        *,
        client: OpenWebUIClient,
        ollama_client: OllamaEmbeddingsClient | None,
        provider: str,
        vector_size: int,
        use_local_hash_embeddings: bool,
    ):
        self.client = client
        self.ollama_client = ollama_client
        self.provider = provider.lower()
        self.vector_size = vector_size
        self.use_local_hash_embeddings = use_local_hash_embeddings

    def embed(self, text: str) -> list[float]:
        if self.provider == "hash":
            return hash_embedding(text, self.vector_size)
        if self.provider == "openwebui":
            return self.client.embed(text)
        if self.provider == "ollama" and self.ollama_client:
            return self.ollama_client.embed(text)
        raise EmbeddingError(f"Unsupported embedding provider: {self.provider}")


def hash_embedding(text: str, vector_size: int) -> list[float]:
    vector = [0.0] * vector_size
    tokens = [token for token in text.lower().split() if token.strip()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % vector_size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
