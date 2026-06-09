from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agent_api_key: str = Field(default="change_me", alias="AGENT_API_KEY")
    database_url: str = Field(alias="DATABASE_URL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")

    openwebui_base_url: str = Field(alias="OPENWEBUI_BASE_URL")
    openwebui_api_key: str = Field(alias="OPENWEBUI_API_KEY")
    openwebui_model: str = Field(alias="OPENWEBUI_MODEL")
    openwebui_fallback_model: str | None = Field(default=None, alias="OPENWEBUI_FALLBACK_MODEL")
    openwebui_use_web_search: bool = Field(default=False, alias="OPENWEBUI_USE_WEB_SEARCH")
    openwebui_timeout_seconds: int = Field(default=120, alias="OPENWEBUI_TIMEOUT_SECONDS")

    embedding_provider: str | None = Field(default=None, alias="EMBEDDING_PROVIDER")
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_embeddings_base_url: str = Field(
        default="http://host.docker.internal:11434",
        alias="OLLAMA_EMBEDDINGS_BASE_URL",
    )
    ollama_status_timeout_seconds: float = Field(default=4.0, alias="OLLAMA_STATUS_TIMEOUT_SECONDS")
    embedding_model: str | None = Field(default=None, alias="EMBEDDING_MODEL")
    use_local_hash_embeddings: bool = Field(default=True, alias="USE_LOCAL_HASH_EMBEDDINGS")
    vector_size: int = Field(default=384, alias="VECTOR_SIZE")

    system_prompt: str = Field(
        default="You are Codee, a private assistant. Answer clearly and practically.",
        alias="SYSTEM_PROMPT",
    )
    recent_history_limit: int = Field(default=20, alias="RECENT_HISTORY_LIMIT")
    summary_every_n_messages: int = Field(default=12, alias="SUMMARY_EVERY_N_MESSAGES")
    max_document_chars: int = Field(default=200000, alias="MAX_DOCUMENT_CHARS")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
