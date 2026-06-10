import math
import re
from datetime import datetime, timezone
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def lexical_score(query: str, content: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    content_tokens = tokenize(content)
    if not content_tokens:
        return 0.0
    content_counts: dict[str, int] = {}
    for token in content_tokens:
        content_counts[token] = content_counts.get(token, 0) + 1
    matched = sum(1 for token in set(query_tokens) if token in content_counts)
    coverage = matched / max(1, len(set(query_tokens)))
    frequency = sum(content_counts.get(token, 0) for token in query_tokens)
    frequency_score = min(1.0, math.log1p(frequency) / 3.0)
    phrase_bonus = 0.15 if query.strip().lower() in content.lower() else 0.0
    return min(1.0, (coverage * 0.75) + (frequency_score * 0.25) + phrase_bonus)


def recency_score(value: Any) -> float:
    created_at = _parse_datetime(value)
    if created_at is None:
        return 0.0
    age_days = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 86400)
    return 1.0 / (1.0 + age_days / 14.0)


def hybrid_rerank(
    *,
    query: str,
    hits: list[dict[str, Any]],
    limit: int,
    allowed_channels: set[str] | None = None,
    allowed_source_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for hit in hits:
        payload = dict(hit.get("payload") or {})
        channel = payload.get("channel")
        if allowed_channels is not None and channel not in allowed_channels:
            continue
        source_type = payload.get("source_type")
        if allowed_source_types is not None and source_type not in allowed_source_types:
            continue
        content = str(payload.get("content") or "")
        if not content.strip():
            continue
        key = _hit_key(payload)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {**hit, "payload": payload}
            continue
        existing["score"] = max(_float_score(existing.get("score")), _float_score(hit.get("score")))
        existing["keyword_score"] = max(
            _float_score(existing.get("keyword_score")),
            _float_score(hit.get("keyword_score")),
        )
        sources = set(existing.get("retrieval_sources") or [])
        sources.update(hit.get("retrieval_sources") or [])
        if hit.get("retrieval_source"):
            sources.add(str(hit["retrieval_source"]))
        existing["retrieval_sources"] = sorted(sources)

    ranked = []
    for hit in merged.values():
        payload = hit["payload"]
        content = str(payload.get("content") or "")
        dense = _float_score(hit.get("score"))
        keyword = max(_float_score(hit.get("keyword_score")), lexical_score(query, content))
        recency = recency_score(payload.get("created_at"))
        importance = min(1.0, max(0.0, _float_score(payload.get("importance")) / 5.0))
        hybrid = (dense * 0.45) + (keyword * 0.40) + (recency * 0.10) + (importance * 0.05)
        hit["keyword_score"] = keyword
        hit["hybrid_score"] = hybrid
        if not hit.get("retrieval_sources"):
            source = hit.get("retrieval_source")
            hit["retrieval_sources"] = [source] if source else []
        ranked.append(hit)

    ranked.sort(
        key=lambda item: (
            _float_score(item.get("hybrid_score")),
            _float_score(item.get("score")),
            _float_score(item.get("keyword_score")),
        ),
        reverse=True,
    )
    return ranked[:limit]


def row_to_hit(row: dict[str, Any], *, source_type: str, retrieval_source: str = "keyword") -> dict[str, Any]:
    payload = {
        "source_type": source_type,
        "content": row.get("content"),
        "channel": row.get("channel"),
        "created_at": row.get("created_at"),
        "importance": row.get("importance"),
        "qdrant_point_id": row.get("qdrant_point_id"),
        "filename": row.get("filename"),
        "document_id": row.get("document_id"),
        "chunk_id": row.get("chunk_id") or row.get("id"),
        "chunk_index": row.get("chunk_index"),
        "telegram_chat_id": row.get("telegram_chat_id"),
        "telegram_message_id": row.get("telegram_message_id"),
        "memory_type": row.get("memory_type"),
        "confidence": row.get("confidence"),
    }
    return {
        "score": None,
        "keyword_score": _float_score(row.get("keyword_score")),
        "retrieval_source": retrieval_source,
        "retrieval_sources": [retrieval_source],
        "payload": {key: value for key, value in payload.items() if value is not None},
    }


def _hit_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    for key in ("qdrant_point_id", "chunk_id", "document_id", "telegram_message_id"):
        value = payload.get(key)
        if value:
            return (key, payload.get("channel"), value)
    return ("content", payload.get("channel"), str(payload.get("content") or "").strip().lower())


def _float_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
