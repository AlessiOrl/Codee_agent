from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import hybrid_rerank  # noqa: E402


DEFAULT_CASES = [
    {
        "id": "passive-feed-keyword",
        "query": "What is the garage sensor battery in domotics?",
        "expected_ids": ["domotics-garage-battery"],
        "allowed_channels": ["domotics"],
        "candidates": [
            {
                "id": "domotics-garage-battery",
                "score": 0.62,
                "keyword_score": 1.0,
                "payload": {
                    "source_type": "context_message",
                    "channel": "domotics",
                    "content": "Garage sensor battery is 9 percent.",
                    "created_at": "2026-06-09T20:00:00Z",
                },
            },
            {
                "id": "plex-noise",
                "score": 0.97,
                "keyword_score": 1.0,
                "payload": {
                    "source_type": "context_message",
                    "channel": "plex",
                    "content": "Plex battery documentary is available.",
                },
            },
        ],
    },
    {
        "id": "memory-conflict",
        "query": "What project am I evaluating?",
        "expected_ids": ["memory-hybrid"],
        "allowed_channels": ["private"],
        "candidates": [
            {
                "id": "memory-old",
                "score": 0.9,
                "keyword_score": 0.2,
                "payload": {
                    "channel": "private",
                    "content": "User previously evaluated a notification parser.",
                    "importance": 2,
                    "created_at": "2026-01-01T00:00:00Z",
                },
            },
            {
                "id": "memory-hybrid",
                "score": 0.75,
                "keyword_score": 1.0,
                "payload": {
                    "channel": "private",
                    "content": "User is evaluating hybrid retrieval for Codee.",
                    "importance": 4,
                    "created_at": "2026-06-09T20:00:00Z",
                },
            },
        ],
    },
    {
        "id": "no-relevant-context",
        "query": "What is my passport number?",
        "expected_ids": [],
        "allowed_channels": ["private"],
        "candidates": [
            {
                "id": "memory-unrelated",
                "score": 0.1,
                "keyword_score": 0.0,
                "payload": {
                    "channel": "private",
                    "content": "User prefers concise replies.",
                    "importance": 3,
                },
            }
        ],
    },
]


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_case(case: dict[str, Any], *, limit: int) -> dict[str, Any]:
    start = time.perf_counter()
    ranked = hybrid_rerank(
        query=case["query"],
        hits=case.get("candidates", []),
        limit=limit,
        allowed_channels=set(case.get("allowed_channels", [])) or None,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    ranked_ids = [hit.get("id") or hit.get("payload", {}).get("id") for hit in ranked]
    expected = list(case.get("expected_ids", []))
    hits = [item for item in ranked_ids if item in expected]
    first_rank = next((index + 1 for index, item in enumerate(ranked_ids) if item in expected), None)
    return {
        "id": case["id"],
        "ranked_ids": ranked_ids,
        "recall": 1.0 if not expected else len(set(hits)) / len(set(expected)),
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "context_precision": 1.0 if not ranked_ids else len(hits) / len(ranked_ids),
        "latency_ms": elapsed_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Codee retrieval reranking on golden cases.")
    parser.add_argument("--cases", type=Path, default=None, help="Optional JSON file with evaluation cases.")
    parser.add_argument("--limit", type=int, default=5, help="Number of context items to keep per case.")
    args = parser.parse_args()

    results = [evaluate_case(case, limit=args.limit) for case in load_cases(args.cases)]
    count = max(1, len(results))
    summary = {
        "cases": len(results),
        "recall_at_k": sum(item["recall"] for item in results) / count,
        "mrr": sum(item["mrr"] for item in results) / count,
        "context_precision": sum(item["context_precision"] for item in results) / count,
        "avg_latency_ms": sum(item["latency_ms"] for item in results) / count,
    }
    print(json.dumps({"summary": summary, "cases": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
