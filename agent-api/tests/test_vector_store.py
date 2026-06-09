from unittest.mock import MagicMock

from qdrant_client.http import models

from app.vector_store import VectorStore


def test_delete_memory_points_uses_qdrant_point_ids() -> None:
    store = object.__new__(VectorStore)
    store.client = MagicMock()
    store.user_memories_collection = "user_memories_2560"

    store.delete_memory_points(["mem-1", "mem-2"])

    kwargs = store.client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "user_memories_2560"
    assert kwargs["points_selector"] == models.PointIdsList(points=["mem-1", "mem-2"])


def test_delete_document_points_is_noop_for_empty_list() -> None:
    store = object.__new__(VectorStore)
    store.client = MagicMock()
    store.documents_collection = "documents_2560"

    store.delete_document_points([])

    store.client.delete.assert_not_called()
