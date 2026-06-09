from uuid import UUID, uuid4

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.embeddings import Embeddings


USER_MEMORIES_COLLECTION = "user_memories"
DOCUMENTS_COLLECTION = "documents"


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, *, url: str, embeddings: Embeddings, vector_size: int):
        self.client = QdrantClient(url=url)
        self.embeddings = embeddings
        self.vector_size = vector_size
        self.user_memories_collection = USER_MEMORIES_COLLECTION
        self.documents_collection = DOCUMENTS_COLLECTION

    def init_collections(self) -> None:
        self.user_memories_collection = self._collection_for_vector_size(USER_MEMORIES_COLLECTION)
        self.documents_collection = self._collection_for_vector_size(DOCUMENTS_COLLECTION)
        for collection in (self.user_memories_collection, self.documents_collection):
            if not self.client.collection_exists(collection):
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

    def health(self) -> str:
        self.client.get_collections()
        return "ok"

    def upsert_memory(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
        memory_type: str,
        importance: int,
    ) -> str:
        point_id = str(uuid4())
        self.client.upsert(
            collection_name=self.user_memories_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=self.embeddings.embed(content),
                    payload={
                        "user_id": str(user_id),
                        "conversation_id": str(conversation_id),
                        "content": content,
                        "memory_type": memory_type,
                        "importance": importance,
                    },
                )
            ],
        )
        return point_id

    def search_memories(self, *, user_id: UUID, query: str, limit: int = 5) -> list[dict]:
        return self._search(
            collection=self.user_memories_collection,
            query=query,
            limit=limit,
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))],
        )

    def upsert_document_chunk(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID | None,
        document_id: UUID,
        chunk_id: UUID,
        chunk_index: int,
        filename: str,
        content: str,
    ) -> str:
        point_id = str(uuid4())
        self.client.upsert(
            collection_name=self.documents_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=self.embeddings.embed(content),
                    payload={
                        "user_id": str(user_id),
                        "conversation_id": str(conversation_id) if conversation_id else None,
                        "document_id": str(document_id),
                        "chunk_id": str(chunk_id),
                        "chunk_index": chunk_index,
                        "filename": filename,
                        "content": content,
                    },
                )
            ],
        )
        return point_id

    def search_documents(self, *, user_id: UUID, query: str, limit: int = 5) -> list[dict]:
        return self._search(
            collection=self.documents_collection,
            query=query,
            limit=limit,
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))],
        )

    def delete_memory_points(self, point_ids: list[str]) -> None:
        self._delete_points(self.user_memories_collection, point_ids)

    def delete_document_points(self, point_ids: list[str]) -> None:
        self._delete_points(self.documents_collection, point_ids)

    def _collection_for_vector_size(self, collection: str) -> str:
        if not self.client.collection_exists(collection):
            return collection
        existing_size = self._collection_vector_size(collection)
        if existing_size in {None, self.vector_size}:
            return collection
        return f"{collection}_{self.vector_size}"

    def _collection_vector_size(self, collection: str) -> int | None:
        info = self.client.get_collection(collection)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        if isinstance(size, int):
            return size
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            named_size = getattr(first, "size", None)
            if isinstance(named_size, int):
                return named_size
        return None

    def _search(
        self,
        *,
        collection: str,
        query: str,
        limit: int,
        must: list[models.FieldCondition],
    ) -> list[dict]:
        query_vector = self.embeddings.embed(query)
        query_filter = models.Filter(must=must)
        try:
            if hasattr(self.client, "query_points"):
                result = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                hits = result.points
            else:
                hits = self.client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
        except UnexpectedResponse as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc
        return [
            {"score": hit.score, "payload": dict(hit.payload or {})}
            for hit in hits
        ]

    def _delete_points(self, collection: str, point_ids: list[str]) -> None:
        if not point_ids:
            return
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=point_ids),
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc
