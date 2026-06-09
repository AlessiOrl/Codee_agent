from io import BytesIO
from uuid import UUID

from pypdf import PdfReader

from app.repository import Repository
from app.vector_store import VectorStore


def extract_text(filename: str, content: bytes, mime_type: str | None) -> str:
    lower_name = filename.lower()
    if mime_type == "application/pdf" or lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="replace")


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


class DocumentService:
    def __init__(self, *, repository: Repository, vector_store: VectorStore, max_chars: int):
        self.repository = repository
        self.vector_store = vector_store
        self.max_chars = max_chars

    def ingest(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID | None,
        channel: str | None,
        filename: str,
        mime_type: str | None,
        content: bytes,
    ) -> tuple[UUID, int]:
        text = extract_text(filename, content, mime_type)[: self.max_chars]
        chunks = chunk_text(text)
        document = self.repository.create_document(
            user_id=user_id,
            conversation_id=conversation_id,
            filename=filename,
            mime_type=mime_type,
        )
        for index, chunk in enumerate(chunks):
            placeholder = self.repository.save_document_chunk(
                document_id=document["id"],
                chunk_index=index,
                content=chunk,
                qdrant_point_id="pending",
            )
            point_id = self.vector_store.upsert_document_chunk(
                user_id=user_id,
                conversation_id=conversation_id,
                channel=channel,
                document_id=document["id"],
                chunk_id=placeholder["id"],
                chunk_index=index,
                filename=filename,
                content=chunk,
            )
            self.repository.update_document_chunk_point(
                chunk_id=placeholder["id"],
                qdrant_point_id=point_id,
            )
        return document["id"], len(chunks)
