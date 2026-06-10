from io import BytesIO
import re
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


HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+\S+|[A-Z][A-Z0-9 _/-]{6,}|[-=]{3,})\s*$")
LOG_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 150,
    filename: str | None = None,
    mime_type: str | None = None,
) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    blocks = _structure_blocks(normalized, filename=filename, mime_type=mime_type)
    if len(blocks) == 1 and len(blocks[0]) > chunk_size:
        return _split_long_block(blocks[0], chunk_size=chunk_size, overlap=overlap)

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_block(block, chunk_size=chunk_size, overlap=overlap))
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        prefix = "" if _is_heading_block(block) else _overlap_prefix(chunks[-1], overlap)
        current = prefix + block if chunks else block
        if len(current) > chunk_size:
            chunks.extend(_split_long_block(current, chunk_size=chunk_size, overlap=overlap))
            current = ""
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _structure_blocks(text: str, *, filename: str | None, mime_type: str | None) -> list[str]:
    lower_name = (filename or "").lower()
    if lower_name.endswith((".log", ".txt")) or mime_type in {"text/plain", "text/x-log"}:
        log_blocks = _log_blocks(text)
        if len(log_blocks) > 1:
            return log_blocks

    paragraph_blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(paragraph_blocks) > 1:
        return _split_heading_blocks(paragraph_blocks)
    return [text]


def _split_heading_blocks(blocks: list[str]) -> list[str]:
    structured: list[str] = []
    current = ""
    for block in blocks:
        first_line = block.splitlines()[0] if block.splitlines() else ""
        if current and HEADING_RE.match(first_line):
            structured.append(current.strip())
            current = block
        else:
            current = f"{current}\n\n{block}".strip() if current else block
    if current:
        structured.append(current.strip())
    return structured


def _is_heading_block(block: str) -> bool:
    first_line = block.splitlines()[0] if block.splitlines() else ""
    return bool(HEADING_RE.match(first_line))


def _log_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if current and LOG_RE.match(line):
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _split_long_block(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _overlap_prefix(previous: str, overlap: int) -> str:
    if overlap <= 0 or not previous:
        return ""
    return previous[-overlap:].strip() + "\n\n"


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
        chunks = chunk_text(text, filename=filename, mime_type=mime_type)
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
