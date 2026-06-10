from app.documents import chunk_text, extract_text


def test_chunk_text_overlaps() -> None:
    text = "abcdefghij" * 300

    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) > 1
    assert chunks[0][-100:] == chunks[1][:100]


def test_extract_text_decodes_plain_text() -> None:
    assert extract_text("note.txt", b"hello", "text/plain") == "hello"


def test_chunk_text_keeps_log_entries_as_structure_boundaries() -> None:
    text = "\n".join(
        [
            "2026-06-09 10:00:00 INFO garage opened",
            "details one",
            "2026-06-09 10:01:00 WARN battery low",
            "details two",
        ]
    )

    chunks = chunk_text(text, chunk_size=80, overlap=10, filename="home.log", mime_type="text/plain")

    assert len(chunks) == 2
    assert "garage opened" in chunks[0]
    assert "battery low" in chunks[1]


def test_chunk_text_splits_on_heading_blocks() -> None:
    text = (
        "# Setup\n\n"
        "Install the stack and configure environment values.\n\n"
        "# Retrieval\n\n"
        "Use hybrid recall and reranking for context."
    )

    chunks = chunk_text(text, chunk_size=80, overlap=10, filename="notes.md", mime_type="text/markdown")

    assert len(chunks) == 2
    assert chunks[0].startswith("# Setup")
    assert chunks[1].startswith("# Retrieval")
