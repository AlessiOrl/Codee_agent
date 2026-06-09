from app.documents import chunk_text, extract_text


def test_chunk_text_overlaps() -> None:
    text = "abcdefghij" * 300

    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) > 1
    assert chunks[0][-100:] == chunks[1][:100]


def test_extract_text_decodes_plain_text() -> None:
    assert extract_text("note.txt", b"hello", "text/plain") == "hello"
