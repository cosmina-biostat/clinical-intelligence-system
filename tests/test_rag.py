from src.rag.pdf_parser import chunk_text

def test_chunk_text_basic():
    text  = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)

def test_chunk_overlap():
    text   = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # With overlap, adjacent chunks should share some words
    assert len(chunks) >= 4
