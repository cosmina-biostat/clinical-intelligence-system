import pdfplumber
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    log.info(f"Extracted {len(text)} chars from {pdf_path}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    log.info(f"Created {len(chunks)} chunks")
    return chunks
