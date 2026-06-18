from pathlib import Path
from typing import Optional
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

# ── Heart pipeline (uploaded modules) ─────────────────────────────────────────
from backend.protocol_parser import get_schema, get_rag_store
from backend.extractor import process_documents

# ── ML inference (this chat's models) ─────────────────────────────────────────
from backend.inference import assess, classify_review, predict_quality, models_status

app = FastAPI(title="ClinOrigin AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Server-side schema registry (protocol_id -> schema dict) ──────────────────
SCHEMA_REGISTRY: dict[str, dict] = {}
PROTOCOL_DIR = Path(__file__).parent / "protocols"
PROTOCOL_DIR.mkdir(exist_ok=True)


def _protocol_id(pdf_path: str) -> str:
    return Path(pdf_path).stem


# ── Request models ────────────────────────────────────────────────────────────
class ParseRequest(BaseModel):
    pdf_path: str                      # server-side path to the protocol PDF


class ProcessRequest(BaseModel):
    protocol_id: str                   # which cached schema to use
    documents: list[str]               # extracted source texts (one per doc)


class FeaturesRequest(BaseModel):
    features: dict                     # 10 meta-features


class AnalyzeRequest(BaseModel):
    protocol_id: str
    documents: list[str]


class AskRequest(BaseModel):
    pdf_path: str                      # protocol to query
    question: str
    top_k: int = 4


# ── Phase 1: parse protocol -> schema (cached) ────────────────────────────────
@app.post("/protocol/parse")
def parse_protocol_endpoint(req: ParseRequest):
    if not Path(req.pdf_path).exists():
        raise HTTPException(404, f"PDF not found: {req.pdf_path}")
    pid = _protocol_id(req.pdf_path)
    schema = get_schema(req.pdf_path)          # parser handles its own JSON cache
    SCHEMA_REGISTRY[pid] = schema
    return {
        "protocol_id": pid,
        "study_name": schema.get("study_name"),
        "indication": schema.get("indication"),
        "field_classification": schema.get("field_classification"),
        "cached": True,
    }


@app.get("/protocol/schema")
def get_schema_endpoint(protocol_id: str):
    schema = SCHEMA_REGISTRY.get(protocol_id)
    if schema is None:
        raise HTTPException(404, f"No cached schema for '{protocol_id}'. Parse it first.")
    return schema


def _require_schema(protocol_id: str) -> dict:
    schema = SCHEMA_REGISTRY.get(protocol_id)
    if schema is None:
        raise HTTPException(
            404,
            f"No cached schema for '{protocol_id}'. Call /protocol/parse first.",
        )
    return schema


# ── Phase 2a: process documents -> data/flags/features ────────────────────────
@app.post("/process")
def process_endpoint(req: ProcessRequest):
    schema = _require_schema(req.protocol_id)
    records = process_documents(schema, req.documents)   # list[{data,flags,features}]
    return {"protocol_id": req.protocol_id, "records": records}


# ── Phase 2b: features -> review + quality ────────────────────────────────────
@app.post("/classify")
def classify_endpoint(req: FeaturesRequest):
    return assess(req.features)


# ── One-shot: documents -> full verdict per patient ───────────────────────────
@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    schema = _require_schema(req.protocol_id)
    records = process_documents(schema, req.documents)

    results = []
    for rec in records:
        verdict = assess(rec["features"])
        results.append({
            "patient_id": rec["data"].get("patient_id"),
            "data": rec["data"],
            "flags": rec["flags"],
            "features": rec["features"],
            **verdict,
        })
    return {"protocol_id": req.protocol_id, "results": results}


# ── PDF -> text helper ────────────────────────────────────────────────────────
def _pdf_bytes_to_text(raw: bytes) -> str:
    """Extract plain text from uploaded PDF bytes (one source document)."""
    reader = PdfReader(io.BytesIO(raw))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


# ── One-shot from uploaded files (PDF or TXT) ─────────────────────────────────
@app.post("/analyze/upload")
async def analyze_upload_endpoint(
    protocol_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Accept patient documents as uploaded files (PDF or TXT), extract text
    server-side, then run the same /analyze pipeline. Each file is one source
    document; documents are merged per patient inside process_documents().
    """
    schema = _require_schema(protocol_id)

    documents: list[str] = []
    skipped: list[str] = []
    for f in files:
        raw = await f.read()
        name = (f.filename or "").lower()
        if name.endswith(".pdf"):
            text = _pdf_bytes_to_text(raw)
        elif name.endswith(".txt"):
            text = raw.decode("utf-8", errors="ignore").strip()
        else:
            skipped.append(f.filename or "unnamed")
            continue
        if text:
            documents.append(text)
        else:
            skipped.append(f.filename or "unnamed (no extractable text)")

    if not documents:
        raise HTTPException(
            422,
            f"No usable text extracted. Skipped: {skipped or 'none'}. "
            "Scanned PDFs without a text layer need OCR (not supported yet).",
        )

    records = process_documents(schema, documents)
    results = []
    for rec in records:
        verdict = assess(rec["features"])
        results.append({
            "patient_id": rec["data"].get("patient_id"),
            "data": rec["data"],
            "flags": rec["flags"],
            "features": rec["features"],
            **verdict,
        })
    return {
        "protocol_id": protocol_id,
        "results": results,
        "documents_used": len(documents),
        "skipped": skipped,
    }


# ── RAG chatbox over the protocol ─────────────────────────────────────────────
@app.post("/protocol/ask")
def ask_endpoint(req: AskRequest):
    if not Path(req.pdf_path).exists():
        raise HTTPException(404, f"PDF not found: {req.pdf_path}")
    store = get_rag_store(req.pdf_path)
    source = Path(req.pdf_path).name
    hits = store.query(req.question, top_k=req.top_k, source=source)
    sources = [
        {
            "section": h["metadata"].get("section"),
            "page": h["metadata"].get("page"),
            "text": h["text"][:300],
        }
        for h in hits
    ]
    return {"question": req.question, "sources": sources}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": models_status(),
        "protocols_loaded": list(SCHEMA_REGISTRY.keys()),
    }