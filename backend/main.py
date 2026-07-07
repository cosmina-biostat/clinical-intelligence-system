from pathlib import Path
from typing import Optional
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from anthropic import Anthropic

# ── Heart pipeline (uploaded modules) ─────────────────────────────────────────
from backend.protocol_parser import get_schema, get_rag_store
from backend.extractor import process_documents

# ── ML inference (this chat's models) ─────────────────────────────────────────
from backend.inference import assess, classify_review, predict_quality, models_status
from backend.disease_prediction.disease_models import build_registry, resolve_features
from backend.disease_prediction.anomaly_models import build_anomaly_registry

app = FastAPI(title="ClinOrigin AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Server-side schema registry (protocol_id -> schema dict) ──────────────────
SCHEMA_REGISTRY: dict[str, dict] = {}
DISEASE_REGISTRY = build_registry()
ANOMALY_REGISTRY = build_anomaly_registry()
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


@app.post("/protocol/parse/upload")
async def parse_protocol_upload(file: UploadFile = File(...)):
    """
    Parse an UPLOADED protocol PDF (drag-and-drop from the dashboard).
    The file is saved server-side into protocols/, then parsed like the
    path-based endpoint. Returns the same shape as /protocol/parse.
    """
    name = (file.filename or "protocol.pdf")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(422, "Only PDF protocols are supported.")

    # Save into the server-side protocols/ folder
    dest = PROTOCOL_DIR / name
    raw = await file.read()
    dest.write_bytes(raw)

    pid = _protocol_id(str(dest))
    schema = get_schema(str(dest))      # builds RAG index + schema cache
    SCHEMA_REGISTRY[pid] = schema
    return {
        "protocol_id": pid,
        "study_name": schema.get("study_name"),
        "indication": schema.get("indication"),
        "field_classification": schema.get("field_classification"),
        "saved_path": str(dest),
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


# ── RAG chatbox over the protocol (grounded answer) ───────────────────────────
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

    if not hits:
        return {
            "question": req.question,
            "answer": "Not found in the provided protocol sections.",
            "sources": [],
        }

    # Build a grounded prompt: answer ONLY from retrieved passages
    context_blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        context_blocks.append(
            f"[Source {i} | section: {m.get('section')}, page: {m.get('page')}]\n{h['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a clinical protocol assistant. Answer the question using ONLY the context below.

Rules:
- Use only information present in the context. Do not add outside knowledge.
- If the context does not contain the answer, reply exactly: "Not found in the provided protocol sections."
- Cite the source number(s) you used, e.g. [Source 1].
- Be concise and precise. Quote exact thresholds, doses, and criteria verbatim.

Context:
{context}

Question: {req.question}

Answer:"""

    try:
        client = Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:
        # Fall back to passages if the LLM call fails
        answer = f"(Could not generate answer: {e})"

    return {"question": req.question, "answer": answer, "sources": sources}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/protocols/list")
def list_protocols():
    """
    List protocol PDFs available on the server (in protocols/). For each,
    try to read the cached schema (backend/schema/<stem>_schema.json) to show
    the indication; if not parsed yet, only the filename is known.

    Returns label like "Multiple Sclerosis — SAP_003".
    """
    schema_dir = Path(__file__).parent / "schema"
    items = []
    for pdf in sorted(PROTOCOL_DIR.glob("*.pdf")):
        pid = _protocol_id(str(pdf))
        indication = None
        study_name = None
        acronym = None

        # Prefer the in-memory registry, then the on-disk schema cache
        if pid in SCHEMA_REGISTRY:
            sch = SCHEMA_REGISTRY[pid]
            indication = sch.get("indication")
            study_name = sch.get("study_name")
            acronym = sch.get("study_acronym")
        else:
            cache = schema_dir / f"{pid}_schema.json"
            if cache.exists():
                try:
                    import json as _json
                    sch = _json.loads(cache.read_text(encoding="utf-8"))
                    indication = sch.get("indication")
                    study_name = sch.get("study_name")
                    acronym = sch.get("study_acronym")
                except Exception:
                    pass

        # Build label: "Acronym — Indication" (fall back gracefully)
        ind = (indication or "").strip()
        acr = (acronym or "").strip()

        if acr and ind:
            label = f"{acr} \u2014 {ind}"
        elif acr:
            label = acr
        elif ind:
            label = f"{ind} \u2014 {pdf.stem}"
        else:
            label = pdf.stem      # not parsed yet: only the filename is known

        items.append({
            "protocol_id": pid,
            "filename": pdf.name,
            "path": str(pdf),
            "parsed": (pid in SCHEMA_REGISTRY) or (schema_dir / f"{pid}_schema.json").exists(),
            "indication": indication,
            "study_name": study_name,
            "study_acronym": acronym,
            "label": label,
        })
    return {"protocols": items, "count": len(items)}


# ── Disease Prediction (clinical risk models, routed by indication) ──────────
class DiseasePredictRequest(BaseModel):
    model_key: str          # "cardio" | "ms" | ... from /disease/match or /disease/models
    features: dict          # raw feature values matching that model's feature_order


class PredictFromRecordRequest(BaseModel):
    indication: str          # protocol's indication, used to pick the model
    data: dict               # the extracted patient record (rec["data"])


@app.post("/disease/predict_from_record")
def disease_predict_from_record(req: PredictFromRecordRequest):
    """
    One-call convenience for the Structured Data table: given a protocol's
    indication and an already-extracted patient record, match the model,
    map the record's fields onto that model's feature contract, and predict.

    Never guesses: if required fields can't be resolved from the record,
    returns available=False with the list of missing/unmappable fields
    instead of a fabricated prediction.
    """
    card = DISEASE_REGISTRY.match_indication(req.indication)
    if card is None:
        return {"available": False, "reason": "no_model_for_indication", "missing": []}

    features, missing = resolve_features(card.key, req.data)
    if features is None:
        return {"available": False, "reason": "missing_fields", "missing": missing,
                "model_key": card.key, "model_name": card.display_name}

    try:
        result = DISEASE_REGISTRY.predict(card.key, features)
    except NotImplementedError as e:
        return {"available": False, "reason": "model_not_implemented", "missing": [],
                "model_key": card.key, "model_name": card.display_name, "detail": str(e)}

    return {
        "available": True,
        "model_key": result.model_key,
        "model_name": result.model_name,
        "label": result.label,
        "probability": result.probability,
    }


class AnomalyFromRecordRequest(BaseModel):
    indication: str
    data: dict


@app.post("/anomaly/detect_from_record")
def anomaly_detect_from_record(req: AnomalyFromRecordRequest):
    """
    Unsupervised anomaly check for the Structured Data table: given a
    protocol's indication and an already-extracted patient record, match
    the anomaly model, resolve the record's fields onto its feature
    contract (same mapping used for risk prediction, since both cardio
    models share the same 12 features), and run IsolationForest.

    Complements the rule-based checks.py flags and the supervised risk
    model: a patient can pass every individual range check and still be
    flagged here if the *combination* of values is statistically unusual.
    Never guesses -- missing fields return available=False.
    """
    card = ANOMALY_REGISTRY.match_indication(req.indication)
    if card is None:
        return {"available": False, "reason": "no_model_for_indication"}

    # Reuse the disease-risk resolver: same model key ("cardio") happens to
    # share the identical 12-feature contract for both registries.
    features, missing = resolve_features(card.key, req.data)
    if features is None:
        return {"available": False, "reason": "missing_fields", "missing": missing,
                "model_key": card.key, "model_name": card.display_name}

    result = ANOMALY_REGISTRY.detect(card.key, features)
    return {
        "available": True,
        "model_key": result.model_key,
        "model_name": result.model_name,
        "is_anomaly": result.is_anomaly,
        "anomaly_score": result.anomaly_score,
        "top_contributors": result.top_contributors,
    }


@app.get("/disease/match")
def disease_match(indication: str):
    """
    Auto-detect which clinical risk model fits a protocol's indication.
    Returns matched=False if no confident match -- the UI must then let
    the user pick manually rather than guessing.
    """
    card = DISEASE_REGISTRY.match_indication(indication)
    if card is None:
        return {"matched": False, "model_key": None,
                "candidates": [c.key for c in DISEASE_REGISTRY.available()]}
    return {
        "matched": True,
        "model_key": card.key,
        "display_name": card.display_name,
        "feature_order": card.feature_order,
        "notes": card.notes,
    }


@app.get("/disease/models")
def disease_models_list():
    """List every registered disease model (for a manual picker)."""
    return {
        "models": [
            {"key": c.key, "display_name": c.display_name,
             "feature_order": c.feature_order, "notes": c.notes}
            for c in DISEASE_REGISTRY.available()
        ]
    }


@app.post("/disease/predict")
def disease_predict(req: DiseasePredictRequest):
    """
    Run the selected model. model_key must come from a prior /disease/match
    or /disease/models call, confirmed by the user in the UI -- this
    endpoint does not auto-select a model itself.
    """
    try:
        result = DISEASE_REGISTRY.predict(req.model_key, req.features)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except NotImplementedError as e:
        raise HTTPException(501, str(e))

    return {
        "model_key": result.model_key,
        "model_name": result.model_name,
        "label": result.label,
        "probability": result.probability,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "disease_models": [c.key for c in DISEASE_REGISTRY.available()],
        "anomaly_models": [c.key for c in ANOMALY_REGISTRY.available()],
        "models": models_status(),
        "protocols_loaded": list(SCHEMA_REGISTRY.keys()),
    }