"""
Document Q&A — Clinical Protocol Intelligence
Uses the backend/ modules directly (ChromaDB + section-aware chunking + Claude).
Pre-trained review classifier (Clean / Review / Block) runs on demand.
"""
import sys, tempfile, os
from pathlib import Path

import streamlit as st

# Make sure root is on the path so `backend` is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ANTHROPIC_API_KEY

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Clinical Protocol Intelligence")
st.caption("Upload a clinical protocol PDF → ask questions → get section-aware answers via Claude")

if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY is not set. Add it to your `.env` file and restart.")
    st.stop()

# ── Lazy imports (heavy) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_backend():
    from backend.rag_chunker import chunk_protocol
    from backend.rag_store import ProtocolVectorStore, LocalEmbedder
    from backend.inference import assess
    import anthropic
    return chunk_protocol, ProtocolVectorStore, LocalEmbedder, assess, anthropic


# ── Pre-loaded demo protocol ──────────────────────────────────────────────────
DEMO_PDF = ROOT / "backend" / "protocols" / "Prot_SAP_000.pdf"
DEMO_CHROMA = ROOT / "backend" / "protocols" / "chroma_db"

# ── Upload section ────────────────────────────────────────────────────────────
col_upload, col_demo = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader("Upload a clinical study protocol (PDF)", type="pdf")

with col_demo:
    st.markdown("**Or use the demo protocol**")
    use_demo = st.button("Load Prot_SAP_000 (pre-indexed)", disabled=not DEMO_PDF.exists())

# ── Index management ──────────────────────────────────────────────────────────
def _index_pdf(pdf_path: Path, chroma_dir: Path):
    chunk_protocol, ProtocolVectorStore, LocalEmbedder, assess, _ = _load_backend()
    store = ProtocolVectorStore(
        persist_dir=str(chroma_dir),
        embedder=LocalEmbedder(),
    )
    source = pdf_path.name
    if not store.is_indexed(source):
        chunks = chunk_protocol(pdf_path)
        store.build_index([c.to_dict() for c in chunks])
    return store, source


if use_demo and DEMO_PDF.exists():
    if st.session_state.get("active_pdf") != str(DEMO_PDF):
        with st.spinner("Loading pre-indexed demo protocol…"):
            store, source = _index_pdf(DEMO_PDF, DEMO_CHROMA)
        st.session_state["rag_store"]   = store
        st.session_state["rag_source"]  = source
        st.session_state["active_pdf"]  = str(DEMO_PDF)
        st.session_state["chat_history"] = []
        st.success(f"Demo protocol loaded: **{source}** ({store.count()} chunks in index)")

if uploaded and st.session_state.get("active_pdf") != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=ROOT / "backend" / "protocols") as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    chroma_dir = ROOT / "backend" / "protocols" / "chroma_db"
    with st.spinner(f"Indexing **{uploaded.name}** (first time only)…"):
        store, source = _index_pdf(tmp_path, chroma_dir)

    st.session_state["rag_store"]    = store
    st.session_state["rag_source"]   = source
    st.session_state["active_pdf"]   = uploaded.name
    st.session_state["chat_history"] = []
    st.success(f"Indexed **{store.count()} chunks** from **{uploaded.name}**")

# Show which document is active
if "rag_source" in st.session_state:
    st.info(f"Active protocol: **{st.session_state['rag_source']}**  "
            f"· {st.session_state['rag_store'].count()} chunks indexed")
else:
    st.warning("No protocol loaded. Upload a PDF or click **Load Prot_SAP_000** above.")
    st.stop()

st.divider()

# ── Chat ──────────────────────────────────────────────────────────────────────
tab_qa, tab_schema, tab_classify = st.tabs(
    ["💬 Ask the protocol", "📋 Protocol schema", "🔍 Record classification"]
)

with tab_qa:
    st.subheader("Ask a question about the protocol")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("e.g. What are the exclusion criteria?  /  What dosing was used?")

    if question:
        store  = st.session_state["rag_store"]
        source = st.session_state["rag_source"]
        st.chat_message("user").write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Retrieving context and asking Claude…"):
            _, _, _, _, anthropic = _load_backend()
            hits = store.query(question, top_k=5, source=source)
            context = "\n\n---\n\n".join(
                f"[Section: {h['metadata'].get('section')}, Page: {h['metadata'].get('page')}]\n{h['text']}"
                for h in hits
            )
            client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content":
                    f"You are a clinical research assistant.\n\n"
                    f"Context from study protocol (section-tagged):\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely. Cite section and page where possible."
                }]
            )
            answer = response.content[0].text

        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.expander("Example questions"):
        st.markdown("""
- What are the inclusion / exclusion criteria?
- What dosing regimen was used?
- What were the primary endpoints?
- What adverse events were reported?
- Summarise the study population.
        """)

with tab_schema:
    st.subheader("Parsed protocol schema")
    schema_file = ROOT / "backend" / "schema" / (
        Path(st.session_state.get("active_pdf", "")).stem + "_schema.json"
    )
    if schema_file.exists():
        import json
        with open(schema_file) as f:
            schema = json.load(f)

        c1, c2 = st.columns(2)
        c1.metric("Study", schema.get("study_name") or "—")
        c2.metric("Indication", schema.get("indication") or "—")

        if schema.get("eligibility_criteria"):
            with st.expander("Eligibility criteria", expanded=True):
                ec = schema["eligibility_criteria"]
                cols = st.columns(2)
                cols[0].markdown("**Inclusion**")
                for item in ec.get("inclusion", []):
                    cols[0].markdown(f"- {item}")
                cols[1].markdown("**Exclusion**")
                for item in ec.get("exclusion", []):
                    cols[1].markdown(f"- {item}")

        if schema.get("field_classification"):
            with st.expander("Field classification"):
                fc = schema["field_classification"]
                for cat, fields in fc.items():
                    st.markdown(f"**{cat.capitalize()}**: {', '.join(fields) if fields else '—'}")

        if schema.get("validation_ranges"):
            with st.expander("Validation ranges"):
                import pandas as pd
                rows = []
                for field, units in schema["validation_ranges"].items():
                    if isinstance(units, dict):
                        for unit, bounds in units.items():
                            if isinstance(bounds, dict):
                                rows.append({"Field": field, "Unit": unit,
                                             "Min": bounds.get("min"), "Max": bounds.get("max")})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Schema not yet parsed for this protocol. "
                "The pre-built schema is available for **Prot_SAP_000** only. "
                "Parsing a new protocol requires your ANTHROPIC_API_KEY and takes ~30 s.")
        if st.button("Parse protocol schema now"):
            from backend.protocol_parser import get_schema
            with st.spinner("Parsing protocol with Claude (this takes ~30 s)…"):
                get_schema(str(ROOT / "backend" / "protocols" / st.session_state["active_pdf"]))
            st.rerun()

with tab_classify:
    st.subheader("Patient record review classification")
    st.caption("Enter the feature scores for a patient record — the pre-trained model returns "
               "Clean / Review / Block with probabilities.")

    _, _, _, assess, _ = _load_backend()

    with st.form("classify_form"):
        c1, c2, c3 = st.columns(3)
        completeness   = c1.slider("Completeness score",      0.0, 1.0, 0.85, 0.01)
        missing_req    = c1.number_input("Missing required fields",  0, 20, 0)
        missing_opt    = c1.number_input("Missing optional fields",  0, 50, 3)
        out_of_range   = c2.number_input("Out-of-range values",      0, 30, 0)
        plausibility   = c2.number_input("Plausibility issues",      0, 20, 0)
        total_flags    = c2.number_input("Total flags",              0, 50, 2)
        critical_miss  = c3.number_input("Critical fields missing",  0, 10, 0)
        high_sev       = c3.number_input("High severity flags",      0, 20, 0)
        confidence     = c3.slider("Extraction confidence",   0.0, 1.0, 0.90, 0.01)
        field_count    = c3.number_input("Total fields present",     1, 200, 45)
        submitted = st.form_submit_button("Classify record", type="primary")

    if submitted:
        features = {
            "completeness_score": completeness,
            "missing_required_count": missing_req,
            "missing_optional_count": missing_opt,
            "out_of_range_count": out_of_range,
            "plausibility_issues": plausibility,
            "total_flags": total_flags,
            "critical_fields_missing": critical_miss,
            "high_severity_flags": high_sev,
            "extraction_confidence": confidence,
            "field_count_total": field_count,
        }
        result = assess(features)
        label  = result["review_status"]
        color  = {"Clean": "#2e7d32", "Review": "#f9a825", "Block": "#c62828"}.get(label, "#555")

        st.markdown(f"<h3 style='color:{color}'>Decision: {label}</h3>", unsafe_allow_html=True)

        detail  = result["review_detail"]
        quality = result["quality_score"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Quality score", f"{quality:.2%}" if quality else "N/A")
        m2.metric("Block probability", f"{detail.get('block_probability', 0):.2%}")
        m3.metric("Safety rule applied", "Yes" if detail.get("safety_rule_applied") else "No")

        st.json({"probabilities": detail["probabilities"],
                 "decision_drivers": result["decision_drivers"]})
