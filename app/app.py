import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API = st.secrets.get("API_URL", "http://localhost:8000") if hasattr(st, "secrets") else "http://localhost:8000"

st.set_page_config(
    page_title="ClinOrigin AI",
    page_icon="\u2695",  # medical symbol
    layout="wide",
)

# ── Verdict styling (the signature element) ───────────────────────────────────
VERDICT_STYLE = {
    "Clean":  {"color": "#1a7f5a", "bg": "#e8f5ef", "icon": "\u2713", "label": "CLEAN"},
    "Review": {"color": "#b8860b", "bg": "#fbf3e0", "icon": "\u26a0", "label": "REVIEW"},
    "Block":  {"color": "#b22222", "bg": "#fbe9e9", "icon": "\u2715", "label": "BLOCK"},
}

st.markdown("""
<style>
    .verdict-card {
        border-radius: 14px; padding: 24px 28px; margin: 8px 0 16px 0;
        display: flex; align-items: center; gap: 20px;
    }
    .verdict-icon { font-size: 44px; line-height: 1; }
    .verdict-label { font-size: 30px; font-weight: 700; letter-spacing: 1px; }
    .verdict-sub { font-size: 14px; opacity: 0.8; margin-top: 2px; }
    .metric-small { font-size: 13px; color: #555; }
    .flag-high { color: #b22222; font-weight: 600; }
    .flag-medium { color: #b8860b; }
    .flag-low { color: #888; }
</style>
""", unsafe_allow_html=True)


# ── Backend helpers ───────────────────────────────────────────────────────────
def api_get(path, **kw):
    try:
        r = requests.get(f"{API}{path}", timeout=30, **kw)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def api_post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


# ── Sidebar: context + control ────────────────────────────────────────────────
with st.sidebar:
    st.title("\u2695 ClinOrigin AI")
    st.caption("Clinical data extraction & validation")

    st.divider()

    # Backend status
    health, err = api_get("/health")
    if err:
        st.error("Backend offline")
        st.caption(f"Start FastAPI, then reload.\n\n`{err[:80]}`")
        st.stop()
    else:
        st.success("Backend online")
        models = health.get("models", {})
        st.caption(
            f"Classifier: {'ok' if models.get('classifier') else 'missing'}  \n"
            f"Quality: {'ok' if models.get('quality_regressor') else 'missing'}  \n"
            f"Safety threshold: {models.get('safety_threshold')}"
        )

    st.divider()

    # Protocol selection
    st.subheader("Protocol")
    loaded = health.get("protocols_loaded", [])
    if loaded:
        protocol_id = st.selectbox("Loaded protocol", loaded)
    else:
        protocol_id = None
        st.info("No protocol loaded yet.")

    with st.expander("Load a new protocol"):
        pdf_path = st.text_input("Server-side PDF path", placeholder="protocols/HeartMagic.pdf")
        if st.button("Parse protocol", use_container_width=True):
            if pdf_path:
                with st.spinner("Parsing protocol (one-time, builds RAG index)..."):
                    res, perr = api_post("/protocol/parse", {"pdf_path": pdf_path})
                if perr:
                    st.error(f"Parse failed: {perr}")
                else:
                    st.success(f"Loaded: {res['study_name']}")
                    st.rerun()
            else:
                st.warning("Enter a PDF path.")


# ── Main area: tabs ───────────────────────────────────────────────────────────
tab_analyse, tab_chat, tab_schema = st.tabs(["Analyse", "Chatbox", "Schema"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1: ANALYSE
# ════════════════════════════════════════════════════════════════════════════
with tab_analyse:
    if not protocol_id:
        st.info("Load a protocol in the sidebar to start analysing patient documents.")
    else:
        st.subheader("Patient documents")
        st.caption("Paste source text or upload files. Each document is one source "
                   "(lab report, visit note); they are merged per patient.")

        col_in, col_btn = st.columns([4, 1])
        with col_in:
            input_mode = st.radio("Input", ["Paste text", "Upload files"],
                                  horizontal=True, label_visibility="collapsed")

        documents = []
        uploaded_files = []
        if input_mode == "Paste text":
            txt = st.text_area("Source document text", height=180,
                               placeholder="Paste lab report / visit note text here...")
            if txt.strip():
                documents = [txt.strip()]
        else:
            uploaded_files = st.file_uploader(
                "Upload .txt or .pdf", type=["txt", "pdf"],
                accept_multiple_files=True) or []
            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) ready. "
                           "PDF text is extracted on the server.")

        has_input = bool(documents) or bool(uploaded_files)
        analyse = st.button("Analyse", type="primary", use_container_width=True,
                            disabled=not has_input)

        if analyse and has_input:
            with st.spinner("Extracting, validating, classifying..."):
                if uploaded_files:
                    # Send files to the upload endpoint (server extracts PDF text)
                    multipart = [
                        ("files", (f.name, f.getvalue(),
                                   "application/pdf" if f.name.lower().endswith(".pdf")
                                   else "text/plain"))
                        for f in uploaded_files
                    ]
                    try:
                        resp = requests.post(
                            f"{API}/analyze/upload",
                            data={"protocol_id": protocol_id},
                            files=multipart, timeout=180)
                        resp.raise_for_status()
                        res, aerr = resp.json(), None
                    except Exception as e:
                        res, aerr = None, str(e)
                else:
                    res, aerr = api_post("/analyze", {
                        "protocol_id": protocol_id,
                        "documents": documents,
                    })
            if aerr:
                st.error(f"Analysis failed: {aerr}")
            else:
                results = res.get("results", [])
                st.caption(f"{len(results)} patient record(s) processed.")
                if res.get("skipped"):
                    st.warning(f"Skipped (no extractable text): {', '.join(res['skipped'])}")

                for rec in results:
                    pid = rec.get("patient_id") or "(no patient_id)"
                    verdict = rec["review_status"]
                    style = VERDICT_STYLE.get(verdict, VERDICT_STYLE["Review"])
                    detail = rec["review_detail"]
                    quality = rec.get("quality_score")

                    st.markdown(f"### Patient: `{pid}`")

                    # --- Signature: verdict card ---
                    sub = ""
                    if detail.get("safety_rule_applied"):
                        sub = "Escalated to Block by safety rule (P(Block) above threshold)"
                    elif detail.get("block_probability") is not None:
                        sub = f"P(Block) = {detail['block_probability']:.1%}"
                    st.markdown(f"""
                    <div class="verdict-card" style="background:{style['bg']};">
                        <div class="verdict-icon" style="color:{style['color']};">{style['icon']}</div>
                        <div>
                            <div class="verdict-label" style="color:{style['color']};">{style['label']}</div>
                            <div class="verdict-sub">{sub}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- Metrics row ---
                    m1, m2, m3 = st.columns(3)
                    if quality is not None:
                        m1.metric("Quality score", f"{quality:.2f}")
                    m2.metric("Total flags", rec["features"].get("total_flags"))
                    m3.metric("Completeness", f"{rec['features'].get('completeness_score', 0):.0%}")

                    # --- Flags + class probabilities ---
                    c_flags, c_prob = st.columns(2)
                    with c_flags:
                        st.markdown("**Validation flags**")
                        flags = rec.get("flags", [])
                        if not flags:
                            st.caption("No flags raised.")
                        for fl in flags:
                            sev = fl.get("severity", "low")
                            cls = {"high": "flag-high", "medium": "flag-medium"}.get(sev, "flag-low")
                            st.markdown(
                                f"<span class='{cls}'>\u25cf {fl['message']}</span>",
                                unsafe_allow_html=True)
                    with c_prob:
                        st.markdown("**Class probabilities**")
                        for cls_name, p in detail["probabilities"].items():
                            st.progress(p, text=f"{cls_name}: {p:.1%}")

                    with st.expander("Extracted data + decision drivers"):
                        st.json({"data": rec["data"],
                                 "decision_drivers": rec.get("decision_drivers")})
                    st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2: CHATBOX (RAG over the protocol)
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Ask the protocol")
    st.caption("Questions are answered from the protocol text only, with the source "
               "section and page shown for every answer.")

    if not protocol_id:
        st.info("Load a protocol first.")
    else:
        pdf_for_chat = st.text_input(
            "Protocol PDF path (for retrieval)",
            placeholder="protocols/HeartMagic.pdf",
            help="The same PDF you parsed; RAG reuses its index, no re-embedding.",
        )
        question = st.text_input("Your question",
                                 placeholder="Which patients are excluded?")
        if st.button("Ask", type="primary", disabled=not (question and pdf_for_chat)):
            with st.spinner("Retrieving relevant protocol sections..."):
                res, cerr = api_post("/protocol/ask", {
                    "pdf_path": pdf_for_chat,
                    "question": question,
                    "top_k": 4,
                })
            if cerr:
                st.error(f"Query failed: {cerr}")
            else:
                st.markdown("**Relevant protocol passages**")
                for i, s in enumerate(res.get("sources", []), 1):
                    st.markdown(f"**Source {i}** \u2014 *{s['section']}, page {s['page']}*")
                    st.caption(s["text"])
                    st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3: SCHEMA
# ════════════════════════════════════════════════════════════════════════════
with tab_schema:
    st.subheader("Study schema")
    if not protocol_id:
        st.info("Load a protocol first.")
    else:
        schema, serr = api_get("/protocol/schema", params={"protocol_id": protocol_id})
        if serr:
            st.error(f"Could not load schema: {serr}")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Study", schema.get("study_name", "\u2014"))
            c2.metric("Indication", schema.get("indication", "\u2014"))

            fc = schema.get("field_classification", {})
            st.markdown("**Field classification (CDISC core)**")
            f1, f2, f3 = st.columns(3)
            f1.markdown(f"**Required**\n\n" + "\n".join(f"- {x}" for x in fc.get("required", [])))
            f2.markdown(f"**Expected**\n\n" + "\n".join(f"- {x}" for x in fc.get("expected", [])))
            f3.markdown(f"**Permissible**\n\n" + "\n".join(f"- {x}" for x in fc.get("permissible", [])))

            with st.expander("Validation ranges"):
                st.json(schema.get("validation_ranges", {}))
            with st.expander("Eligibility criteria"):
                st.json(schema.get("eligibility_criteria", {}))
            with st.expander("Full schema JSON"):
                st.json(schema)