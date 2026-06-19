import os
import io
import csv
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ClinOrigin AI", page_icon="\u2695", layout="wide")

# ── Verdict palette ───────────────────────────────────────────────────────────
VERDICT = {
    "Clean":  {"color": "#1a7f5a", "bg": "#e8f5ef", "icon": "\u2713", "label": "CLEAN"},
    "Review": {"color": "#b8860b", "bg": "#fbf3e0", "icon": "\u26a0", "label": "REVIEW"},
    "Block":  {"color": "#b22222", "bg": "#fbe9e9", "icon": "\u2715", "label": "BLOCK"},
}

st.markdown("""
<style>
    .verdict-card { border-radius: 14px; padding: 22px 26px; margin: 8px 0 16px 0;
        display: flex; align-items: center; gap: 18px; }
    .verdict-icon { font-size: 40px; line-height: 1; }
    .verdict-label { font-size: 26px; font-weight: 700; letter-spacing: 1px; }
    .verdict-sub { font-size: 13px; opacity: 0.8; margin-top: 2px; }
    .flag-high { color: #b22222; font-weight: 600; }
    .flag-medium { color: #b8860b; }
    .flag-low { color: #888; }
    .step-pill { display:inline-block; padding:4px 10px; border-radius:20px;
        font-size:12px; margin-right:6px; background:#eef1f4; color:#555; }
    .step-done { background:#e8f5ef; color:#1a7f5a; font-weight:600; }
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


def api_post_files(path, data, files):
    try:
        r = requests.post(f"{API}{path}", data=data, files=files, timeout=180)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("schema", None), ("protocol_id", None),
                     ("pdf_path", None), ("records", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar: navigation + status ──────────────────────────────────────────────
with st.sidebar:
    st.title("\u2695 ClinOrigin AI")
    st.caption("Clinical data extraction & validation")
    st.divider()

    health, herr = api_get("/health")
    if herr:
        st.error("Backend offline")
        st.caption("Start FastAPI, then reload.")
        st.stop()
    st.success("Backend online")

    st.divider()
    page = st.radio(
        "Workflow",
        ["1. Protocol", "2. Extraction", "3. Structured",
         "4. Monitoring", "5. Chatbox", "6. Export"],
        label_visibility="collapsed",
    )

    st.divider()
    # current context
    if st.session_state.schema:
        st.caption(f"Protocol loaded:\n\n**{st.session_state.schema.get('study_name','?')}**")
    else:
        st.caption("No protocol loaded yet.")
    st.caption(f"Records extracted: {len(st.session_state.records)}")


# ── Reusable input box (text + file in one place) ─────────────────────────────
def input_box(key_prefix: str, placeholder: str):
    """
    Returns (text, uploaded_files). Mimics the demo's chat-style box:
    a text area plus an attach-files control, together.
    """
    text = st.text_area("Input", height=150, placeholder=placeholder,
                        label_visibility="collapsed", key=f"{key_prefix}_text")
    files = st.file_uploader("Attach files (.txt / .pdf)", type=["txt", "pdf"],
                            accept_multiple_files=True, key=f"{key_prefix}_files")
    return text, files


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1: PROTOCOL
# ════════════════════════════════════════════════════════════════════════════
if page == "1. Protocol":
    st.header("Protocol")
    st.caption("Parse a clinical study protocol into a structured schema. "
               "This runs once per study and is cached.")

    st.markdown(
        "<span class='step-pill step-done'>PDF</span>"
        "<span class='step-pill'>RAG index</span>"
        "<span class='step-pill'>Parse</span>"
        "<span class='step-pill'>Schema</span>",
        unsafe_allow_html=True)
    st.write("")

    pdf_path = st.text_input("Protocol PDF path (server-side)",
                             value=st.session_state.pdf_path or "",
                             placeholder="protocols/Prot_SAP_000.pdf")

    if st.button("Parse protocol", type="primary", disabled=not pdf_path):
        with st.spinner("Parsing protocol (builds RAG index, one-time)..."):
            res, err = api_post("/protocol/parse", {"pdf_path": pdf_path})
        if err:
            st.error(f"Parse failed: {err}")
        else:
            st.session_state.protocol_id = res["protocol_id"]
            st.session_state.pdf_path = pdf_path
            sch, serr = api_get("/protocol/schema",
                                params={"protocol_id": res["protocol_id"]})
            st.session_state.schema = sch if not serr else None
            st.success(f"Loaded: {res.get('study_name')}")

    # Show schema if loaded
    if st.session_state.schema:
        sc = st.session_state.schema
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Study", sc.get("study_name", "\u2014"))
        c2.metric("Indication", sc.get("indication", "\u2014"))

        fc = sc.get("field_classification", {})
        st.markdown("**Field classification (CDISC core)**")
        f1, f2, f3 = st.columns(3)
        f1.markdown("**Required**\n\n" + "\n".join(f"- {x}" for x in fc.get("required", [])))
        f2.markdown("**Expected**\n\n" + "\n".join(f"- {x}" for x in fc.get("expected", [])))
        f3.markdown("**Permissible**\n\n" + "\n".join(f"- {x}" for x in fc.get("permissible", [])))

        with st.expander("Validation ranges"):
            st.json(sc.get("validation_ranges", {}))
        with st.expander("Eligibility criteria"):
            st.json(sc.get("eligibility_criteria", {}))


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2: EXTRACTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "2. Extraction":
    st.header("Extraction")
    st.caption("Paste patient source text or attach files. The system extracts, "
               "validates, and classifies each record.")

    st.markdown(
        "<span class='step-pill step-done'>Input</span>"
        "<span class='step-pill'>Extract</span>"
        "<span class='step-pill'>Validate</span>"
        "<span class='step-pill'>Review</span>"
        "<span class='step-pill'>Result</span>",
        unsafe_allow_html=True)
    st.write("")

    if not st.session_state.protocol_id:
        st.info("Load a protocol first (page 1).")
    else:
        text, files = input_box("extract", "Paste lab report / visit note text...")
        has_input = bool(text.strip()) or bool(files)

        if st.button("Extract & Analyse", type="primary", disabled=not has_input):
            with st.spinner("Extracting, validating, classifying..."):
                if files:
                    multipart = [
                        ("files", (f.name, f.getvalue(),
                                   "application/pdf" if f.name.lower().endswith(".pdf")
                                   else "text/plain"))
                        for f in files
                    ]
                    res, err = api_post_files(
                        "/analyze/upload",
                        {"protocol_id": st.session_state.protocol_id}, multipart)
                else:
                    res, err = api_post("/analyze", {
                        "protocol_id": st.session_state.protocol_id,
                        "documents": [text.strip()],
                    })
            if err:
                st.error(f"Analysis failed: {err}")
            else:
                st.session_state.records = res.get("results", [])
                if res.get("skipped"):
                    st.warning(f"Skipped: {', '.join(res['skipped'])}")

        # Show results
        for rec in st.session_state.records:
            pid = rec.get("patient_id") or "(no patient_id)"
            v = rec["review_status"]
            style = VERDICT.get(v, VERDICT["Review"])
            detail = rec["review_detail"]

            st.markdown(f"### Patient: `{pid}`")
            sub = ""
            if detail.get("safety_rule_applied"):
                sub = "Escalated to Block by safety rule"
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

            m1, m2, m3 = st.columns(3)
            if rec.get("quality_score") is not None:
                m1.metric("Quality score", f"{rec['quality_score']:.2f}")
            m2.metric("Total flags", rec["features"].get("total_flags"))
            m3.metric("Completeness", f"{rec['features'].get('completeness_score', 0):.0%}")

            cf, cp = st.columns(2)
            with cf:
                st.markdown("**Validation flags**")
                if not rec.get("flags"):
                    st.caption("No flags raised.")
                for fl in rec.get("flags", []):
                    sev = fl.get("severity", "low")
                    cls = {"high": "flag-high", "medium": "flag-medium"}.get(sev, "flag-low")
                    st.markdown(f"<span class='{cls}'>\u25cf {fl['message']}</span>",
                               unsafe_allow_html=True)
            with cp:
                st.markdown("**Class probabilities**")
                for cls_name, p in detail["probabilities"].items():
                    st.progress(p, text=f"{cls_name}: {p:.1%}")

            with st.expander("Extracted data"):
                st.json(rec["data"])
            st.divider()


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3: STRUCTURED DATA
# ════════════════════════════════════════════════════════════════════════════
elif page == "3. Structured":
    st.header("Structured data")
    st.caption("All extracted records in tabular form.")

    if not st.session_state.records:
        st.info("No records yet. Run an extraction (page 2).")
    else:
        rows = []
        for rec in st.session_state.records:
            row = dict(rec["data"])
            row["_review"] = rec["review_status"]
            row["_quality"] = rec.get("quality_score")
            row["_flags"] = rec["features"].get("total_flags")
            rows.append(row)
        st.dataframe(rows, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4: MONITORING
# ════════════════════════════════════════════════════════════════════════════
elif page == "4. Monitoring":
    st.header("Monitoring review")
    st.caption("Review-status overview across all extracted records.")

    if not st.session_state.records:
        st.info("No records yet. Run an extraction (page 2).")
    else:
        counts = {"Clean": 0, "Review": 0, "Block": 0}
        for rec in st.session_state.records:
            counts[rec["review_status"]] = counts.get(rec["review_status"], 0) + 1

        c1, c2, c3 = st.columns(3)
        c1.metric("\u2713 Clean", counts["Clean"])
        c2.metric("\u26a0 Review", counts["Review"])
        c3.metric("\u2715 Block", counts["Block"])

        st.divider()
        for rec in st.session_state.records:
            v = rec["review_status"]
            style = VERDICT.get(v, VERDICT["Review"])
            pid = rec.get("patient_id") or "(no id)"
            st.markdown(
                f"<span style='color:{style['color']};font-weight:600'>"
                f"{style['icon']} {style['label']}</span> &nbsp; "
                f"`{pid}` &nbsp; \u2014 quality "
                f"{rec.get('quality_score', 0):.2f}, "
                f"{rec['features'].get('total_flags')} flags",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5: CHATBOX
# ════════════════════════════════════════════════════════════════════════════
elif page == "5. Chatbox":
    st.header("Ask the protocol")
    st.caption("Questions are answered from the protocol text only, with the "
               "source section and page shown.")

    if not st.session_state.pdf_path:
        st.info("Load a protocol first (page 1).")
    else:
        st.caption(f"Querying: `{st.session_state.pdf_path}`")
        question = st.text_input("Your question",
                                 placeholder="Which patients are excluded?")
        if st.button("Ask", type="primary", disabled=not question):
            with st.spinner("Retrieving and answering..."):
                res, err = api_post("/protocol/ask", {
                    "pdf_path": st.session_state.pdf_path,
                    "question": question,
                    "top_k": 4,
                })
            if err:
                st.error(f"Query failed: {err}")
            else:
                st.markdown("**Answer**")
                st.write(res.get("answer", ""))
                with st.expander("Sources"):
                    for i, s in enumerate(res.get("sources", []), 1):
                        st.markdown(f"**Source {i}** \u2014 *{s['section']}, page {s['page']}*")
                        st.caption(s["text"])


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 6: EXPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "6. Export":
    st.header("Export")
    st.caption("Download structured, review-ready records as CSV.")

    if not st.session_state.records:
        st.info("No records yet. Run an extraction (page 2).")
    else:
        rows = []
        for rec in st.session_state.records:
            row = dict(rec["data"])
            row["review_status"] = rec["review_status"]
            row["quality_score"] = rec.get("quality_score")
            row["total_flags"] = rec["features"].get("total_flags")
            rows.append(row)

        # Build CSV
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

        st.dataframe(rows, use_container_width=True)
        st.download_button("Download CSV", buf.getvalue(),
                          file_name="clinorigin_records.csv", mime="text/csv",
                          type="primary")