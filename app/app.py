"""
app.py
ClinOrigin AI -- Streamlit Dashboard (dark theme, real backend)

Design adapted from the early Perplexity demo (dark surface, cyan accent,
Inter + JetBrains Mono), wired to the REAL FastAPI backend:
  - /protocol/parse, /protocol/parse/upload   parse a protocol -> schema
  - /protocol/ask                             grounded RAG answer
  - /analyze, /analyze/upload                 extract + validate + classify
  - /protocol/schema, /health

Pages:
  1. Protocol & Chat   parse a protocol (path OR drag-and-drop PDF),
                       then ask the protocol -- answer shown below the box
  2. Extraction        patient documents -> Clean/Review/Block verdict
  3. Structured        all records as a table
  4. Monitoring        review-status overview
  5. Insights          charts over real records
  6. Export            CSV download

Run (from repo root):  streamlit run app/app.py
Backend URL via API_URL env (Docker -> http://backend:8000).
"""

import os
import io
import csv
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ClinOrigin AI", page_icon="\u2695",
                   layout="wide", initial_sidebar_state="expanded")

# ─── Theme (dark, cyan accent) ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#0d0f12; --surface:#131618; --surface2:#1a1d21;
    --border:rgba(255,255,255,0.07); --text:#e8eaed; --muted:#8b9299;
    --cyan:#22d3ee; --green:#4ade80; --amber:#fbbf24; --red:#f87171;
}
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif !important; }
.stApp { background-color:#0d0f12 !important; }
section[data-testid="stSidebar"] { background-color:#131618 !important;
    border-right:1px solid rgba(255,255,255,0.07) !important; }
section[data-testid="stSidebar"] > div { background-color:#131618 !important; }
.main .block-container { background-color:#0d0f12; padding-top:1.5rem; max-width:1100px; }
h1 { font-size:22px !important; font-weight:700 !important; letter-spacing:-0.02em !important; color:#e8eaed !important; }
h2 { font-size:16px !important; font-weight:600 !important; color:#e8eaed !important; }
h3 { font-size:14px !important; font-weight:600 !important; color:#e8eaed !important; }
p, label, div { color:#e8eaed; }
[data-testid="stMetricValue"] { color:#22d3ee !important; font-size:28px !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { color:#8b9299 !important; font-size:12px !important; font-weight:600 !important; text-transform:uppercase; letter-spacing:0.05em; }
.stButton > button { background-color:#22d3ee !important; color:#000 !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; font-size:13px !important; padding:8px 18px !important; transition:all .15s !important; }
.stButton > button:hover { background-color:#38e4f5 !important; transform:translateY(-1px); }
.stButton > button[disabled] { opacity:.5 !important; cursor:not-allowed !important; }
.stTextArea textarea, .stTextInput input { background-color:#0d0f12 !important; color:#e8eaed !important;
    border:1px solid rgba(255,255,255,0.12) !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:12px !important; line-height:1.7 !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:#22d3ee !important;
    box-shadow:0 0 0 1px rgba(34,211,238,0.3) !important; }
.stSelectbox > div > div { background-color:#131618 !important; border:1px solid rgba(255,255,255,0.12) !important;
    color:#e8eaed !important; border-radius:8px !important; }
[data-testid="stFileUploaderDropzone"] { background-color:#131618 !important;
    border:1.5px dashed rgba(34,211,238,0.4) !important; border-radius:10px !important; }
.stDataFrame { border:1px solid rgba(255,255,255,0.07) !important; border-radius:10px !important; overflow:hidden !important; }
.stProgress > div > div { background-color:#22d3ee !important; }
hr { border-color:rgba(255,255,255,0.07) !important; }
/* Custom cards */
.card { background:#131618; border:1px solid rgba(255,255,255,0.07); border-radius:12px;
    padding:18px 20px; margin:10px 0; }
.verdict { border-radius:12px; padding:18px 22px; margin:10px 0; display:flex; align-items:center; gap:16px; }
.verdict-icon { font-size:34px; line-height:1; }
.verdict-label { font-size:22px; font-weight:700; letter-spacing:1px; }
.verdict-sub { font-size:12px; color:#8b9299; margin-top:2px; }
.answer-box { background:#131618; border:1px solid rgba(34,211,238,0.25); border-left:3px solid #22d3ee;
    border-radius:10px; padding:16px 20px; margin:12px 0; font-size:14px; line-height:1.6; }
.src-chip { display:inline-block; background:#1a1d21; border:1px solid rgba(255,255,255,0.1);
    border-radius:6px; padding:3px 9px; margin:3px 4px 0 0; font-size:11px; color:#8b9299;
    font-family:'JetBrains Mono',monospace; }
.flag-high { color:#f87171; font-weight:600; }
.flag-medium { color:#fbbf24; }
.flag-low { color:#8b9299; }
.section-label { color:#8b9299; font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; margin:8px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

VERDICT = {
    "Clean":  {"c": "#4ade80", "bg": "rgba(74,222,128,0.10)", "i": "\u2713", "l": "CLEAN"},
    "Review": {"c": "#fbbf24", "bg": "rgba(251,191,36,0.10)", "i": "\u26a0", "l": "REVIEW"},
    "Block":  {"c": "#f87171", "bg": "rgba(248,113,113,0.10)", "i": "\u2715", "l": "BLOCK"},
}

# ─── Backend helpers ──────────────────────────────────────────────────────────
def api_get(path, **kw):
    try:
        r = requests.get(f"{API}{path}", timeout=30, **kw); r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=120); r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post_files(path, data, files):
    try:
        r = requests.post(f"{API}{path}", data=data, files=files, timeout=180); r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [("schema", None), ("protocol_id", None), ("pdf_path", None),
             ("records", []), ("chat", []), ("page", "Protocol & Chat")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:18px;font-weight:700;color:#e8eaed'>"
                "\u2695 Clin<span style='color:#22d3ee'>Origin</span> AI</div>"
                "<div style='color:#8b9299;font-size:11px;margin-bottom:14px'>"
                "Clinical data extraction & validation</div>", unsafe_allow_html=True)

    health, herr = api_get("/health")
    if herr:
        st.markdown("<span style='color:#f87171'>\u25cf Backend offline</span>", unsafe_allow_html=True)
        st.caption("Start FastAPI, then reload.")
        st.stop()
    st.markdown("<span style='color:#4ade80'>\u25cf Backend online</span>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Workflow</div>", unsafe_allow_html=True)
    pages = ["Protocol & Chat", "Extraction", "Structured", "Monitoring", "Insights", "Export"]
    for p in pages:
        active = st.session_state.page == p
        if st.button(("\u2192 " if active else "") + p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.session_state.schema:
        st.markdown(f"<div class='section-label'>Loaded</div>"
                    f"<div style='font-size:12px;color:#e8eaed'>{st.session_state.schema.get('study_name','?')}</div>",
                    unsafe_allow_html=True)
    st.markdown(f"<div style='color:#8b9299;font-size:11px;margin-top:8px'>"
                f"Records: {len(st.session_state.records)}</div>", unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1: PROTOCOL & CHAT  (combined)
# ═══════════════════════════════════════════════════════════════════════════
if page == "Protocol & Chat":
    st.markdown("# Protocol & Chat")
    st.caption("Load a study protocol, then ask questions about it. "
               "Answers are grounded in the protocol text with source citations.")

    # ── Load protocol: saved list, path, OR drag-and-drop ──
    st.markdown("<div class='section-label'>1 · Load protocol</div>", unsafe_allow_html=True)

    # Saved protocols on the server
    plist, _ = api_get("/protocols/list")
    saved = plist.get("protocols", []) if plist else []
    if saved:
        st.markdown("<div style='color:#8b9299;font-size:12px;margin-bottom:4px'>"
                    "Saved protocols on server:</div>", unsafe_allow_html=True)
        labels = [
            p.get("label", p["filename"]) + ("  \u2713" if p["parsed"] else "  (not parsed)")
            for p in saved
        ]
        chosen = st.selectbox("Saved protocols", ["\u2014 select \u2014"] + labels,
                              label_visibility="collapsed")
        if chosen != "\u2014 select \u2014":
            sel = saved[labels.index(chosen)]
            if st.button(f"Load: {sel['filename']}"):
                with st.spinner("Parsing protocol (cached if already parsed)..."):
                    res, err = api_post("/protocol/parse", {"pdf_path": sel["path"]})
                if err:
                    st.error(f"Parse failed: {err}")
                else:
                    st.session_state.protocol_id = res["protocol_id"]
                    st.session_state.pdf_path = sel["path"]
                    sch, _ = api_get("/protocol/schema",
                                     params={"protocol_id": res["protocol_id"]})
                    st.session_state.schema = sch
                    st.session_state.chat = []
                    st.success(f"Loaded: {res.get('study_name')}")
                    st.rerun()

    st.markdown("<div style='color:#8b9299;font-size:12px;margin:10px 0 4px 0'>"
                "or load by path / upload:</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        pdf_path = st.text_input("Server-side PDF path",
                                 value=st.session_state.pdf_path or "",
                                 placeholder="protocols/Prot_SAP_000.pdf",
                                 label_visibility="collapsed")
        if st.button("Parse from path", disabled=not pdf_path):
            with st.spinner("Parsing protocol (builds RAG index)..."):
                res, err = api_post("/protocol/parse", {"pdf_path": pdf_path})
            if err:
                st.error(f"Parse failed: {err}")
            else:
                st.session_state.protocol_id = res["protocol_id"]
                st.session_state.pdf_path = pdf_path
                sch, _ = api_get("/protocol/schema", params={"protocol_id": res["protocol_id"]})
                st.session_state.schema = sch
                st.session_state.chat = []
                st.success(f"Loaded: {res.get('study_name')}")
                st.rerun()
    with c2:
        up = st.file_uploader("or drag & drop a PDF", type=["pdf"], label_visibility="collapsed")
        if up is not None and st.button("Parse uploaded PDF"):
            with st.spinner("Uploading & parsing..."):
                res, err = api_post_files("/protocol/parse/upload", {},
                                          [("file", (up.name, up.getvalue(), "application/pdf"))])
            if err:
                st.error(f"Parse failed: {err}")
            else:
                st.session_state.protocol_id = res["protocol_id"]
                st.session_state.pdf_path = res.get("saved_path")
                sch, _ = api_get("/protocol/schema", params={"protocol_id": res["protocol_id"]})
                st.session_state.schema = sch
                st.session_state.chat = []
                st.success(f"Loaded: {res.get('study_name')}")
                st.rerun()

    # ── Schema summary (compact) ──
    if st.session_state.schema:
        sc = st.session_state.schema
        m1, m2, m3 = st.columns(3)
        m1.metric("Study", (sc.get("study_acronym") or sc.get("study_name") or "\u2014")[:22])
        m2.metric("Indication", (sc.get("indication") or "\u2014")[:22])
        fc = sc.get("field_classification", {})
        m3.metric("Fields", sum(len(fc.get(k, [])) for k in ["required", "expected", "permissible"]))
        with st.expander("View schema (required / expected / permissible)"):
            f1, f2, f3 = st.columns(3)
            f1.markdown("**Required**\n\n" + "\n".join(f"- {x}" for x in fc.get("required", [])))
            f2.markdown("**Expected**\n\n" + "\n".join(f"- {x}" for x in fc.get("expected", [])))
            f3.markdown("**Permissible**\n\n" + "\n".join(f"- {x}" for x in fc.get("permissible", [])))

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chatbox ──
    st.markdown("<div class='section-label'>2 · Ask the protocol</div>", unsafe_allow_html=True)
    if not st.session_state.pdf_path:
        st.info("Load a protocol above to start asking questions.")
    else:
        q = st.text_input("question", placeholder="Which patients are excluded?",
                          label_visibility="collapsed", key="chat_q")
        if st.button("Ask", disabled=not q):
            with st.spinner("Retrieving and answering..."):
                res, err = api_post("/protocol/ask",
                                    {"pdf_path": st.session_state.pdf_path, "question": q, "top_k": 4})
            if err:
                st.error(f"Query failed: {err}")
            else:
                st.session_state.chat.insert(0, res)

        # Answers below the box (most recent first)
        for item in st.session_state.chat:
            st.markdown(f"<div style='color:#8b9299;font-size:12px;margin-top:10px'>"
                        f"Q: {item['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'>{item.get('answer','')}</div>", unsafe_allow_html=True)
            chips = "".join(
                f"<span class='src-chip'>{s.get('section')} · p.{s.get('page')}</span>"
                for s in item.get("sources", []))
            if chips:
                st.markdown(chips, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2: EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Extraction":
    st.markdown("# Extraction")
    st.caption("Paste patient source text or drag & drop files. "
               "The system extracts, validates, and classifies each record.")

    if not st.session_state.protocol_id:
        st.info("Load a protocol first (Protocol & Chat page).")
    else:
        text = st.text_area("Source text", height=140,
                            placeholder="Paste lab report / visit note text...",
                            label_visibility="collapsed")
        files = st.file_uploader("or drag & drop .txt / .pdf", type=["txt", "pdf"],
                                accept_multiple_files=True)
        has_input = bool(text.strip()) or bool(files)

        if st.button("Extract & Analyse", disabled=not has_input):
            with st.spinner("Extracting, validating, classifying..."):
                if files:
                    mp = [("files", (f.name, f.getvalue(),
                                     "application/pdf" if f.name.lower().endswith(".pdf") else "text/plain"))
                          for f in files]
                    res, err = api_post_files("/analyze/upload",
                                              {"protocol_id": st.session_state.protocol_id}, mp)
                else:
                    res, err = api_post("/analyze", {"protocol_id": st.session_state.protocol_id,
                                                     "documents": [text.strip()]})
            if err:
                st.error(f"Analysis failed: {err}")
            else:
                st.session_state.records = res.get("results", [])
                if res.get("skipped"):
                    st.warning(f"Skipped: {', '.join(res['skipped'])}")

        for rec in st.session_state.records:
            pid = rec.get("patient_id") or "(no patient_id)"
            v = rec["review_status"]; s = VERDICT.get(v, VERDICT["Review"])
            d = rec["review_detail"]
            st.markdown(f"### Patient: `{pid}`")
            sub = ("Escalated to Block by safety rule" if d.get("safety_rule_applied")
                   else (f"P(Block) = {d['block_probability']:.1%}" if d.get("block_probability") is not None else ""))
            st.markdown(f"<div class='verdict' style='background:{s['bg']}'>"
                        f"<div class='verdict-icon' style='color:{s['c']}'>{s['i']}</div>"
                        f"<div><div class='verdict-label' style='color:{s['c']}'>{s['l']}</div>"
                        f"<div class='verdict-sub'>{sub}</div></div></div>", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            if rec.get("quality_score") is not None:
                a.metric("Quality", f"{rec['quality_score']:.2f}")
            b.metric("Flags", rec["features"].get("total_flags"))
            c.metric("Completeness", f"{rec['features'].get('completeness_score',0):.0%}")
            cf, cp = st.columns(2)
            with cf:
                st.markdown("**Validation flags**")
                if not rec.get("flags"):
                    st.caption("No flags raised.")
                for fl in rec.get("flags", []):
                    sev = fl.get("severity", "low")
                    cls = {"high": "flag-high", "medium": "flag-medium"}.get(sev, "flag-low")
                    st.markdown(f"<span class='{cls}'>\u25cf {fl['message']}</span>", unsafe_allow_html=True)
            with cp:
                st.markdown("**Class probabilities**")
                for cn, p in d["probabilities"].items():
                    st.progress(p, text=f"{cn}: {p:.1%}")
            with st.expander("Extracted data"):
                st.json(rec["data"])
            st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3: STRUCTURED
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Structured":
    st.markdown("# Structured data")
    st.caption("All extracted records in tabular form.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        rows = []
        for rec in st.session_state.records:
            row = dict(rec["data"])
            row["_review"] = rec["review_status"]
            row["_quality"] = rec.get("quality_score")
            row["_flags"] = rec["features"].get("total_flags")
            rows.append(row)
        st.dataframe(rows, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4: MONITORING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Monitoring":
    st.markdown("# Monitoring review")
    st.caption("Review-status overview across all extracted records.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        counts = {"Clean": 0, "Review": 0, "Block": 0}
        for rec in st.session_state.records:
            counts[rec["review_status"]] = counts.get(rec["review_status"], 0) + 1
        a, b, c = st.columns(3)
        a.metric("\u2713 Clean", counts["Clean"])
        b.metric("\u26a0 Review", counts["Review"])
        c.metric("\u2715 Block", counts["Block"])
        st.markdown("<hr>", unsafe_allow_html=True)
        for rec in st.session_state.records:
            v = rec["review_status"]; s = VERDICT.get(v, VERDICT["Review"])
            pid = rec.get("patient_id") or "(no id)"
            st.markdown(f"<span style='color:{s['c']};font-weight:600'>{s['i']} {s['l']}</span> "
                        f"&nbsp; <code>{pid}</code> &nbsp; \u2014 quality "
                        f"{rec.get('quality_score',0):.2f}, {rec['features'].get('total_flags')} flags",
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 5: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Insights":
    st.markdown("# Insights")
    st.caption("Aggregate view over the extracted records.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        import pandas as pd
        recs = st.session_state.records
        counts = {"Clean": 0, "Review": 0, "Block": 0}
        qualities, flags_list = [], []
        for rec in recs:
            counts[rec["review_status"]] = counts.get(rec["review_status"], 0) + 1
            if rec.get("quality_score") is not None:
                qualities.append(rec["quality_score"])
            flags_list.append(rec["features"].get("total_flags", 0))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Review status distribution**")
            st.bar_chart(pd.DataFrame({"count": counts}))
        with c2:
            st.markdown("**Quality score per record**")
            if qualities:
                st.bar_chart(pd.DataFrame({"quality": qualities}))
            else:
                st.caption("No quality scores available.")

        a, b, c = st.columns(3)
        a.metric("Records", len(recs))
        b.metric("Avg quality", f"{(sum(qualities)/len(qualities)):.2f}" if qualities else "\u2014")
        c.metric("Avg flags", f"{(sum(flags_list)/len(flags_list)):.1f}" if flags_list else "\u2014")

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 6: EXPORT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Export":
    st.markdown("# Export")
    st.caption("Download structured, review-ready records as CSV.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        rows = []
        for rec in st.session_state.records:
            row = dict(rec["data"])
            row["review_status"] = rec["review_status"]
            row["quality_score"] = rec.get("quality_score")
            row["total_flags"] = rec["features"].get("total_flags")
            rows.append(row)
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=all_keys); w.writeheader(); w.writerows(rows)
        st.dataframe(rows, use_container_width=True)
        st.download_button("Download CSV", buf.getvalue(),
                          file_name="clinorigin_records.csv", mime="text/csv")