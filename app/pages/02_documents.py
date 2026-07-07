import os
import io
import csv
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Documents \u2014 ClinOrigin", page_icon="\U0001F4C4",
                   layout="wide")

# --- Light theme (matches the platform) -------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --green:#2e7d32; --amber:#f9a825; --red:#c62828;
    --ink:#1a2027; --muted:#5b6670; --line:#e6e9ec; --accent:#1565c0;
}
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.main .block-container { max-width:1050px; padding-top:1.5rem; }
h1 { font-size:22px !important; font-weight:700 !important; color:#1a2027 !important; letter-spacing:-0.02em; }
h2,h3 { color:#1a2027 !important; }
[data-testid="stMetricValue"] { color:#1565c0 !important; font-size:26px !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { color:#5b6670 !important; font-size:12px !important; font-weight:600 !important; text-transform:uppercase; letter-spacing:0.04em; }
.stButton > button { background:#1565c0 !important; color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; font-size:13px !important; padding:8px 16px !important; }
.stButton > button:hover { background:#1976d2 !important; }
.stTextArea textarea, .stTextInput input { background:#fff !important; color:#1a2027 !important;
    border:1px solid #d5dbe1 !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:12.5px !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:#1565c0 !important; box-shadow:0 0 0 1px rgba(21,101,192,0.25) !important; }
[data-testid="stFileUploaderDropzone"] { background:#f7f9fb !important; border:1.5px dashed #a9c3e0 !important; border-radius:10px !important; }
.verdict { border-radius:12px; padding:16px 20px; margin:10px 0; display:flex; align-items:center; gap:16px; }
.verdict-icon { font-size:32px; line-height:1; }
.verdict-label { font-size:22px; font-weight:700; letter-spacing:1px; }
.verdict-sub { font-size:12px; color:#5b6670; margin-top:2px; }
.answer-box { background:#f7f9fb; border:1px solid #d9e2ec; border-left:3px solid #1565c0;
    border-radius:10px; padding:14px 18px; margin:12px 0; font-size:14px; line-height:1.6; color:#1a2027; }
.src-chip { display:inline-block; background:#eef2f6; border:1px solid #d9e2ec; border-radius:6px;
    padding:3px 9px; margin:3px 4px 0 0; font-size:11px; color:#5b6670; font-family:'JetBrains Mono',monospace; }
.flag-high { color:#c62828; font-weight:600; }
.flag-medium { color:#f9a825; }
.flag-low { color:#8b9299; }
.section-label { color:#5b6670; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin:8px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

VERDICT = {
    "Clean":  {"c": "#2e7d32", "bg": "rgba(46,125,50,0.10)",  "i": "\u2713", "l": "CLEAN"},
    "Review": {"c": "#f9a825", "bg": "rgba(249,168,37,0.12)", "i": "\u26a0", "l": "REVIEW"},
    "Block":  {"c": "#c62828", "bg": "rgba(198,40,40,0.10)",  "i": "\u2715", "l": "BLOCK"},
}

# --- Backend helpers --------------------------------------------------------
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

# --- Session state ----------------------------------------------------------
for k, v in [("schema", None), ("protocol_id", None), ("pdf_path", None),
             ("records", []), ("chat", []), ("doc_page", "Protocol & Chat")]:
    if k not in st.session_state:
        st.session_state[k] = v

# --- Header + backend status ------------------------------------------------
st.markdown("# \U0001F4C4 Documents \u2014 ClinOrigin AI")
health, herr = api_get("/health")
if herr:
    st.error("ClinOrigin backend offline. Start the FastAPI backend, then reload.")
    st.stop()

# --- Internal navigation (horizontal, keeps the app sidebar for the platform) ---
NAV = ["Protocol & Chat", "Extraction", "Structured", "Monitoring", "Insights"]
cols = st.columns(len(NAV))
for i, name in enumerate(NAV):
    label = ("\u25CF " if st.session_state.doc_page == name else "") + name
    if cols[i].button(label, key=f"docnav_{name}", use_container_width=True):
        st.session_state.doc_page = name
        st.rerun()
st.markdown("<hr style='border-color:#e6e9ec;margin:8px 0 16px 0'>", unsafe_allow_html=True)

def get_cached_predictions():
    """
    Returns predictions for the current record set, computing (and caching)
    them once via /disease/predict_from_record. Shared by Structured and
    Insights so switching between them doesn't re-trigger API calls.
    Cache key is a hash of the actual record data, so it correctly
    invalidates when the underlying data changes (not just record count).
    """
    import hashlib, json as _json
    indication = (st.session_state.schema or {}).get("indication", "")
    fingerprint = hashlib.md5(
        _json.dumps([r["data"] for r in st.session_state.records],
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    cache_key = f"{indication}|{fingerprint}"

    if st.session_state.get("_pred_cache_key") != cache_key:
        with st.spinner("Computing risk predictions for each patient..."):
            predictions = []
            for rec in st.session_state.records:
                res, err = api_post("/disease/predict_from_record", {
                    "indication": indication, "data": rec["data"],
                })
                predictions.append(res if not err else {"available": False,
                                                        "reason": "request_failed"})
        st.session_state._pred_cache = predictions
        st.session_state._pred_cache_key = cache_key
    return st.session_state._pred_cache


def normalize_sex(val):
    """Best-effort, display-only normalisation of a sex/gender value."""
    if val is None or val == "":
        return "Unknown"
    s = str(val).strip().lower()
    if s in ("1", "female", "f", "woman"):
        return "Female"
    if s in ("2", "male", "m", "man"):
        return "Male"
    return "Unknown"


def donut_chart(labels_values: dict, colors: dict = None, height: int = 300):
    """
    Renders a donut chart for small categorical breakdowns (2-4 values),
    e.g. review status, sex, or prediction outcome. Falls back gracefully
    if all values are zero.

    Design choices to avoid clipped/ghost labels:
    - Zero-value categories are dropped entirely (no floating "0%" label
      for a slice that doesn't visually exist).
    - Percentage text sits INSIDE each slice (textposition="inside"),
      so it can never get cut off at the chart's outer edge.
    - Category names move to a horizontal legend below the chart instead
      of being drawn next to slices, which is what was causing labels to
      overflow past the chart boundary for thin/edge slices.
    """
    import plotly.graph_objects as go

    # Drop zero-value categories -- they'd otherwise render as a "0%"
    # label floating with no visible slice behind it.
    items = [(l, v) for l, v in labels_values.items() if v and v > 0]
    if not items:
        st.caption("No data available.")
        return
    labels = [l for l, _ in items]
    values = [v for _, v in items]

    marker_colors = [colors.get(l, "#c9d2da") for l in labels] if colors else None
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=marker_colors, line=dict(color="#ffffff", width=2)),
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        textfont=dict(size=12, color="#ffffff"),
        sort=False,
    )])
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                   font=dict(size=11, color="#3a434c")),
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


page = st.session_state.doc_page

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
                "or upload a new protocol:</div>", unsafe_allow_html=True)
    up = st.file_uploader("drag & drop a PDF", type=["pdf"], label_visibility="collapsed")
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
    st.caption("All extracted records, with an automatic risk prediction column "
               "computed from each patient's own extracted values.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        predictions = get_cached_predictions()

        rows = []
        for rec, pred in zip(st.session_state.records, predictions):
            row = dict(rec["data"])
            row["_review"] = rec["review_status"]
            row["_quality"] = rec.get("quality_score")
            row["_flags"] = rec["features"].get("total_flags")

            if pred and pred.get("available"):
                row["risk_prediction"] = pred["label"]
                row["risk_probability"] = pred["probability"]
            elif pred and pred.get("reason") == "missing_fields":
                row["risk_prediction"] = "insufficient data"
                row["risk_probability"] = None
            elif pred and pred.get("reason") == "no_model_for_indication":
                row["risk_prediction"] = "no model for this indication"
                row["risk_probability"] = None
            elif pred and pred.get("reason") == "model_not_implemented":
                row["risk_prediction"] = "model not available yet"
                row["risk_probability"] = None
            else:
                row["risk_prediction"] = "\u2014"
                row["risk_probability"] = None
            rows.append(row)

        st.dataframe(rows, use_container_width=True)

        # Explicit CSV download (in addition to the table's built-in export
        # icon) since there's no separate Export page anymore.
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=all_keys); w.writeheader(); w.writerows(rows)
        st.download_button("Download CSV", buf.getvalue(),
                          file_name="clinorigin_records.csv", mime="text/csv")

        with st.expander("Why is a prediction missing for some patients?"):
            st.caption(
                "The system never guesses a prediction from incomplete data. "
                "'insufficient data' means one or more required values "
                "(e.g. blood pressure, cholesterol category) could not be "
                "found in that patient's extracted record.")
            missing_examples = [
                (rec["data"].get("patient_id", "?"), p.get("missing"))
                for rec, p in zip(st.session_state.records, predictions)
                if p and p.get("reason") == "missing_fields"
            ]
            if missing_examples:
                for pid, missing in missing_examples[:10]:
                    st.caption(f"\u2022 {pid}: missing {', '.join(missing)}")

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
    st.caption("Aggregate view over the extracted records: patient demographics, "
               "risk predictions, and underlying data quality.")
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

        # ── Demographics ──
        st.markdown("<div class='section-label'>Demographics</div>", unsafe_allow_html=True)

        ages = []
        for rec in recs:
            raw_age = rec["data"].get("age") or rec["data"].get("Age")
            try:
                if raw_age not in (None, ""):
                    ages.append(int(float(str(raw_age))))
            except ValueError:
                pass

        sex_counts = {"Female": 0, "Male": 0, "Unknown": 0}
        for rec in recs:
            raw_sex = rec["data"].get("sex") or rec["data"].get("Sex") or rec["data"].get("gender")
            sex_counts[normalize_sex(raw_sex)] += 1

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Age distribution**")
            if ages:
                age_bins = pd.cut(ages, bins=[0,30,40,50,60,70,80,120],
                                  labels=["<30","30-39","40-49","50-59","60-69","70-79","80+"])
                age_counts = age_bins.value_counts().sort_index()
                st.bar_chart(age_counts)
                st.caption(f"Mean age: {sum(ages)/len(ages):.0f} years "
                          f"(n={len(ages)} of {len(recs)} records)")
            else:
                st.caption("No age data available.")
        with d2:
            st.markdown("**Sex distribution**")
            donut_chart(sex_counts, colors={"Female": "#8e5fd6", "Male": "#1565c0", "Unknown": "#c9d2da"})

        # ── Risk predictions ──
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Risk predictions</div>", unsafe_allow_html=True)

        predictions = get_cached_predictions()
        pred_counts = {}
        probabilities = []
        for pred in predictions:
            if pred and pred.get("available"):
                label = pred["label"]
                probabilities.append(pred["probability"])
            elif pred and pred.get("reason") == "missing_fields":
                label = "insufficient data"
            elif pred and pred.get("reason") == "no_model_for_indication":
                label = "no model for indication"
            elif pred and pred.get("reason") == "model_not_implemented":
                label = "model not available"
            else:
                label = "unavailable"
            pred_counts[label] = pred_counts.get(label, 0) + 1

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Prediction outcome distribution**")
            pred_colors = {}
            for label in pred_counts:
                low = label.lower()
                if "elevated" in low or "high" in low:
                    pred_colors[label] = "#c62828"
                elif "low" in low:
                    pred_colors[label] = "#2e7d32"
                else:
                    pred_colors[label] = "#c9d2da"  # insufficient data / unavailable / no model
            donut_chart(pred_counts, colors=pred_colors)
        with p2:
            st.markdown("**Risk tier breakdown**")
            if probabilities:
                # Bucket into tiers instead of plotting one bar per patient --
                # a raw per-patient chart becomes unreadable at scale
                # (hundreds/thousands of records). Tiers mirror the app's
                # existing traffic-light language.
                tiers = {"Low risk (<33%)": 0, "Moderate risk (33-66%)": 0, "High risk (>=66%)": 0}
                for p in probabilities:
                    if p < 0.33:
                        tiers["Low risk (<33%)"] += 1
                    elif p < 0.66:
                        tiers["Moderate risk (33-66%)"] += 1
                    else:
                        tiers["High risk (>=66%)"] += 1
                donut_chart(tiers, colors={
                    "Low risk (<33%)": "#2e7d32",
                    "Moderate risk (33-66%)": "#f9a825",
                    "High risk (>=66%)": "#c62828",
                })
                st.caption(f"Mean predicted risk: {sum(probabilities)/len(probabilities):.1%} "
                          f"(n={len(probabilities)} predictions available)")
            else:
                st.caption("No predictions available yet for this record set.")

        # ── Data quality (moved to the end) ──
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Data quality</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Review status distribution**")
            donut_chart(counts, colors={"Clean": "#2e7d32", "Review": "#f9a825", "Block": "#c62828"})
        with c2:
            st.markdown("**Quality tier breakdown**")
            if qualities:
                q_tiers = {"Low quality (<50%)": 0, "Moderate quality (50-80%)": 0, "High quality (>=80%)": 0}
                for q in qualities:
                    if q < 0.5:
                        q_tiers["Low quality (<50%)"] += 1
                    elif q < 0.8:
                        q_tiers["Moderate quality (50-80%)"] += 1
                    else:
                        q_tiers["High quality (>=80%)"] += 1
                donut_chart(q_tiers, colors={
                    "Low quality (<50%)": "#c62828",
                    "Moderate quality (50-80%)": "#f9a825",
                    "High quality (>=80%)": "#2e7d32",
                })
                st.caption(f"Mean quality: {sum(qualities)/len(qualities):.1%} "
                          f"(n={len(qualities)} of {len(recs)} records)")
            else:
                st.caption("No quality scores available.")

        a, b, c = st.columns(3)
        a.metric("Records", len(recs))
        b.metric("Avg quality", f"{(sum(qualities)/len(qualities)):.2f}" if qualities else "\u2014")
        c.metric("Avg flags", f"{(sum(flags_list)/len(flags_list)):.1f}" if flags_list else "\u2014")