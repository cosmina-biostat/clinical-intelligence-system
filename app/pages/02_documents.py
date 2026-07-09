"""
Documents — Clinical Protocol Intelligence
Ziya's UI (5-tab layout + CSS) wired directly to backend/ modules (no FastAPI needed).
"""
import sys, io, csv, tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Documents — CDIM System", page_icon="📄", layout="wide")

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
[data-testid="stMetricLabel"] { color:#5b6670 !important; font-size:12px !important; font-weight:600 !important;
    text-transform:uppercase; letter-spacing:0.04em; }
.stButton > button { background:#1565c0 !important; color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; font-size:13px !important; padding:8px 16px !important; }
.stButton > button:hover { background:#1976d2 !important; }
.stTextArea textarea, .stTextInput input { background:#fff !important; color:#1a2027 !important;
    border:1px solid #d5dbe1 !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:12.5px !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:#1565c0 !important;
    box-shadow:0 0 0 1px rgba(21,101,192,0.25) !important; }
[data-testid="stFileUploaderDropzone"] { background:#f7f9fb !important;
    border:1.5px dashed #a9c3e0 !important; border-radius:10px !important; }
.verdict { border-radius:12px; padding:16px 20px; margin:10px 0; display:flex; align-items:center; gap:16px; }
.verdict-icon { font-size:32px; line-height:1; }
.verdict-label { font-size:22px; font-weight:700; letter-spacing:1px; }
.verdict-sub { font-size:12px; color:#5b6670; margin-top:2px; }
.answer-box { background:#f7f9fb; border:1px solid #d9e2ec; border-left:3px solid #1565c0;
    border-radius:10px; padding:14px 18px; margin:12px 0; font-size:14px; line-height:1.6; color:#1a2027; }
.src-chip { display:inline-block; background:#eef2f6; border:1px solid #d9e2ec; border-radius:6px;
    padding:3px 9px; margin:3px 4px 0 0; font-size:11px; color:#5b6670;
    font-family:'JetBrains Mono',monospace; }
.flag-high { color:#c62828; font-weight:600; }
.flag-medium { color:#f9a825; }
.flag-low { color:#8b9299; }
.section-label { color:#5b6670; font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; margin:8px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

VERDICT = {
    "Clean":  {"c": "#2e7d32", "bg": "rgba(46,125,50,0.10)",  "i": "✓", "l": "CLEAN"},
    "Review": {"c": "#f9a825", "bg": "rgba(249,168,37,0.12)", "i": "⚠", "l": "REVIEW"},
    "Block":  {"c": "#c62828", "bg": "rgba(198,40,40,0.10)",  "i": "✕", "l": "BLOCK"},
}


@st.cache_resource(show_spinner=False)
def _load_backend():
    from backend.protocol_parser import get_schema, get_rag_store
    from backend.extractor import process_documents
    from backend.inference import assess
    from backend.disease_prediction.disease_models import build_registry, resolve_features
    from anthropic import Anthropic
    return get_schema, get_rag_store, process_documents, assess, resolve_features, build_registry(), Anthropic


for k, v in [("schema", None), ("pdf_path", None), ("records", []),
             ("chat", []), ("doc_page", "Protocol & Chat"),
             ("_pred_cache", None), ("_pred_cache_key", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("# 📄 Documents — CDIM System")

NAV = ["Protocol & Chat", "Extraction", "Structured", "Monitoring", "Insights"]
cols = st.columns(len(NAV))
for i, name in enumerate(NAV):
    label = ("● " if st.session_state.doc_page == name else "") + name
    if cols[i].button(label, key=f"docnav_{name}", use_container_width=True):
        st.session_state.doc_page = name
        st.rerun()
st.markdown("<hr style='border-color:#e6e9ec;margin:8px 0 16px 0'>", unsafe_allow_html=True)


def normalize_sex(val):
    if val is None or val == "": return "Unknown"
    s = str(val).strip().lower()
    if s in ("1", "female", "f", "woman"): return "Female"
    if s in ("2", "male",   "m", "man"):   return "Male"
    return "Unknown"


def donut_chart(labels_values: dict, colors: dict = None, height: int = 300):
    import plotly.graph_objects as go
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
        textinfo="percent", textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(size=12, color="#ffffff"), sort=False,
    )])
    fig.update_layout(
        height=height, showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05,
                    xanchor="center", x=0.5, font=dict(size=11, color="#3a434c")),
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def get_cached_predictions():
    import hashlib, json as _json
    _, _, _, _, resolve_features, registry, _ = _load_backend()
    indication  = (st.session_state.schema or {}).get("indication", "")
    fingerprint = hashlib.md5(
        _json.dumps([r["data"] for r in st.session_state.records],
                    sort_keys=True, default=str).encode()
    ).hexdigest()
    cache_key = f"{indication}|{fingerprint}"
    if st.session_state._pred_cache_key != cache_key:
        with st.spinner("Computing risk predictions..."):
            model_card  = registry.match_indication(indication)
            predictions = []
            for rec in st.session_state.records:
                if model_card is None:
                    predictions.append({"available": False, "reason": "no_model_for_indication"})
                    continue
                features, missing = resolve_features(model_card.key, rec["data"])
                if features is None:
                    predictions.append({"available": False, "reason": "missing_fields", "missing": missing})
                    continue
                try:
                    row = pd.DataFrame(
                        [[features[f] for f in model_card.feature_order]],
                        columns=model_card.feature_order,
                    )
                    result = model_card.predict_fn(row)
                    predictions.append({"available": True, "label": result.label,
                                        "probability": result.probability})
                except Exception:
                    predictions.append({"available": False, "reason": "model_not_implemented"})
        st.session_state._pred_cache     = predictions
        st.session_state._pred_cache_key = cache_key
    return st.session_state._pred_cache


page = st.session_state.doc_page

# ── PROTOCOL & CHAT ───────────────────────────────────────────────────────────
if page == "Protocol & Chat":
    st.markdown("# Protocol & Chat")
    st.caption("Load a study protocol, then ask questions about it. "
               "Answers are grounded in the protocol text with source citations.")

    get_schema, get_rag_store, _, _, _, _, Anthropic = _load_backend()

    st.markdown("<div class='section-label'>1 · Load protocol</div>", unsafe_allow_html=True)

    proto_dir = ROOT / "backend" / "protocols"
    raw_dir   = ROOT / "data" / "raw"
    all_pdfs  = (sorted(proto_dir.glob("*.pdf")) if proto_dir.exists() else []) + \
                (sorted(raw_dir.glob("*.pdf"))   if raw_dir.exists()   else [])

    if all_pdfs:
        labels = [p.name for p in all_pdfs]
        chosen = st.selectbox("Saved protocols", ["— select —"] + labels,
                              label_visibility="collapsed")
        if chosen != "— select —":
            sel_path = str(next(p for p in all_pdfs if p.name == chosen))
            if st.button(f"Load: {chosen}"):
                with st.spinner("Parsing protocol (cached if already parsed)..."):
                    try:
                        schema = get_schema(sel_path)
                        st.session_state.schema   = schema
                        st.session_state.pdf_path = sel_path
                        st.session_state.chat     = []
                        st.success(f"Loaded: {schema.get('study_name') or chosen}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Parse failed: {e}")

    st.markdown("<div style='color:#8b9299;font-size:12px;margin:10px 0 4px 0'>"
                "or upload a new protocol:</div>", unsafe_allow_html=True)
    up = st.file_uploader("drag & drop a PDF", type=["pdf"], label_visibility="collapsed")
    if up is not None and st.button("Parse uploaded PDF"):
        proto_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=proto_dir) as tmp:
            tmp.write(up.getvalue())
            tmp_path = tmp.name
        with st.spinner("Uploading & parsing..."):
            try:
                schema = get_schema(tmp_path)
                st.session_state.schema   = schema
                st.session_state.pdf_path = tmp_path
                st.session_state.chat     = []
                st.success(f"Loaded: {schema.get('study_name') or up.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Parse failed: {e}")

    if st.session_state.schema:
        sc = st.session_state.schema
        m1, m2, m3 = st.columns(3)
        m1.metric("Study",      (sc.get("study_acronym") or sc.get("study_name") or "—")[:22])
        m2.metric("Indication", (sc.get("indication") or "—")[:22])
        fc = sc.get("field_classification", {})
        m3.metric("Fields", sum(len(fc.get(k, [])) for k in ["required", "expected", "permissible"]))
        with st.expander("View schema (required / expected / permissible)"):
            f1, f2, f3 = st.columns(3)
            f1.markdown("**Required**\n\n"    + "\n".join(f"- {x}" for x in fc.get("required", [])))
            f2.markdown("**Expected**\n\n"    + "\n".join(f"- {x}" for x in fc.get("expected", [])))
            f3.markdown("**Permissible**\n\n" + "\n".join(f"- {x}" for x in fc.get("permissible", [])))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>2 · Ask the protocol</div>", unsafe_allow_html=True)

    if not st.session_state.pdf_path:
        st.info("Load a protocol above to start asking questions.")
    else:
        q = st.text_input("question", placeholder="Which patients are excluded?",
                          label_visibility="collapsed", key="chat_q")
        if st.button("Ask", disabled=not q):
            with st.spinner("Retrieving and answering..."):
                try:
                    store  = get_rag_store(st.session_state.pdf_path)
                    source = Path(st.session_state.pdf_path).name
                    hits   = store.query(q, top_k=4, source=source)
                    sources = [{"section": h["metadata"].get("section"),
                                "page":    h["metadata"].get("page")} for h in hits]
                    if not hits:
                        answer = "Not found in the provided protocol sections."
                    else:
                        context = "\n\n".join(
                            f"[Source {i} | section: {h['metadata'].get('section')}, "
                            f"page: {h['metadata'].get('page')}]\n{h['text']}"
                            for i, h in enumerate(hits, 1)
                        )
                        prompt = (
                            "You are a clinical protocol assistant. Answer using ONLY the context below.\n"
                            "- Use only information present in the context.\n"
                            "- If not found, reply: \"Not found in the provided protocol sections.\"\n"
                            "- Cite source numbers, e.g. [Source 1]. Be concise and precise.\n\n"
                            f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
                        )
                        client = Anthropic()
                        resp   = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=500, temperature=0,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        answer = "".join(b.text for b in resp.content if b.type == "text").strip()
                    st.session_state.chat.insert(0, {"question": q, "answer": answer, "sources": sources})
                except Exception as e:
                    st.error(f"Query failed: {e}")

        for item in st.session_state.chat:
            st.markdown(f"<div style='color:#8b9299;font-size:12px;margin-top:10px'>"
                        f"Q: {item['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'>{item.get('answer','')}</div>",
                        unsafe_allow_html=True)
            chips = "".join(
                f"<span class='src-chip'>{s.get('section')} · p.{s.get('page')}</span>"
                for s in item.get("sources", []))
            if chips:
                st.markdown(chips, unsafe_allow_html=True)


# ── EXTRACTION ────────────────────────────────────────────────────────────────
elif page == "Extraction":
    st.markdown("# Extraction")
    st.caption("Paste patient source text or drag & drop files. "
               "The system extracts, validates, and classifies each record.")

    _, _, process_documents, assess, _, _, _ = _load_backend()

    if not st.session_state.schema:
        st.info("Load a protocol first (Protocol & Chat page).")
    else:
        text  = st.text_area("Source text", height=140,
                             placeholder="Paste lab report / visit note text...",
                             label_visibility="collapsed")
        files = st.file_uploader("or drag & drop .txt / .pdf", type=["txt", "pdf"],
                                 accept_multiple_files=True)
        has_input = bool(text.strip()) or bool(files)

        if st.button("Extract & Analyse", disabled=not has_input):
            with st.spinner("Extracting, validating, classifying..."):
                try:
                    import pdfplumber
                    documents = []
                    for f in (files or []):
                        if f.name.lower().endswith(".pdf"):
                            with pdfplumber.open(f) as pdf:
                                documents.append("\n".join(p.extract_text() or "" for p in pdf.pages))
                        else:
                            documents.append(f.read().decode("utf-8", errors="ignore"))
                    if text.strip():
                        documents.append(text.strip())
                    records = process_documents(st.session_state.schema, documents)
                    for rec in records:
                        verdict = assess(rec["features"])
                        rec["review_status"] = verdict.get("review_status", "Review")
                        rec["review_detail"] = verdict.get("review_detail", {})
                        rec["quality_score"]  = verdict.get("quality_score")
                        rec["patient_id"]     = rec["data"].get("patient_id")
                    st.session_state.records         = records
                    st.session_state._pred_cache     = None
                    st.session_state._pred_cache_key = None
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        for rec in st.session_state.records:
            pid = rec.get("patient_id") or rec["data"].get("patient_id") or "(no patient_id)"
            v = rec.get("review_status", "Review")
            s = VERDICT.get(v, VERDICT["Review"])
            d = rec.get("review_detail", {})
            st.markdown(f"### Patient: `{pid}`")
            sub = (f"P(Block) = {d['block_probability']:.1%}"
                   if d.get("block_probability") is not None else "")
            st.markdown(
                f"<div class='verdict' style='background:{s['bg']}'>"
                f"<div class='verdict-icon' style='color:{s['c']}'>{s['i']}</div>"
                f"<div><div class='verdict-label' style='color:{s['c']}'>{s['l']}</div>"
                f"<div class='verdict-sub'>{sub}</div></div></div>",
                unsafe_allow_html=True)
            a, b, c = st.columns(3)
            if rec.get("quality_score") is not None:
                a.metric("Quality", f"{rec['quality_score']:.2f}")
            b.metric("Flags",        rec["features"].get("total_flags", 0))
            c.metric("Completeness", f"{rec['features'].get('completeness_score', 0):.0%}")
            cf, cp = st.columns(2)
            with cf:
                st.markdown("**Validation flags**")
                if not rec.get("flags"):
                    st.caption("No flags raised.")
                for fl in rec.get("flags", []):
                    sev = fl.get("severity", "low")
                    cls = {"high": "flag-high", "medium": "flag-medium"}.get(sev, "flag-low")
                    st.markdown(f"<span class='{cls}'>● {fl['message']}</span>", unsafe_allow_html=True)
            with cp:
                st.markdown("**Class probabilities**")
                for cn, p in d.get("probabilities", {}).items():
                    st.progress(float(p), text=f"{cn}: {float(p):.1%}")
            with st.expander("Extracted data"):
                st.json(rec["data"])
            st.markdown("<hr>", unsafe_allow_html=True)


# ── STRUCTURED ────────────────────────────────────────────────────────────────
elif page == "Structured":
    st.markdown("# Structured data")
    st.caption("All extracted records with review status, quality, risk predictions, and anomaly profiles.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        predictions = get_cached_predictions()
        _, _, _, _, resolve_features, _, _ = _load_backend()

        # Load isolation forest for anomaly detection
        @st.cache_resource(show_spinner=False)
        def _load_iso():
            import joblib as _jl
            p = Path("models/saved/iso_forest_cardio.pkl")
            return _jl.load(p) if p.exists() else None
        iso = _load_iso()

        _ISO_FEATURES = ["age_years", "gender", "height", "weight", "bmi",
                         "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]

        # Build rows for the dataframe
        rows = []
        for rec, pred in zip(st.session_state.records, predictions):
            row = dict(rec["data"])
            row["Review"]       = rec.get("review_status", "Review")
            row["Quality %"]    = round((rec.get("quality_score") or 0) * 100, 1)
            row["Completeness %"] = round((rec["features"].get("completeness_score") or 0) * 100, 1)
            row["Flags"]        = int(rec["features"].get("total_flags", 0))
            row["High Severity Flags"] = int(rec["features"].get("high_severity_flags", 0))
            if pred and pred.get("available"):
                row["Risk Prediction"]  = pred["label"]
                row["Risk Probability"] = pred["probability"]
            elif pred and pred.get("reason") == "missing_fields":
                row["Risk Prediction"]  = "insufficient data"
                row["Risk Probability"] = None
            elif pred and pred.get("reason") == "no_model_for_indication":
                row["Risk Prediction"]  = "no model for this indication"
                row["Risk Probability"] = None
            else:
                row["Risk Prediction"]  = "—"
                row["Risk Probability"] = None
            rows.append(row)

        df = pd.DataFrame(rows)

        # ── Column visibility filter ────────────────────────────────────────────
        system_cols = ["Review", "Quality %", "Completeness %", "Flags",
                       "High Severity Flags", "Risk Prediction", "Risk Probability"]
        data_cols   = [c for c in df.columns if c not in system_cols]

        with st.expander("Column visibility", expanded=False):
            shown_data = st.multiselect(
                "Patient data fields",
                options=data_cols,
                default=data_cols[:min(8, len(data_cols))],
            )
            shown_sys = st.multiselect(
                "System fields",
                options=system_cols,
                default=system_cols,
            )

        display_cols = [c for c in shown_data + shown_sys if c in df.columns]
        display_df   = df[display_cols] if display_cols else df

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Review": st.column_config.TextColumn("Review"),
                "Quality %": st.column_config.ProgressColumn(
                    "Quality %", min_value=0, max_value=100, format="%.1f%%"),
                "Completeness %": st.column_config.ProgressColumn(
                    "Completeness %", min_value=0, max_value=100, format="%.1f%%"),
                "Flags": st.column_config.NumberColumn("Flags", format="%d ⚑"),
                "High Severity Flags": st.column_config.NumberColumn("High Sev. Flags", format="%d"),
                "Risk Prediction": st.column_config.TextColumn("Risk Prediction"),
                "Risk Probability": st.column_config.ProgressColumn(
                    "Risk Probability", min_value=0, max_value=1, format="%.1%"),
            },
        )

        all_keys = list(dict.fromkeys(k for r in rows for k in r))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=all_keys)
        w.writeheader(); w.writerows(rows)
        st.download_button("Download CSV", buf.getvalue(),
                           file_name="cdim_records.csv", mime="text/csv")

        # ── Per-patient detail cards ────────────────────────────────────────────
        st.markdown("<hr style='border-color:#e6e9ec;margin:20px 0'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Patient detail cards</div>",
                    unsafe_allow_html=True)

        for i, (rec, pred) in enumerate(zip(st.session_state.records, predictions)):
            pid = (rec.get("patient_id") or rec["data"].get("patient_id")
                   or f"Patient {i + 1}")

            with st.expander(f"Patient: {pid}", expanded=(i == 0)):
                c1, c2, c3 = st.columns(3)

                # Review status
                v = rec.get("review_status", "Review")
                s = VERDICT.get(v, VERDICT["Review"])
                c1.markdown(
                    f"<div style='text-align:center;padding:10px;background:{s['bg']};"
                    f"border-radius:10px'>"
                    f"<div style='font-size:24px;color:{s['c']}'>{s['i']}</div>"
                    f"<div style='font-weight:700;color:{s['c']};font-size:13px'>{s['l']}</div>"
                    f"<div style='font-size:11px;color:#5b6670'>Review status</div></div>",
                    unsafe_allow_html=True)

                # Quality
                qs = rec.get("quality_score") or 0
                qcolor = "#2e7d32" if qs >= 0.8 else ("#f9a825" if qs >= 0.6 else "#c62828")
                c2.markdown(
                    f"<div style='text-align:center;padding:10px'>"
                    f"<div style='font-size:22px;font-weight:700;color:{qcolor}'>{qs:.0%}</div>"
                    f"<div style='font-size:11px;color:#5b6670;margin-bottom:6px'>Quality score</div>"
                    f"<div style='background:#e6e9ec;border-radius:6px;height:8px;width:100%'>"
                    f"<div style='background:{qcolor};width:{qs*100:.0f}%;height:8px;border-radius:6px'>"
                    f"</div></div></div>",
                    unsafe_allow_html=True)

                # Completeness
                cs = rec["features"].get("completeness_score") or 0
                ccolor = "#2e7d32" if cs >= 1.0 else ("#f9a825" if cs >= 0.6 else "#c62828")
                c3.markdown(
                    f"<div style='text-align:center;padding:10px'>"
                    f"<div style='font-size:22px;font-weight:700;color:{ccolor}'>{cs:.0%}</div>"
                    f"<div style='font-size:11px;color:#5b6670;margin-bottom:6px'>Completeness</div>"
                    f"<div style='background:#e6e9ec;border-radius:6px;height:8px;width:100%'>"
                    f"<div style='background:{ccolor};width:{cs*100:.0f}%;height:8px;border-radius:6px'>"
                    f"</div></div></div>",
                    unsafe_allow_html=True)

                # Flags
                total_flags = int(rec["features"].get("total_flags", 0))
                high_flags  = int(rec["features"].get("high_severity_flags", 0))
                flag_color  = "#c62828" if high_flags > 0 else ("#f9a825" if total_flags > 0 else "#2e7d32")
                st.markdown(
                    f"<div style='padding:6px 0 4px 0;font-size:12px;color:#5b6670'>"
                    f"Flags: <span style='font-weight:700;color:{flag_color}'>{total_flags} total"
                    f"</span> &nbsp;·&nbsp; "
                    f"<span style='color:#c62828;font-weight:600'>{high_flags} high severity</span>"
                    f"</div>",
                    unsafe_allow_html=True)

                st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

                # Risk prediction + probability
                if pred and pred.get("available"):
                    prob  = pred["probability"]
                    label = pred["label"]
                    pcolor = ("#c62828" if prob >= 0.66
                              else "#f9a825" if prob >= 0.33 else "#2e7d32")
                    st.markdown(
                        f"<div class='section-label'>Risk prediction</div>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='font-size:15px;font-weight:700;color:{pcolor};"
                        f"margin-bottom:6px'>{label}</div>",
                        unsafe_allow_html=True)
                    st.progress(prob, text=f"Risk probability: {prob:.1%}")
                elif pred and pred.get("reason") == "missing_fields":
                    missing = pred.get("missing", [])
                    st.markdown(
                        f"<div class='answer-box' style='border-left-color:#f9a825'>"
                        f"⚠️ <b>Risk prediction: insufficient data</b><br>"
                        f"Missing fields: <code>{', '.join(missing)}</code></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='answer-box' style='border-left-color:#c9d2da'>"
                        "ℹ️ No risk model available for this indication.</div>",
                        unsafe_allow_html=True)

                # Anomaly profile
                if iso is not None:
                    feats, _ = resolve_features("cardio", rec["data"])
                    if feats is not None:
                        import numpy as np
                        feat_row = pd.DataFrame(
                            [[feats[f] for f in _ISO_FEATURES]], columns=_ISO_FEATURES)
                        iso_flag  = iso.predict(feat_row)[0]
                        iso_score = float(iso.decision_function(feat_row)[0])
                        zone  = ("Anomalous" if iso_score < 0
                                 else "Borderline" if iso_score < 0.05 else "Normal")
                        acolor = ("#c62828" if iso_score < 0
                                  else "#f9a825" if iso_score < 0.05 else "#2e7d32")
                        st.markdown("<div class='section-label'>Anomaly profile</div>",
                                    unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='answer-box' style='border-left-color:{acolor}'>"
                            f"{'⚠️ <b>Anomalous profile</b>' if iso_flag == -1 else '✅ <b>Normal profile</b>'}"
                            f" — Isolation Forest score: <code>{iso_score:.4f}</code> "
                            f"({zone})</div>",
                            unsafe_allow_html=True)


# ── MONITORING ────────────────────────────────────────────────────────────────
elif page == "Monitoring":
    st.markdown("# Monitoring review")
    st.caption("Review-status overview across all extracted records.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        counts = {"Clean": 0, "Review": 0, "Block": 0}
        for rec in st.session_state.records:
            v = rec.get("review_status", "Review")
            counts[v] = counts.get(v, 0) + 1
        a, b, c = st.columns(3)
        a.metric("✓ Clean",  counts["Clean"])
        b.metric("⚠ Review", counts["Review"])
        c.metric("✕ Block",  counts["Block"])
        st.markdown("<hr>", unsafe_allow_html=True)
        for rec in st.session_state.records:
            v   = rec.get("review_status", "Review")
            s   = VERDICT.get(v, VERDICT["Review"])
            pid = rec.get("patient_id") or rec["data"].get("patient_id") or "(no id)"
            qs  = rec.get("quality_score") or 0
            st.markdown(
                f"<span style='color:{s['c']};font-weight:600'>{s['i']} {s['l']}</span>"
                f" &nbsp; <code>{pid}</code> &nbsp; — quality "
                f"{qs:.2f}, {rec['features'].get('total_flags', 0)} flags",
                unsafe_allow_html=True)


# ── INSIGHTS ──────────────────────────────────────────────────────────────────
elif page == "Insights":
    st.markdown("# Insights")
    st.caption("Aggregate view: demographics, risk predictions, and data quality.")
    if not st.session_state.records:
        st.info("No records yet. Run an extraction.")
    else:
        recs = st.session_state.records
        counts   = {"Clean": 0, "Review": 0, "Block": 0}
        qualities, flags_list = [], []
        for rec in recs:
            v = rec.get("review_status", "Review")
            counts[v] = counts.get(v, 0) + 1
            if rec.get("quality_score") is not None:
                qualities.append(rec["quality_score"])
            flags_list.append(rec["features"].get("total_flags", 0))

        st.markdown("<div class='section-label'>Demographics</div>", unsafe_allow_html=True)
        ages = []
        for rec in recs:
            raw = rec["data"].get("age") or rec["data"].get("Age")
            try:
                if raw not in (None, ""): ages.append(int(float(str(raw))))
            except ValueError: pass
        sex_counts = {"Female": 0, "Male": 0, "Unknown": 0}
        for rec in recs:
            raw = rec["data"].get("sex") or rec["data"].get("Sex") or rec["data"].get("gender")
            sex_counts[normalize_sex(raw)] += 1

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Age distribution**")
            if ages:
                age_bins = pd.cut(ages, bins=[0,30,40,50,60,70,80,120],
                                  labels=["<30","30–39","40–49","50–59","60–69","70–79","80+"])
                st.bar_chart(age_bins.value_counts().sort_index())
                st.caption(f"Mean age: {sum(ages)/len(ages):.0f} years (n={len(ages)} of {len(recs)})")
            else:
                st.caption("No age data available.")
        with d2:
            st.markdown("**Sex distribution**")
            donut_chart(sex_counts, colors={"Female": "#8e5fd6", "Male": "#1565c0", "Unknown": "#c9d2da"})

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Risk predictions</div>", unsafe_allow_html=True)
        predictions = get_cached_predictions()
        pred_counts, probabilities = {}, []
        for pred in predictions:
            if pred and pred.get("available"):
                label = pred["label"]; probabilities.append(pred["probability"])
            elif pred and pred.get("reason") == "missing_fields":
                label = "insufficient data"
            elif pred and pred.get("reason") == "no_model_for_indication":
                label = "no model for indication"
            else:
                label = "unavailable"
            pred_counts[label] = pred_counts.get(label, 0) + 1

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Prediction outcome distribution**")
            pred_colors = {l: ("#c62828" if "elevated" in l.lower() or "high" in l.lower()
                               else "#2e7d32" if "low" in l.lower() else "#c9d2da")
                           for l in pred_counts}
            donut_chart(pred_counts, colors=pred_colors)
        with p2:
            st.markdown("**Risk tier breakdown**")
            if probabilities:
                tiers = {"Low risk (<33%)": 0, "Moderate risk (33–66%)": 0, "High risk (≥66%)": 0}
                for p in probabilities:
                    if p < 0.33:   tiers["Low risk (<33%)"] += 1
                    elif p < 0.66: tiers["Moderate risk (33–66%)"] += 1
                    else:          tiers["High risk (≥66%)"] += 1
                donut_chart(tiers, colors={"Low risk (<33%)": "#2e7d32",
                                           "Moderate risk (33–66%)": "#f9a825",
                                           "High risk (≥66%)": "#c62828"})
                st.caption(f"Mean predicted risk: {sum(probabilities)/len(probabilities):.1%} "
                           f"(n={len(probabilities)} predictions available)")
            else:
                st.caption("No predictions available yet.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Data quality</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Review status distribution**")
            donut_chart(counts, colors={"Clean": "#2e7d32", "Review": "#f9a825", "Block": "#c62828"})
        with c2:
            st.markdown("**Quality tier breakdown**")
            if qualities:
                q_tiers = {"Low quality (<50%)": 0, "Moderate quality (50–80%)": 0, "High quality (≥80%)": 0}
                for q in qualities:
                    if q < 0.5:   q_tiers["Low quality (<50%)"] += 1
                    elif q < 0.8: q_tiers["Moderate quality (50–80%)"] += 1
                    else:         q_tiers["High quality (≥80%)"] += 1
                donut_chart(q_tiers, colors={"Low quality (<50%)": "#c62828",
                                             "Moderate quality (50–80%)": "#f9a825",
                                             "High quality (≥80%)": "#2e7d32"})
                st.caption(f"Mean quality: {sum(qualities)/len(qualities):.1%} "
                           f"(n={len(qualities)} of {len(recs)} records)")
            else:
                st.caption("No quality scores available.")

        a, b, c = st.columns(3)
        a.metric("Records",     len(recs))
        b.metric("Avg quality", f"{sum(qualities)/len(qualities):.2f}" if qualities else "—")
        c.metric("Avg flags",   f"{sum(flags_list)/len(flags_list):.1f}" if flags_list else "—")
