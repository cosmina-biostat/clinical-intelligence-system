import streamlit as st

st.set_page_config(
    page_title="Clinical Data Intelligence & Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --green:#2e7d32; --amber:#f9a825; --red:#c62828;
    --ink:#1a2027; --muted:#5b6670; --line:#e6e9ec; --accent:#1565c0;
}
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.main .block-container { max-width:1000px; padding-top:2rem; }
.hero-title { font-size:30px; font-weight:700; color:#1a2027; letter-spacing:-0.02em; margin-bottom:4px; }
.hero-sub { font-size:15px; color:#5b6670; margin-bottom:28px; }
.tile {
    background:#ffffff; border:1px solid #e6e9ec; border-radius:16px;
    padding:30px 28px; height:100%; transition:all .18s ease;
    box-shadow:0 1px 2px rgba(0,0,0,0.03);
}
.tile:hover { border-color:#c9d2da; box-shadow:0 6px 20px rgba(0,0,0,0.07); transform:translateY(-2px); }
.tile-icon { font-size:40px; margin-bottom:14px; }
.tile-title { font-size:20px; font-weight:700; color:#1a2027; margin-bottom:8px; }
.tile-desc { font-size:13.5px; color:#5b6670; line-height:1.6; margin-bottom:16px; min-height:63px; }
.tile-feat { font-size:12.5px; color:#3a434c; margin:3px 0; }
.badge { display:inline-block; background:#f0f3f6; color:#5b6670; border-radius:20px;
    padding:3px 11px; font-size:11px; font-weight:600; margin-right:6px; }
.stButton > button {
    background:#1565c0 !important; color:#fff !important; border:none !important;
    border-radius:9px !important; font-weight:600 !important; font-size:14px !important;
    padding:9px 20px !important; width:100%;
}
.stButton > button:hover { background:#1976d2 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-title'>🏥 Clinical Data Intelligence &amp; Monitoring</div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>One platform for clinical prediction and document intelligence "
            "— built for researchers, data teams, and clinicians.</div>",
            unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("""
    <div class='tile'>
        <div class='tile-icon'>🫀</div>
        <div class='tile-title'>Patient Risk & Anomaly Checker</div>
        <div class='tile-desc'>Patient-level cardiovascular risk with transparent,
            explainable models and record-level anomaly monitoring.</div>
        <div class='tile-feat'>• CVD risk prediction (LightGBM)</div>
        <div class='tile-feat'>• Per-patient explanation (SHAP)</div>
        <div class='tile-feat'>• Anomaly monitoring (Isolation Forest)</div>
        <div style='margin-top:14px'>
            <span class='badge'>ML</span><span class='badge'>SHAP</span><span class='badge'>Monitoring</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Open Patient Checker →", key="go_pred"):
        st.switch_page("pages/01_predict.py")

with c2:
    st.markdown("""
    <div class='tile'>
        <div class='tile-icon'>📄</div>
        <div class='tile-title'>Documents — ClinOrigin AI</div>
        <div class='tile-desc'>Turn clinical study protocols into structured, validated
            data — with grounded Q&amp;A and traffic-light review verdicts.</div>
        <div class='tile-feat'>• Protocol parsing &amp; RAG chat</div>
        <div class='tile-feat'>• Extraction → Clean / Review / Block</div>
        <div class='tile-feat'>• Risk predictions &amp; insights</div>
        <div style='margin-top:14px'>
            <span class='badge'>RAG</span><span class='badge'>Extraction</span><span class='badge'>Anti-hallucination</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Open Documents →", key="go_docs"):
        st.switch_page("pages/02_documents.py")

st.divider()
st.markdown(
    "<div style='color:#5b6670;font-size:12.5px'>"
    "<b>Stack</b> · LightGBM · SHAP · Isolation Forest · ChromaDB · Claude API · Streamlit<br>"
    "<b>Trained on</b> 68,595 cardiovascular records · ROC-AUC 0.8065</div>",
    unsafe_allow_html=True)
