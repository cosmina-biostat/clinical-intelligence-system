import streamlit as st

st.set_page_config(
    page_title="Clinical Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Clinical Data Intelligence & Monitoring System")
st.markdown("""
We exist to put the power of machine learning directly in the hands of clinicians —
no data science degree required.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🫀 Patient Risk & Anomaly Checker")
    st.markdown(
        "Enter patient vitals → get a cardiovascular risk score, a plain-English explanation "
        "of what is driving it, and an anomaly flag if the profile is unusual."
    )
    st.page_link("pages/01_predict.py", label="Open Patient Checker →")

with col2:
    st.markdown("### 📄 Document Q&A")
    st.markdown(
        "Ask any question about a clinical protocol PDF in plain English. "
        "Claude AI retrieves the relevant passage and answers with the source quoted back."
    )
    st.page_link("pages/02_documents.py", label="Open Document Q&A →")

st.divider()
st.caption(
    "Trained on 68,595 cardiovascular records · "
    "LightGBM · Isolation Forest · SHAP · ChromaDB · Claude Haiku · Streamlit"
)
