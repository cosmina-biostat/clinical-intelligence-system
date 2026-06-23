import streamlit as st

st.set_page_config(
    page_title="Clinical Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Clinical Data Intelligence & Monitoring System")
st.markdown("""
A production-ready platform combining **ML risk prediction**, **explainability**,
**anomaly detection**, and **document Q&A** for clinical researchers, hospital data teams,
and clinicians.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🫀 Risk Prediction")
    st.markdown("Enter patient vitals → LightGBM returns a CVD risk score in <1 second.")
    st.page_link("pages/01_predict.py", label="Open Risk Prediction →")

    st.markdown("### 🔍 Anomaly Monitor")
    st.markdown("Isolation Forest flags patients whose profiles are unusual vs the training cohort.")
    st.page_link("pages/02_anomaly.py", label="Open Anomaly Monitor →")

with col2:
    st.markdown("### 📊 Model Comparison")
    st.markdown("Side-by-side ROC-AUC / F1 for all 4 classifiers on the cardiovascular dataset.")
    st.page_link("pages/03_models.py", label="Open Model Comparison →")

    st.markdown("### 📄 Document Q&A")
    st.markdown("Upload a clinical PDF → ask questions in plain English → Claude AI answers with context.")
    st.page_link("pages/04_documents.py", label="Open Document Q&A →")

st.divider()
st.markdown("""
**Dataset** · Cardiovascular Disease — 70,000 records (Kaggle)
**Stack** · scikit-learn · XGBoost · LightGBM · SHAP · MLflow · FAISS · Claude API · Streamlit
**Note** · Document Q&A requires an `ANTHROPIC_API_KEY` in your `.env` file.
""")
