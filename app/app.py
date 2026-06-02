import streamlit as st

st.set_page_config(
    page_title="Clinical Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Clinical Data Intelligence & Monitoring System")
st.markdown("""
Welcome. Use the sidebar to navigate between modules.

| Page | What it does |
|---|---|
| **Predict** | Enter patient data → get risk score + SHAP explanation |
| **Anomaly Monitor** | View patients flagged as unusual by Isolation Forest |
| **Model Comparison** | Compare all 4 classifiers across 3 datasets |
| **Documents** | Upload clinical PDFs and ask questions via RAG |
""")

st.info("Make sure you have run the training notebooks before using Predict or Anomaly pages.")
