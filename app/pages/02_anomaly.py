import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.utils.config import MODELS_DIR, DATA_PROC

st.set_page_config(page_title="Anomaly Monitor", layout="wide")
st.title("Anomaly Detection — Patient Review Queue")

@st.cache_resource
def load_anomaly_model(dataset: str):
    path = MODELS_DIR / f"iso_forest_{dataset}.pkl"
    return joblib.load(path) if path.exists() else None

dataset = st.selectbox("Select dataset", ["cardio", "ms", "melanoma"])
model   = load_anomaly_model(dataset)

if model is None:
    st.error("Anomaly model not found. Run notebook 05 first.")
else:
    proc_path = DATA_PROC / f"{dataset}_test.csv"
    if proc_path.exists():
        df = pd.read_csv(proc_path)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values
        scores = model.decision_function(X)
        threshold = st.slider("Anomaly threshold", -1.0, 1.0, -0.1, 0.05)
        df["anomaly_score"] = scores.round(3)
        df["flagged"] = scores < threshold
        flagged = df[df["flagged"]].sort_values("anomaly_score")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total patients", len(df))
        m2.metric("Flagged", len(flagged))
        m3.metric("Flag rate", f"{len(flagged)/len(df)*100:.1f}%")

        st.subheader(f"Flagged patients ({len(flagged)})")
        st.dataframe(flagged, use_container_width=True, height=400)
    else:
        st.warning(f"No processed test data found at {proc_path}. Run preprocessing first.")
