import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from src.utils.config import MODELS_DIR
from src.models.explain import explain_single_patient

st.set_page_config(page_title="Predict", layout="wide")
st.title("Patient Risk Prediction")

@st.cache_resource
def load_models():
    models, pipelines, features = {}, {}, {}
    for name in ["cardio", "ms", "melanoma"]:
        xgb_path = MODELS_DIR / f"xgboost_{name}.pkl"
        pipe_path = MODELS_DIR / f"pipeline_{name}.pkl"
        feat_path = MODELS_DIR / f"features_{name}.pkl"
        if xgb_path.exists():
            models[name]   = joblib.load(xgb_path)
            pipelines[name] = joblib.load(pipe_path)
            features[name]  = joblib.load(feat_path)
    iso_path = MODELS_DIR / "iso_forest_cardio.pkl"
    anomaly  = joblib.load(iso_path) if iso_path.exists() else None
    return models, pipelines, features, anomaly

models, pipelines, features, anomaly_model = load_models()

# ── Sidebar inputs ────────────────────────────────────────────────────────
st.sidebar.header("Patient data")
domain = st.sidebar.selectbox("Dataset domain", ["cardio", "ms", "melanoma"])
age    = st.sidebar.number_input("Age", 18, 100, 58)
chol   = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 240)
bp     = st.sidebar.number_input("Resting BP (mmHg)", 80, 220, 130)
hr     = st.sidebar.number_input("Max heart rate", 60, 220, 150)

if st.sidebar.button("Run prediction", type="primary"):
    if domain not in models:
        st.error(f"Model for '{domain}' not found. Run notebook 04 first.")
    else:
        model    = models[domain]
        pipeline = pipelines[domain]
        cols     = features[domain]

        # Build patient row — fill missing feature cols with median placeholder 0
        patient_raw = {"age": age, "chol": chol, "trestbps": bp, "thalach": hr}
        row = np.array([[patient_raw.get(c, 0) for c in cols]])
        row_scaled = pipeline.transform(row)

        prob  = model.predict_proba(row_scaled)[0][1]
        label = "🔴 High risk" if prob > 0.65 else ("🟡 Moderate" if prob > 0.35 else "🟢 Low risk")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk probability", f"{prob*100:.1f}%")
        col2.metric("Classification", label)

        if anomaly_model:
            from src.anomaly.detector import score_patient
            a = score_patient(anomaly_model, row_scaled)
            col3.metric("Anomaly score", a["anomaly_score"],
                        delta="⚠ Flagged" if a["flagged"] else "Normal",
                        delta_color="inverse")

        st.subheader("SHAP explanation")
        try:
            shap_dict = explain_single_patient(model, row_scaled, cols)
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            import pandas as pd, plotly.express as px
            df_shap = pd.DataFrame(sorted_shap, columns=["Feature", "SHAP value"])
            df_shap["Direction"] = df_shap["SHAP value"].apply(lambda v: "Increases risk" if v > 0 else "Decreases risk")
            fig = px.bar(df_shap, x="SHAP value", y="Feature", color="Direction",
                         orientation="h", color_discrete_map={"Increases risk": "#E24B4A", "Decreases risk": "#1D9E75"})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP not available for this model type: {e}")
