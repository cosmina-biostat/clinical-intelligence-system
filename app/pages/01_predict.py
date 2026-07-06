"""
Risk Prediction + Anomaly Detection — combined patient checker.
Loads pre-trained LightGBM pipeline and Isolation Forest from models/saved/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap
from pathlib import Path

st.set_page_config(page_title="Patient Risk & Anomaly", page_icon="🫀", layout="wide")

FEATURES = ["age_years", "gender", "height", "weight", "bmi",
            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
NUMERIC  = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CODED    = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
MODELS_DIR = Path("models/saved")


@st.cache_resource(show_spinner=False)
def load_model():
    pkl = MODELS_DIR / "lgbm_model_cardio.pkl"
    if pkl.exists():
        return joblib.load(pkl)
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    pkl = MODELS_DIR / "lgbm_scaler_cardio.pkl"
    if pkl.exists():
        return joblib.load(pkl)
    return None


@st.cache_resource(show_spinner=False)
def load_iso():
    pkl = MODELS_DIR / "iso_forest_cardio.pkl"
    if pkl.exists():
        return joblib.load(pkl)
    return None


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ── Load models ───────────────────────────────────────────────────────────────
model  = load_model()
scaler = load_scaler()
iso    = load_iso()

if model is None or scaler is None:
    st.error("Prediction model not found. Pre-trained model file is missing from models/saved/.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫀 Patient Risk & Anomaly Checker")
st.caption("LightGBM cardiovascular risk · Isolation Forest anomaly detection · 68,595 patient training cohort")

st.divider()

# ── Patient input ─────────────────────────────────────────────────────────────
st.subheader("Enter patient details")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Demographics**")
    age    = st.number_input("Age (years)", min_value=20, max_value=90,  value=50, step=1)
    sex    = st.selectbox("Sex", ["Female", "Male"])
    height = st.number_input("Height (cm)", min_value=130, max_value=210, value=165, step=1)
    weight = st.number_input("Weight (kg)", min_value=35,  max_value=180, value=70,  step=1)

with c2:
    st.markdown("**Vitals**")
    ap_hi  = st.number_input("Systolic BP (mmHg)",  min_value=80,  max_value=220, value=120, step=1)
    ap_lo  = st.number_input("Diastolic BP (mmHg)", min_value=50,  max_value=140, value=80,  step=1)
    chol   = st.selectbox("Cholesterol", [1, 2, 3],
               format_func=lambda v: {1:"Normal", 2:"Above normal", 3:"Well above normal"}[v])
    gluc   = st.selectbox("Glucose", [1, 2, 3],
               format_func=lambda v: {1:"Normal", 2:"Above normal", 3:"Well above normal"}[v])

with c3:
    st.markdown("**Lifestyle**")
    smoke  = st.checkbox("Smoker")
    alco   = st.checkbox("Alcohol intake")
    active = st.checkbox("Physically active", value=True)
    bmi    = round(weight / (height / 100) ** 2, 1)
    st.metric("Computed BMI", bmi)

row_df = pd.DataFrame([{
    "age_years": age, "gender": 1 if sex == "Female" else 2,
    "height": height, "weight": weight, "bmi": bmi,
    "ap_hi": ap_hi, "ap_lo": ap_lo,
    "cholesterol": chol, "gluc": gluc,
    "smoke": int(smoke), "alco": int(alco), "active": int(active),
}])

st.divider()

# ── Preprocess + predict ──────────────────────────────────────────────────────
num_scaled = scaler.transform(row_df[NUMERIC])
X_ready    = np.hstack([num_scaled, row_df[CODED].values])

prob  = float(model.predict_proba(X_ready)[:, 1][0])
band  = "Low" if prob < 0.33 else ("Moderate" if prob < 0.66 else "High")
color = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#c62828"}[band]

if iso is not None:
    flag  = iso.predict(row_df[FEATURES])[0]
    score = float(iso.decision_function(row_df[FEATURES])[0])
    zone  = "Anomalous (< 0)" if score < 0 else ("Borderline (≈ 0)" if score < 0.05 else "Normal (> 0)")
else:
    flag, score, zone = 1, 0.0, "Model unavailable"

m1, m2, m3, m4 = st.columns(4)
m1.metric("CVD Risk Score", f"{prob * 100:.1f}%")
m2.markdown(f"<div style='padding-top:0.4rem'><b>Risk level</b><br><span style='font-size:1.4rem;color:{color}'>{band}</span></div>", unsafe_allow_html=True)
m3.metric("Anomaly Score", f"{score:.4f}", help="Negative = unusual profile. Positive = typical.")
m4.metric("Profile zone", zone)

st.progress(prob)

if flag == -1:
    st.warning("⚠️ **Anomalous profile** — this patient's vitals are unusual relative to the 68,595-patient cohort. Possible data-entry error or rare physiology — recommend clinical review.")
else:
    st.success("✅ Patient profile is **typical** for the cardiovascular cohort.")

st.divider()

# ── SHAP ──────────────────────────────────────────────────────────────────────
st.subheader("What is driving this patient's risk?")
st.caption("Each bar shows how strongly a feature pushed the risk score up (red) or down (green) for this specific patient.")

explainer = get_explainer(model)
sv = explainer.shap_values(X_ready)
sv = sv[1] if isinstance(sv, list) else sv
sv = np.array(sv).ravel()
FEATURE_LABELS = {
    "age_years": "Age", "gender": "Gender", "height": "Height",
    "weight": "Weight", "bmi": "BMI", "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP", "cholesterol": "Cholesterol",
    "gluc": "Glucose", "smoke": "Smoke", "alco": "Alcohol",
    "active": "Physically Active",
}

order  = np.argsort(np.abs(sv))[::-1]
names  = [FEATURE_LABELS[FEATURES[i]] for i in order]
values = sv[order]

fig, ax = plt.subplots(figsize=(5, 3))
colors  = ["#c62828" if v > 0 else "#2e7d32" for v in values]
ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("SHAP value  (red = raises risk · green = lowers risk)", fontsize=6)
ax.tick_params(axis="both", labelsize=6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig, use_container_width=False)

with st.expander("How to interpret the anomaly score"):
    st.markdown("""
| Score range | Meaning |
|---|---|
| > 0.05 | Clearly normal — well within the cohort distribution |
| 0 to 0.05 | Borderline — worth a second look |
| -0.05 to 0 | Mildly unusual |
| < -0.05 | Clearly anomalous — flag for review |

The Isolation Forest was trained with `contamination=0.02` — it expects roughly **2% of records** in any real population to be anomalous.
    """)

with st.expander("About the models"):
    st.markdown("""
**Risk model:** LightGBM · trained on 68,595 cardiovascular records · ROC-AUC 0.8065

**Anomaly model:** Isolation Forest · same training cohort · contamination rate 2%

**Not a clinical diagnostic tool.** Predictions are for research and demonstration purposes only.
    """)
