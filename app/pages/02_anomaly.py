"""
Anomaly Detection — single-patient checker.
Loads pre-trained Isolation Forest from models/saved/.
No raw data required — works fully from the saved model.
"""
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Anomaly Monitor", page_icon="🔍", layout="wide")

FEATURES = ["age_years", "gender", "height", "weight", "bmi",
            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
MODELS_DIR = Path("models/saved")


@st.cache_resource(show_spinner=False)
def load_iso():
    pkl = MODELS_DIR / "iso_forest_cardio.pkl"
    if pkl.exists():
        return joblib.load(pkl)
    return None


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Anomaly Detection — Patient Profile Checker")
st.caption("Isolation Forest · Trained on 68,595 cardiovascular records")

iso = load_iso()
if iso is None:
    st.error("Anomaly model not found. The pre-trained model file is missing from models/saved/.")
    st.stop()

st.markdown("""
Enter a patient's details below to check whether their profile is **typical**
for the training cohort or **anomalous** — flagged for clinical review.
""")

st.divider()

# ── Patient input ─────────────────────────────────────────────────────────────
st.subheader("Patient details")
c1, c2, c3 = st.columns(3)

with c1:
    age    = st.slider("Age (years)", 20, 90, 50)
    sex    = st.selectbox("Sex", ["Female", "Male"])
    height = st.slider("Height (cm)", 130, 210, 165)
    weight = st.slider("Weight (kg)", 35, 180, 70)

with c2:
    ap_hi  = st.slider("Systolic BP", 80, 220, 120)
    ap_lo  = st.slider("Diastolic BP", 50, 140, 80)
    chol   = st.selectbox("Cholesterol", [1, 2, 3],
               format_func=lambda v: {1:"Normal", 2:"Above normal", 3:"Well above normal"}[v])
    gluc   = st.selectbox("Glucose", [1, 2, 3],
               format_func=lambda v: {1:"Normal", 2:"Above normal", 3:"Well above normal"}[v])

with c3:
    smoke  = st.checkbox("Smoker")
    alco   = st.checkbox("Alcohol intake")
    active = st.checkbox("Physically active", value=True)
    bmi    = round(weight / (height / 100) ** 2, 1)
    st.metric("Computed BMI", bmi)

row = pd.DataFrame([{
    "age_years": age, "gender": 1 if sex == "Female" else 2,
    "height": height, "weight": weight, "bmi": bmi,
    "ap_hi": ap_hi, "ap_lo": ap_lo,
    "cholesterol": chol, "gluc": gluc,
    "smoke": int(smoke), "alco": int(alco), "active": int(active),
}])

# ── Score ─────────────────────────────────────────────────────────────────────
st.divider()
flag  = iso.predict(row[FEATURES])[0]       # -1 = anomaly, 1 = normal
score = float(iso.decision_function(row[FEATURES])[0])

m1, m2, m3 = st.columns(3)
m1.metric("Isolation Forest score", f"{score:.4f}",
          help="Higher = more typical. Negative = outlier territory.")
m2.metric("Status", "⚠️ Anomalous" if flag == -1 else "✅ Normal")
zone = "Anomalous (< 0)" if score < 0 else ("Borderline (≈ 0)" if score < 0.05 else "Normal (> 0)")
m3.metric("Risk zone", zone, help="Boundary is 0.0 — negative = flagged, positive = normal")

if flag == -1:
    st.warning(
        "⚠️ **This patient profile is flagged as anomalous** relative to the 68,595-patient "
        "training cohort. In a monitoring workflow this record would be routed for clinical "
        "review — possible data-entry error or rare physiology."
    )
else:
    st.success("✅ This patient profile looks **typical** for the cardiovascular cohort.")

# ── Score interpretation ──────────────────────────────────────────────────────
with st.expander("How to interpret the score"):
    st.markdown("""
| Score range | Meaning |
|---|---|
| > 0.05 | Clearly normal — well within the cohort distribution |
| 0 to 0.05 | Borderline — worth a second look |
| -0.05 to 0 | Mildly unusual |
| < -0.05 | Clearly anomalous — flag for review |

The Isolation Forest was trained with `contamination=0.02`, meaning it expects
roughly **2% of records** in a real population to be anomalous.
    """)

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("About this model"):
    st.markdown("""
**Algorithm:** Isolation Forest (scikit-learn)
**Training data:** 68,595 cleaned cardiovascular records (Kaggle)
**Features used:** age, sex, height, weight, BMI, systolic BP, diastolic BP,
cholesterol level, glucose level, smoking, alcohol, physical activity
**Contamination rate:** 2% — the model expects ~2% of any population to be anomalous
    """)
