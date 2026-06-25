"""
Risk Prediction — Cardiovascular domain
Loads pre-trained LightGBM pipeline from models/saved/.
Falls back to inline training only if the .pkl is missing AND data is available.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from pathlib import Path
import shap

st.set_page_config(page_title="Risk Prediction", page_icon="🫀", layout="wide")

RANDOM_STATE = 42
FEATURES = ["age_years", "gender", "height", "weight", "bmi",
            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
NUMERIC  = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CODED    = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

MODELS_DIR = Path("models/saved")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load pre-trained pipeline from disk."""
    pkl = MODELS_DIR / "lgbm_pipeline_cardio.pkl"
    if pkl.exists():
        return joblib.load(pkl)
    # fallback: train inline if data is available
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    df = _load_data_optional()
    if df is None:
        return None
    X, y = df[FEATURES], df["cardio"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("pass", "passthrough", CODED)])
    pipe = Pipeline([("prep", prep), ("model", LGBMClassifier(
        n_estimators=423, max_depth=7, learning_rate=0.02, num_leaves=15,
        subsample=0.66, colsample_bytree=0.80, reg_lambda=1.2,
        random_state=RANDOM_STATE, verbose=-1))])
    pipe.fit(X_tr, y_tr)
    return pipe


@st.cache_resource(show_spinner=False)
def get_explainer(_pipe):
    return shap.TreeExplainer(_pipe.named_steps["model"])


def _load_data_optional():
    """Returns dataframe if data files exist, else None — never errors."""
    proc = Path("data/processed/cardio_clean.csv")
    raw  = Path("data/raw/cardio_train.csv")
    try:
        if proc.exists():
            return pd.read_csv(proc)
        if raw.exists():
            df = pd.read_csv(raw, sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(raw, sep=",")
            return _clean(df)
    except Exception:
        pass
    return None


def _clean(df):
    if "id" in df.columns: df = df.drop(columns="id")
    df = df.drop_duplicates().reset_index(drop=True)
    if "age_years" not in df.columns: df["age_years"] = (df["age"] / 365.25).round(1)
    if "bmi" not in df.columns: df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    if "age" in df.columns: df = df.drop(columns="age")
    return df[df["ap_hi"].between(60,250) & df["ap_lo"].between(40,200) &
              (df["ap_hi"] >= df["ap_lo"]) & df["height"].between(120,220) &
              df["weight"].between(30,200) & df["bmi"].between(10,60)].copy()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.title("🫀 Patient Risk Prediction")
st.caption("Cardiovascular disease · LightGBM + SHAP")

st.sidebar.header("Enter patient details")
age    = st.sidebar.slider("Age (years)", 20, 90, 50)
sex    = st.sidebar.selectbox("Sex", ["Female", "Male"])
height = st.sidebar.slider("Height (cm)", 130, 210, 165)
weight = st.sidebar.slider("Weight (kg)", 35, 180, 70)
ap_hi  = st.sidebar.slider("Systolic BP", 80, 220, 120)
ap_lo  = st.sidebar.slider("Diastolic BP", 50, 140, 80)
chol   = st.sidebar.selectbox("Cholesterol", [1, 2, 3],
           format_func=lambda v: {1:"Normal", 2:"Above normal", 3:"Well above normal"}[v])
gluc   = st.sidebar.selectbox("Glucose", [1, 2, 3],
           format_func=lambda v: {1:"Normal", 2:"Above normal", 3:"Well above normal"}[v])
smoke  = st.sidebar.checkbox("Smoker")
alco   = st.sidebar.checkbox("Alcohol intake")
active = st.sidebar.checkbox("Physically active", value=True)

bmi = round(weight / (height / 100) ** 2, 1)
st.sidebar.metric("Computed BMI", bmi)

row_df = pd.DataFrame([{
    "age_years": age, "gender": 1 if sex == "Female" else 2,
    "height": height, "weight": weight, "bmi": bmi,
    "ap_hi": ap_hi, "ap_lo": ap_lo,
    "cholesterol": chol, "gluc": gluc,
    "smoke": int(smoke), "alco": int(alco), "active": int(active),
}])

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    pipe = load_pipeline()

if pipe is None:
    st.error("Model not found. The pre-trained model file is missing from models/saved/.")
    st.stop()

explainer = get_explainer(pipe)

# ── Prediction ────────────────────────────────────────────────────────────────
prob  = float(pipe.predict_proba(row_df[FEATURES])[:, 1][0])
band  = "Low" if prob < 0.33 else ("Moderate" if prob < 0.66 else "High")
color = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#c62828"}[band]

tab1, tab2 = st.tabs(["Risk score", "SHAP explanation"])

with tab1:
    c1, c2 = st.columns([1, 2])
    c1.metric("Predicted CVD risk", f"{prob * 100:.1f}%")
    c1.markdown(f"<h3 style='color:{color}'>{band} risk</h3>", unsafe_allow_html=True)
    c2.progress(prob)
    c2.caption("Model: LightGBM trained on 70k cardio records. **Not** a clinical diagnostic tool.")

with tab2:
    st.subheader("Why this score? See which factors matter most for this patient.")
    x  = pipe.named_steps["prep"].transform(row_df[FEATURES])
    sv = explainer.shap_values(x)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.array(sv).ravel()
    order  = np.argsort(np.abs(sv))[::-1]
    names  = [FEATURES[i] for i in order]
    values = sv[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#c62828" if v > 0 else "#2e7d32" for v in values]
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value  (red = raises risk · green = lowers risk)")
    ax.set_title("Feature contributions for this patient")
    st.pyplot(fig)
    st.caption("Bar length = how strongly each feature drove the prediction for this specific patient.")

with st.expander("Other disease domains (coming soon)"):
    st.info("MS and Melanoma models are in development. The cardiovascular model uses the complete 70k-record Kaggle dataset.")
