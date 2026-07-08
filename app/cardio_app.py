"""
Clinical Data Intelligence — Cardiovascular Risk App
====================================================

A Streamlit proof-of-concept built on the cardio_train analysis:
  • Patient CVD risk prediction (LightGBM)
  • Per-patient explanation (SHAP)
  • Record-level anomaly check (Isolation Forest) — the "monitoring" layer
  • Dataset & model overview

Run with:
    streamlit run cardio_app.py

The app looks for data in this order:
    1. data/processed/cardio_clean.csv   (already cleaned by the notebook)
    2. data/raw/cardio_train.csv          (raw; cleaned on the fly)
    3. a file you upload in the sidebar
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
import shap

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
FEATURES = ["age_years", "gender", "height", "weight", "bmi",
            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
NUMERIC = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CODED = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

# Tuned LightGBM params (from the notebook's RandomizedSearchCV result).
LGBM_PARAMS = dict(n_estimators=423, max_depth=7, learning_rate=0.02,
                   num_leaves=15, subsample=0.66, colsample_bytree=0.80,
                   reg_lambda=1.2, random_state=RANDOM_STATE, verbose=-1)

st.set_page_config(page_title="Cardio Risk — Clinical Intelligence",
                   page_icon="🫀", layout="wide")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same structural cleaning as the EDA notebook."""
    if "id" in df.columns:
        df = df.drop(columns="id")
    df = df.drop_duplicates().reset_index(drop=True)
    if "age_years" not in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)
    if "bmi" not in df.columns:
        df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    if "age" in df.columns:
        df = df.drop(columns="age")
    df = df[
        df["ap_hi"].between(60, 250)
        & df["ap_lo"].between(40, 200)
        & (df["ap_hi"] >= df["ap_lo"])
        & df["height"].between(120, 220)
        & df["weight"].between(30, 200)
        & df["bmi"].between(10, 60)
    ].copy()
    return df


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes=None) -> pd.DataFrame:
    """Load from processed -> raw -> uploaded file, cleaning as needed."""
    processed = Path("data/processed/cardio_clean.csv")
    raw = Path("data/raw/cardio_train.csv")

    if processed.exists():
        return pd.read_csv(processed)
    if raw.exists():
        df = pd.read_csv(raw, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(raw, sep=",")
        return clean_raw(df)
    if uploaded_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(uploaded_bytes), sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(BytesIO(uploaded_bytes), sep=",")
        return clean_raw(df)
    return None


# --------------------------------------------------------------------------- #
# Models (cached resources)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train the LightGBM pipeline; return pipeline + held-out test metrics."""
    X, y = df[FEATURES], df["cardio"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("pass", "passthrough", CODED)])
    pipe = Pipeline([("prep", prep), ("model", LGBMClassifier(**LGBM_PARAMS))])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {"roc_auc": roc_auc_score(y_te, proba),
               "accuracy": accuracy_score(y_te, pipe.predict(X_te)),
               "n_train": len(X_tr), "n_test": len(X_te)}
    return pipe, metrics


@st.cache_resource(show_spinner=False)
def train_anomaly(df: pd.DataFrame):
    iso = IsolationForest(n_estimators=200, contamination=0.02,
                          random_state=RANDOM_STATE)
    iso.fit(df[FEATURES])
    return iso


@st.cache_resource(show_spinner=False)
def get_explainer(_pipe):
    """SHAP TreeExplainer on the fitted LightGBM step."""
    return shap.TreeExplainer(_pipe.named_steps["model"])


def shap_for_instance(pipe, explainer, row_df):
    """Return (feature_names, shap_values, base_value) for one patient row."""
    x = pipe.named_steps["prep"].transform(row_df[FEATURES])
    sv = explainer.shap_values(x)
    sv = sv[1] if isinstance(sv, list) else sv          # positive class
    base = explainer.expected_value
    base = base[1] if isinstance(base, (list, np.ndarray)) else base
    return FEATURES, np.array(sv).ravel(), float(base)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def patient_sidebar():
    st.sidebar.header("Patient details")
    age = st.sidebar.slider("Age (years)", 20, 90, 50)
    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    height = st.sidebar.slider("Height (cm)", 130, 210, 165)
    weight = st.sidebar.slider("Weight (kg)", 35, 180, 70)
    ap_hi = st.sidebar.slider("Systolic BP (ap_hi)", 80, 220, 120)
    ap_lo = st.sidebar.slider("Diastolic BP (ap_lo)", 50, 140, 80)
    chol = st.sidebar.selectbox("Cholesterol", [1, 2, 3],
                                format_func=lambda v: {1: "Normal", 2: "Above normal",
                                                       3: "Well above normal"}[v])
    gluc = st.sidebar.selectbox("Glucose", [1, 2, 3],
                                format_func=lambda v: {1: "Normal", 2: "Above normal",
                                                       3: "Well above normal"}[v])
    smoke = st.sidebar.checkbox("Smoker")
    alco = st.sidebar.checkbox("Alcohol intake")
    active = st.sidebar.checkbox("Physically active", value=True)

    bmi = round(weight / (height / 100) ** 2, 1)
    st.sidebar.metric("Computed BMI", bmi)

    # gender encoding in cardio_train: 1 = women, 2 = men
    row = {
        "age_years": age, "gender": 1 if sex == "Female" else 2,
        "height": height, "weight": weight, "bmi": bmi,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
        "smoke": int(smoke), "alco": int(alco), "active": int(active),
    }
    return pd.DataFrame([row])


def tab_prediction(pipe, row_df):
    st.subheader("Cardiovascular disease risk")
    prob = float(pipe.predict_proba(row_df[FEATURES])[:, 1][0])
    band = "Low" if prob < 0.33 else ("Moderate" if prob < 0.66 else "High")
    color = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#c62828"}[band]

    c1, c2 = st.columns([1, 2])
    c1.metric("Predicted risk", f"{prob*100:.1f}%")
    c1.markdown(f"<h3 style='color:{color}'>{band} risk</h3>", unsafe_allow_html=True)
    c2.progress(prob)
    c2.caption("Model: LightGBM, trained on the cardio_train cohort. "
               "This is a proof-of-concept, **not** a clinical diagnostic tool.")


def tab_explain(pipe, explainer, row_df):
    st.subheader("Why this prediction? (SHAP)")
    names, sv, base = shap_for_instance(pipe, explainer, row_df)
    order = np.argsort(np.abs(sv))[::-1]
    names = [names[i] for i in order]
    sv = sv[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c62828" if v > 0 else "#2e7d32" for v in sv]
    ax.barh(names[::-1], sv[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value  (red = pushes risk up, green = pushes risk down)")
    ax.set_title("Feature contributions for this patient")
    st.pyplot(fig)
    st.caption("Bars to the right increase predicted risk; bars to the left lower it. "
               "Length = strength of the effect for this specific patient.")


def tab_anomaly(iso, row_df):
    st.subheader("Record monitoring — is this patient unusual?")
    flag = iso.predict(row_df[FEATURES])[0]        # -1 anomaly, 1 normal
    score = iso.decision_function(row_df[FEATURES])[0]
    if flag == -1:
        st.warning("⚠️ This record is flagged as **anomalous** relative to the cohort. "
                   "In a monitoring setting it would be routed for review (possible "
                   "data-entry error or rare physiology).")
    else:
        st.success("✅ This record looks **typical** for the cohort.")
    st.metric("Isolation Forest score", f"{score:.3f}",
              help="Higher = more normal; negative = outlier territory.")


def tab_overview(df, metrics):
    st.subheader("Dataset & model overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients (clean)", f"{len(df):,}")
    c2.metric("CVD prevalence", f"{df['cardio'].mean()*100:.1f}%")
    c3.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c4.metric("Test accuracy", f"{metrics['accuracy']:.3f}")

    st.markdown("**Class balance**")
    st.bar_chart(df["cardio"].value_counts().rename({0: "No CVD", 1: "CVD"}))

    st.markdown("**Sample of the cleaned data**")
    st.dataframe(df.head(20), use_container_width=True)


def main():
    st.title("🫀 Cardiovascular Risk — Clinical Data Intelligence (PoC)")

    uploaded = st.sidebar.file_uploader("Optional: upload cardio_train.csv", type="csv")
    df = load_data(uploaded.getvalue() if uploaded else None)

    if df is None:
        st.error("No data found. Place `cardio_train.csv` in `data/raw/` (or the cleaned "
                 "file in `data/processed/`), or upload it in the sidebar.")
        st.stop()

    pipe, metrics = train_model(df)
    iso = train_anomaly(df)
    explainer = get_explainer(pipe)

    row_df = patient_sidebar()

    t1, t2, t3, t4 = st.tabs(["Risk prediction", "Explainability",
                              "Monitoring", "Data & model"])
    with t1:
        tab_prediction(pipe, row_df)
    with t2:
        tab_explain(pipe, explainer, row_df)
    with t3:
        tab_anomaly(iso, row_df)
    with t4:
        tab_overview(df, metrics)


if __name__ == "__main__":
    main()
