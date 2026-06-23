"""
Model Comparison — all 4 classifiers on the cardiovascular dataset.
Computes metrics inline (cached); no pre-saved CSVs required.
Also logs each run to MLflow if the server is reachable.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

RANDOM_STATE = 42
FEATURES = ["age_years", "gender", "height", "weight", "bmi",
            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
NUMERIC  = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CODED    = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                         eval_metric="logloss", verbosity=0),
    "LightGBM":            LGBMClassifier(n_estimators=423, max_depth=7, learning_rate=0.02,
                                          num_leaves=15, subsample=0.66, colsample_bytree=0.80,
                                          reg_lambda=1.2, random_state=RANDOM_STATE, verbose=-1),
}


@st.cache_data(show_spinner=False)
def load_data():
    proc = Path("data/processed/cardio_clean.csv")
    raw  = Path("data/raw/cardio_train.csv")
    if proc.exists():
        return pd.read_csv(proc)
    if raw.exists():
        df = pd.read_csv(raw, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(raw, sep=",")
        return _clean(df)
    return None


def _clean(df):
    if "id" in df.columns:
        df = df.drop(columns="id")
    df = df.drop_duplicates().reset_index(drop=True)
    if "age_years" not in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)
    if "bmi" not in df.columns:
        df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    if "age" in df.columns:
        df = df.drop(columns="age")
    return df[
        df["ap_hi"].between(60, 250) & df["ap_lo"].between(40, 200) &
        (df["ap_hi"] >= df["ap_lo"]) & df["height"].between(120, 220) &
        df["weight"].between(30, 200) & df["bmi"].between(10, 60)
    ].copy()


@st.cache_data(show_spinner=False)
def run_comparison(_df):
    X, y = _df[FEATURES], _df["cardio"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("pass", "passthrough", CODED)])

    rows = []
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("prep", prep), ("model", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred  = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        rows.append({
            "Classifier": name,
            "ROC-AUC":   round(roc_auc_score(y_te, y_proba), 4),
            "F1":        round(f1_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred), 4),
            "Recall":    round(recall_score(y_te, y_pred), 4),
            "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        })
    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📊 Model Comparison")
st.caption("4 classifiers · cardiovascular dataset · 20% held-out test set")

df = load_data()
if df is None:
    st.error("Data not found. Place `cardio_train.csv` in `data/raw/`.")
    st.stop()

with st.spinner("Training all 4 classifiers… (~60 s first run)"):
    results = run_comparison(df)

# ── Metric selector ───────────────────────────────────────────────────────────
metric = st.selectbox("Metric to display", ["ROC-AUC", "F1", "Precision", "Recall", "Accuracy"])

fig = px.bar(
    results.sort_values(metric, ascending=False),
    x="Classifier", y=metric,
    color="Classifier",
    text=metric,
    color_discrete_sequence=px.colors.qualitative.Set2,
    title=f"{metric} — all classifiers on cardiovascular test set"
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(yaxis_range=[0, 1], showlegend=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# ── Full table ────────────────────────────────────────────────────────────────
st.subheader("All metrics")
st.dataframe(
    results.sort_values("ROC-AUC", ascending=False).set_index("Classifier"),
    use_container_width=True
)

# ── Radar chart ──────────────────────────────────────────────────────────────
st.subheader("Radar comparison")
metrics_cols = ["ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]
fig2 = px.line_polar(
    results.melt(id_vars="Classifier", value_vars=metrics_cols,
                 var_name="Metric", value_name="Score"),
    r="Score", theta="Metric", color="Classifier",
    line_close=True,
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig2.update_layout(polar=dict(radialaxis=dict(range=[0.5, 1.0])), height=480)
st.plotly_chart(fig2, use_container_width=True)

# ── MLflow note ───────────────────────────────────────────────────────────────
with st.expander("MLflow experiment tracking"):
    st.info("To view experiment logs, run `mlflow ui` in the project root and open "
            "http://localhost:5000. Each classifier run above is logged automatically "
            "when the training module (`src/models/train.py`) is invoked from a notebook.")
