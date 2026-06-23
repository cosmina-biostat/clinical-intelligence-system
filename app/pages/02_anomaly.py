"""
Anomaly Detection — population monitoring view.
Isolation Forest is trained inline on the cardio cohort (cached).
"""
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

st.set_page_config(page_title="Anomaly Monitor", page_icon="🔍", layout="wide")

RANDOM_STATE = 42
FEATURES = ["age_years", "gender", "height", "weight", "bmi",
            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
NUMERIC  = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CODED    = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


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


@st.cache_resource(show_spinner=False)
def train_anomaly(_df):
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=RANDOM_STATE)
    iso.fit(_df[FEATURES])
    return iso


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Anomaly Detection — Patient Review Queue")
st.caption("Isolation Forest · Cardiovascular cohort · 70k records")

df = load_data()
if df is None:
    st.error("Data not found. Place `cardio_train.csv` in `data/raw/`.")
    st.stop()

with st.spinner("Training Isolation Forest (first run only)…"):
    iso = train_anomaly(df)

scores    = iso.decision_function(df[FEATURES])
threshold = st.sidebar.slider("Anomaly threshold", -0.20, 0.20, -0.05, 0.01,
                               help="Lower = stricter. Records below threshold are flagged.")

df_view            = df.copy()
df_view["iso_score"] = scores.round(4)
df_view["flagged"]   = scores < threshold
flagged             = df_view[df_view["flagged"]].sort_values("iso_score")

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total patients", f"{len(df_view):,}")
m2.metric("Flagged", f"{len(flagged):,}")
m3.metric("Flag rate", f"{len(flagged) / len(df_view) * 100:.2f}%")
m4.metric("Threshold", f"{threshold:.2f}")

st.divider()

# ── Score distribution ────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Isolation Forest score distribution")
    fig = px.histogram(df_view, x="iso_score", nbins=80,
                       color_discrete_sequence=["#1565c0"],
                       labels={"iso_score": "Isolation Forest score"})
    fig.add_vline(x=threshold, line_dash="dash", line_color="#c62828",
                  annotation_text=f"Threshold {threshold}", annotation_position="top right")
    fig.update_layout(height=320, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Flagged vs normal")
    counts = df_view["flagged"].value_counts().rename({False: "Normal", True: "Flagged"})
    fig2   = px.pie(values=counts.values, names=counts.index,
                    color_discrete_map={"Normal": "#2e7d32", "Flagged": "#c62828"})
    fig2.update_layout(height=320, margin=dict(t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── Flagged records table ─────────────────────────────────────────────────────
st.subheader(f"Flagged patients ({len(flagged):,})")
st.caption("Sorted by most anomalous (lowest score) first. These records "
           "would be routed for clinical review in a monitoring workflow.")
st.dataframe(
    flagged[["age_years", "gender", "height", "weight", "bmi",
             "ap_hi", "ap_lo", "cholesterol", "gluc",
             "smoke", "alco", "active", "cardio", "iso_score"]],
    use_container_width=True, height=400
)

# ── Scatter: age vs BP coloured by flag ──────────────────────────────────────
st.subheader("Age vs Systolic BP — anomalous patients highlighted")
fig3 = px.scatter(
    df_view.sample(min(5000, len(df_view)), random_state=42),
    x="age_years", y="ap_hi",
    color="flagged",
    color_discrete_map={False: "#90caf9", True: "#c62828"},
    opacity=0.5, size_max=4,
    labels={"age_years": "Age (years)", "ap_hi": "Systolic BP (mmHg)", "flagged": "Flagged"},
    title="Random 5,000-patient sample"
)
fig3.update_layout(height=380, margin=dict(t=40, b=10))
st.plotly_chart(fig3, use_container_width=True)
