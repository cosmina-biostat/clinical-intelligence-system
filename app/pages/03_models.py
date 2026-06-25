"""
Model Comparison — loads results from reports/model_comparison.csv.
No raw data or retraining required.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Model Comparison")
st.caption("4 classifiers · cardiovascular dataset · 68,595 patients · 20% held-out test set")

csv_path = Path("reports/model_comparison.csv")

if not csv_path.exists():
    st.error("results file not found at reports/model_comparison.csv")
    st.stop()

df = pd.read_csv(csv_path)

# Normalise column names — handle both lower and title case
df.columns = [c.strip() for c in df.columns]
col_map = {c: c.replace("roc_auc","ROC-AUC").replace("f1","F1")
              .replace("precision","Precision").replace("recall","Recall")
              .replace("accuracy","Accuracy").replace("classifier","Classifier")
              .replace("dataset","Dataset")
           for c in df.columns}
df = df.rename(columns=col_map)

metrics = [c for c in ["ROC-AUC","F1","Precision","Recall","Accuracy"] if c in df.columns]
metric  = st.selectbox("Metric to display", metrics)

# ── Bar chart ─────────────────────────────────────────────────────────────────
fig = px.bar(
    df.sort_values(metric, ascending=False),
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
display_cols = ["Classifier"] + metrics
st.dataframe(
    df[display_cols].sort_values("ROC-AUC", ascending=False).set_index("Classifier"),
    use_container_width=True
)

# ── Radar chart ───────────────────────────────────────────────────────────────
st.subheader("Radar comparison")
fig2 = px.line_polar(
    df.melt(id_vars="Classifier", value_vars=metrics,
            var_name="Metric", value_name="Score"),
    r="Score", theta="Metric", color="Classifier",
    line_close=True,
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig2.update_layout(polar=dict(radialaxis=dict(range=[0.5, 1.0])), height=480)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("MLflow experiment tracking"):
    st.info(
        "To view full experiment logs locally, run `mlflow ui` in the project root "
        "and open http://localhost:5000. Each classifier run is logged automatically "
        "when training via `src/models/train.py`."
    )
