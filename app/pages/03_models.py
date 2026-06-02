import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils.config import REPORTS_DIR

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison — All Classifiers × All Datasets")

csv_path = REPORTS_DIR / "model_comparison.csv"
if not csv_path.exists():
    st.warning("No results yet. Run notebook 04 (train_models) first.")
else:
    df = pd.read_csv(csv_path)
    metric = st.selectbox("Metric", ["roc_auc", "f1", "precision", "recall"])
    dataset_filter = st.multiselect("Datasets", df["dataset"].unique().tolist(),
                                    default=df["dataset"].unique().tolist())
    df_filt = df[df["dataset"].isin(dataset_filter)]

    fig = px.bar(df_filt, x="classifier", y=metric, color="dataset", barmode="group",
                 title=f"{metric.upper()} by classifier and dataset",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_filt.sort_values(metric, ascending=False), use_container_width=True)
