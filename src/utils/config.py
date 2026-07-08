import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_SYN    = ROOT / "data" / "synthetic"
MODELS_DIR  = ROOT / "models" / "saved"
REPORTS_DIR = ROOT / "reports"

# ── API keys ───────────────────────────────────────────────────────────────
# Reads from .env locally; falls back to Streamlit Cloud secrets when deployed
def _get_api_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return key

ANTHROPIC_API_KEY = _get_api_key()
KAGGLE_USERNAME   = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY        = os.getenv("KAGGLE_KEY", "")

# ── Dataset file names ─────────────────────────────────────────────────────
DATASETS = {
    "cardio":   DATA_RAW / "heart_disease.csv",
    "ms":       DATA_RAW / "ms_cis.csv",
    "melanoma": DATA_RAW / "melanoma.csv",
    "synthetic": DATA_SYN / "synthetic_patients.csv",
}

# ── Model settings ─────────────────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
VAL_SIZE         = 0.15
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.65"))
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "3"))

# ── Classifier names used throughout the project ───────────────────────────
CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

# ── Feature columns per dataset ───────────────────────────────────────────
CARDIO_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                   "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

MS_FEATURES = ["age", "gender", "initial_symptoms", "lesion_count",
               "spinal_lesions", "brain_lesions", "oligoclonal_bands"]

MELANOMA_FEATURES = ["age_approx", "sex", "anatom_site", "tbp_lv_areaMM2",
                     "tbp_lv_color_std_mean", "tbp_lv_deltaLBnorm", "tbp_lv_H"]
