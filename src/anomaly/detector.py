import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.utils.config import MODELS_DIR, ANOMALY_THRESHOLD, RANDOM_STATE
from src.utils.logger import get_logger

log = get_logger(__name__)

def train_anomaly_detectors(X_train, dataset_name: str):
    iso = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
    iso.fit(X_train)
    joblib.dump(iso, MODELS_DIR / f"iso_forest_{dataset_name}.pkl")
    log.info(f"Isolation Forest saved for {dataset_name}")
    return iso

def score_patient(iso_model, patient_array) -> dict:
    score = iso_model.decision_function(patient_array)[0]
    is_anomaly = score < ANOMALY_THRESHOLD
    return {"anomaly_score": round(float(score), 3), "flagged": bool(is_anomaly)}
