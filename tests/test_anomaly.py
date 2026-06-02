import numpy as np
from sklearn.ensemble import IsolationForest
from src.anomaly.detector import score_patient

def test_score_patient_structure():
    model = IsolationForest(random_state=42)
    X = np.random.randn(100, 5)
    model.fit(X)
    result = score_patient(model, X[:1])
    assert "anomaly_score" in result
    assert "flagged" in result
    assert isinstance(result["flagged"], bool)

def test_anomaly_flag():
    model = IsolationForest(contamination=0.1, random_state=42)
    X_normal = np.random.randn(100, 3)
    model.fit(X_normal)
    # An extreme outlier should be flagged
    outlier = np.array([[999, 999, 999]])
    result = score_patient(model, outlier)
    assert result["flagged"] is True
