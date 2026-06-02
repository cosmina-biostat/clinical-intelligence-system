import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.models.train import CLASSIFIER_MAP
from src.models.evaluate import compute_metrics

@pytest.fixture
def dummy_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    split = 160
    return X[:split], X[split:], y[:split], y[split:]

def test_all_classifiers_train(dummy_data):
    X_train, X_test, y_train, y_test = dummy_data
    for name, clf in CLASSIFIER_MAP.items():
        clf.fit(X_train, y_train)
        metrics = compute_metrics(clf, X_test, y_test)
        assert metrics["roc_auc"] > 0.5, f"{name} AUC below 0.5"
        assert "f1" in metrics

def test_metrics_keys(dummy_data):
    X_train, X_test, y_train, y_test = dummy_data
    clf = CLASSIFIER_MAP["random_forest"]
    clf.fit(X_train, y_train)
    m = compute_metrics(clf, X_test, y_test)
    assert set(m.keys()) == {"roc_auc", "f1", "precision", "recall"}
