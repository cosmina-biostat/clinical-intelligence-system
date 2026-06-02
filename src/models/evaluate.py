import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, RocCurveDisplay)
from src.utils.config import REPORTS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

def compute_metrics(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }

def comparison_table(results: dict) -> pd.DataFrame:
    """results = {dataset: {clf: {metric: value}}}"""
    rows = []
    for dataset, clfs in results.items():
        for clf, metrics in clfs.items():
            rows.append({"dataset": dataset, "classifier": clf, **metrics})
    df = pd.DataFrame(rows).sort_values(["dataset", "roc_auc"], ascending=[True, False])
    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    log.info("Model comparison saved to reports/model_comparison.csv")
    return df

def plot_roc_curves(models: dict, X_test, y_test, dataset_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title(f"ROC curves — {dataset_name}")
    fig.savefig(REPORTS_DIR / "figures" / f"roc_{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"ROC curve saved → reports/figures/roc_{dataset_name}.png")
