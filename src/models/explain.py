import shap
import matplotlib.pyplot as plt
import numpy as np
from src.utils.config import REPORTS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

def get_shap_values(model, X, model_type: str = "tree"):
    """Return SHAP values. Use model_type='linear' for LogReg."""
    if model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.TreeExplainer(model)
    return explainer, explainer(X)

def shap_summary_plot(model, X, feature_names: list, dataset_name: str, clf_name: str):
    explainer, shap_vals = get_shap_values(model, X)
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_vals, show=False)
    fig.savefig(
        REPORTS_DIR / "figures" / f"shap_{clf_name}_{dataset_name}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    log.info(f"SHAP summary saved → reports/figures/shap_{clf_name}_{dataset_name}.png")

def explain_single_patient(model, patient_array, feature_names: list):
    """Return a dict of {feature: shap_value} for one patient. Used by the dashboard."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(patient_array)
    return dict(zip(feature_names, shap_vals.values[0]))
