import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.utils.config import RANDOM_STATE, MODELS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

CLASSIFIER_MAP = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "xgboost":             XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                         eval_metric="logloss", verbosity=0),
    "lightgbm":            LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                          verbose=-1),
}

def train_all(X_train, y_train, X_test, y_test, dataset_name: str) -> dict:
    """Train all 4 classifiers, log to MLflow, save .pkl files. Returns results dict."""
    results = {}
    mlflow.set_experiment(f"clinical-{dataset_name}")

    for clf_name, clf in CLASSIFIER_MAP.items():
        log.info(f"Training {clf_name} on {dataset_name}...")
        with mlflow.start_run(run_name=f"{clf_name}_{dataset_name}"):
            clf.fit(X_train, y_train)

            from src.models.evaluate import compute_metrics
            metrics = compute_metrics(clf, X_test, y_test)

            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(clf, clf_name)

            save_path = MODELS_DIR / f"{clf_name}_{dataset_name}.pkl"
            joblib.dump(clf, save_path)
            log.info(f"Saved → {save_path} | AUC: {metrics['roc_auc']:.3f}")
            results[clf_name] = metrics

    return results
