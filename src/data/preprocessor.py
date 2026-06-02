import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from src.utils.config import RANDOM_STATE, TEST_SIZE, MODELS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

def build_pipeline() -> Pipeline:
    """Impute missing values then scale. Saved alongside each model."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

def split_and_balance(X: pd.DataFrame, y: pd.Series, dataset_name: str):
    """Train/test split + SMOTE on train set only (never on test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"Pre-SMOTE class balance: {y_train.value_counts().to_dict()}")

    pipe = build_pipeline()
    X_train_scaled = pipe.fit_transform(X_train)
    X_test_scaled  = pipe.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    log.info(f"Post-SMOTE class balance: {pd.Series(y_train_bal).value_counts().to_dict()}")

    # Save the fitted pipeline so the dashboard can use the same scaler
    joblib.dump(pipe, MODELS_DIR / f"pipeline_{dataset_name}.pkl")
    joblib.dump(list(X.columns), MODELS_DIR / f"features_{dataset_name}.pkl")
    log.info(f"Pipeline saved to models/saved/pipeline_{dataset_name}.pkl")

    return X_train_bal, X_test_scaled, y_train_bal, y_test
