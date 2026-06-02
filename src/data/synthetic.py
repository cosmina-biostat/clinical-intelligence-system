import pandas as pd
import numpy as np
from src.utils.config import DATA_SYN, RANDOM_STATE
from src.utils.logger import get_logger

log = get_logger(__name__)

def generate_synthetic_patients(n: int = 300, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":       rng.integers(30, 80, n),
        "sex":       rng.integers(0, 2, n),
        "chol":      rng.normal(220, 50, n).clip(120, 500).round(1),
        "trestbps":  rng.normal(130, 20, n).clip(90, 200).round(1),
        "thalach":   rng.normal(145, 25, n).clip(80, 200).round(1),
        "fbs":       rng.integers(0, 2, n),
        "exang":     rng.integers(0, 2, n),
        "oldpeak":   rng.exponential(1.2, n).clip(0, 6).round(2),
        "cp":        rng.integers(0, 4, n),
        "restecg":   rng.integers(0, 3, n),
        "slope":     rng.integers(0, 3, n),
        "ca":        rng.integers(0, 4, n),
        "thal":      rng.integers(1, 4, n),
    })
    # Simple rule-based label for realism
    risk_score = (df["chol"] > 240).astype(int) + (df["trestbps"] > 140).astype(int) + \
                 (df["age"] > 60).astype(int) + (df["exang"] == 1).astype(int)
    df["target"] = (risk_score >= 2).astype(int)
    path = DATA_SYN / "synthetic_patients.csv"
    df.to_csv(path, index=False)
    log.info(f"Generated {n} synthetic patients → {path}")
    return df
