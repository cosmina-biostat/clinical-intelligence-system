import pandas as pd
from src.utils.config import DATASETS
from src.utils.logger import get_logger

log = get_logger(__name__)

def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset by name. name must be one of: cardio, ms, melanoma, synthetic."""
    path = DATASETS.get(name)
    if path is None:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Run: python -m src.data.loader --download to fetch from Kaggle."
        )
    log.info(f"Loading {name} dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Quick summary stats for EDA."""
    return {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "class_balance": df.iloc[:, -1].value_counts(normalize=True).to_dict()
    }
