import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATA_PATH  = BASE / "training_data_quality.csv"
MODEL_DIR  = BASE / "models"
REPORT_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "completeness_score",
    "missing_required_count",
    "missing_optional_count",
    "out_of_range_count",
    "plausibility_issues",
    "total_flags",
    "critical_fields_missing",
    "high_severity_flags",
    "extraction_confidence",
    "field_count_total",
]
TARGET_COL = "quality_score"


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records from {DATA_PATH.name}")
    print(f"  quality_score: min={df[TARGET_COL].min():.3f} "
          f"max={df[TARGET_COL].max():.3f} mean={df[TARGET_COL].mean():.3f}\n")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


# ── Models ────────────────────────────────────────────────────────────────────
def get_models():
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        print("XGBoost not installed -- skipping. pip install xgboost")
    return models


# ── Band-wise error (domain-relevant) ─────────────────────────────────────────
def mae_per_band(y_true, y_pred):
    """MAE separately for Low/Mid/High true scores."""
    bands = {
        "Low  (0-0.33)":  (y_true < 0.33),
        "Mid  (0.33-0.66)": (y_true >= 0.33) & (y_true < 0.66),
        "High (0.66-1.0)": (y_true >= 0.66),
    }
    return {
        name: mean_absolute_error(y_true[mask], y_pred[mask]) if mask.any() else float("nan")
        for name, mask in bands.items()
    }


# ── Train & evaluate ──────────────────────────────────────────────────────────
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}\n")

    models  = get_models()
    results = {}
    cv      = KFold(n_splits=5, shuffle=True, random_state=42)
    report_lines = []
    sep = "=" * 60

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Train metrics (overfitting gap)
        y_train_pred = model.predict(X_train)
        train_r2  = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)

        # Test metrics
        y_pred   = np.clip(model.predict(X_test), 0.0, 1.0)
        test_r2  = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        gap = train_r2 - test_r2

        band_mae = mae_per_band(y_test, y_pred)

        # 5-fold CV (selection criterion)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)

        results[name] = {
            "model":     model,
            "train_r2":  train_r2,
            "test_r2":   test_r2,
            "gap":       gap,
            "train_mae": train_mae,
            "test_mae":  test_mae,
            "test_rmse": test_rmse,
            "band_mae":  band_mae,
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
        }

        print(f"  Train R^2    : {train_r2:.4f}   (MAE {train_mae:.4f})")
        print(f"  Test  R^2    : {test_r2:.4f}   (MAE {test_mae:.4f}, RMSE {test_rmse:.4f})")
        print(f"  Overfit gap  : {gap:+.4f}")
        print(f"  CV R^2       : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  MAE per band:")
        for band, val in band_mae.items():
            print(f"    {band:<18} {val:.4f}")
        print()

        model_path = MODEL_DIR / f"quality_regressor_{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved -> {model_path.name}\n")

        report_lines += [
            sep,
            f"  Model: {name}",
            sep,
            f"  Train R^2    : {train_r2:.4f}   (MAE {train_mae:.4f})",
            f"  Test  R^2    : {test_r2:.4f}   (MAE {test_mae:.4f}, RMSE {test_rmse:.4f})",
            f"  Overfit gap  : {gap:+.4f}",
            f"  CV R^2       : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
            "  MAE per band:",
        ]
        for band, val in band_mae.items():
            report_lines.append(f"    {band:<18} {val:.4f}")
        report_lines.append("")

    return results, report_lines


# ── Pick best (by CV R^2) ─────────────────────────────────────────────────────
def save_best(results, report_lines):
    best_name = max(results, key=lambda n: (results[n]["cv_mean"], -results[n]["cv_std"]))
    best = results[best_name]

    print("=" * 60)
    print(f"  Best model (by CV R^2): {best_name}")
    print(f"    CV R^2   : {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")
    print(f"    Test R^2 : {best['test_r2']:.4f}")
    print(f"    Test MAE : {best['test_mae']:.4f}")
    print("=" * 60 + "\n")

    with open(MODEL_DIR / "quality_regressor_best.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    print("Best model saved -> quality_regressor_best.pkl")

    comparison = {
        name: {
            "train_r2":   round(r["train_r2"], 4),
            "test_r2":    round(r["test_r2"], 4),
            "overfit_gap": round(r["gap"], 4),
            "test_mae":   round(r["test_mae"], 4),
            "test_rmse":  round(r["test_rmse"], 4),
            "cv_r2_mean": round(r["cv_mean"], 4),
            "cv_r2_std":  round(r["cv_std"], 4),
        }
        for name, r in results.items()
    }
    comparison["_best"] = best_name
    comparison["_selection_criterion"] = "cv_r2_mean (tie: cv_std)"
    with open(MODEL_DIR / "quality_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("Comparison saved -> quality_comparison.json")

    summary = [
        "=" * 60,
        "  ClinOrigin AI -- Quality Score Regressor Training Report",
        "=" * 60,
        f"  Best model (by CV R^2) : {best_name}",
        f"  CV R^2                 : {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}",
        f"  Test R^2               : {best['test_r2']:.4f}",
        f"  Test MAE               : {best['test_mae']:.4f}",
        "",
    ] + report_lines

    with open(REPORT_DIR / "quality_report.txt", "w") as f:
        f.write("\n".join(summary))
    print("Full report saved -> reports/quality_report.txt\n")

    return best_name


# ── Feature importance ────────────────────────────────────────────────────────
def print_feature_importance(results):
    print("Feature Importances (tree-based models, normalised):")
    for name, r in results.items():
        model = r["model"]
        est = model.named_steps["reg"] if isinstance(model, Pipeline) else model
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
            total = importances.sum()
            importances = importances / total if total > 0 else importances
            pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
            print(f"\n  {name}:")
            for feat, imp in pairs:
                bar = "#" * int(imp * 40)
                print(f"  {feat:<30} {imp:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ClinOrigin AI -- Quality Score Regressor Training")
    print("=" * 60 + "\n")

    X, y = load_data()
    results, report_lines = train_and_evaluate(X, y)
    best_name = save_best(results, report_lines)
    print_feature_importance(results)

    print("\nTraining complete.")
    print(f"  Models saved to : {MODEL_DIR}")
    print(f"  Report saved to : {REPORT_DIR / 'quality_report.txt'}")