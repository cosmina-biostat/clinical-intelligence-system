import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATA_PATH  = BASE / "training_data.csv"
MODEL_DIR  = BASE / "models"
REPORT_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────────
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
TARGET_COL  = "label"
LABEL_ORDER = ["Clean", "Review", "Block"]   # severity ascending


# ── Load & prepare ────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records from {DATA_PATH.name}")
    print(f"  Label counts: {df[TARGET_COL].value_counts().to_dict()}\n")

    X = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(y_raw)

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return X, y, le


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models():
    """
    Returns dict {name: model}. Models needing feature scaling are wrapped
    in a Pipeline so cross-validation scales inside each fold (no leakage).
    """
    models = {}

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",   # Block is the minority class
            random_state=42,
        )),
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,                  # capped: prevents pure memorisation
        min_samples_leaf=5,            # forces generalising splits
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,            # L2 regularisation
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        print("XGBoost not installed -- skipping. pip install xgboost")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        print("LightGBM not installed -- skipping. pip install lightgbm")

    return models


# ── Train & evaluate ──────────────────────────────────────────────────────────
def train_and_evaluate(X, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}\n")

    models  = get_models()
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    block_idx = list(le.classes_).index("Block")
    report_lines = []
    sep = "=" * 60

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        # Train metrics (overfitting gap)
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1  = f1_score(y_train, y_train_pred, average="macro")

        # Test metrics
        y_pred   = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1  = f1_score(y_test, y_pred, average="macro")
        gap      = train_f1 - test_f1

        # Patient-safety headline: Block recall
        # (how many true Blocks did we catch?)
        block_recall = recall_score(
            y_test, y_pred, labels=[block_idx], average=None
        )[0]

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(
            y_test, y_pred, target_names=le.classes_, digits=3
        )

        # 5-fold CV -- Pipeline handles scaling inside folds (no leakage)
        cv_scores = cross_val_score(
            model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        results[name] = {
            "model":        model,
            "train_acc":    train_acc,
            "train_f1":     train_f1,
            "test_acc":     test_acc,
            "test_f1":      test_f1,
            "gap":          gap,
            "block_recall": block_recall,
            "cv_mean":      cv_scores.mean(),
            "cv_std":       cv_scores.std(),
            "cm":           cm,
            "cr":           cr,
        }

        print(f"  Train F1     : {train_f1:.4f}   (Acc {train_acc:.4f})")
        print(f"  Test  F1     : {test_f1:.4f}   (Acc {test_acc:.4f})")
        print(f"  Overfit gap  : {gap:+.4f}")
        print(f"  Block recall : {block_recall:.4f}   <- patient safety")
        print(f"  CV F1        : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  Confusion Matrix (rows=true, cols=pred):")
        print(f"    Labels: {list(le.classes_)}")
        for i, row in enumerate(cm):
            print(f"    {le.classes_[i]:<8} {row}")
        print()

        model_path = MODEL_DIR / f"review_classifier_{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved -> {model_path.name}\n")

        report_lines += [
            sep,
            f"  Model: {name}",
            sep,
            f"  Train F1     : {train_f1:.4f}   (Acc {train_acc:.4f})",
            f"  Test  F1     : {test_f1:.4f}   (Acc {test_acc:.4f})",
            f"  Overfit gap  : {gap:+.4f}",
            f"  Block recall : {block_recall:.4f}",
            f"  CV F1        : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
            "",
            "  Classification Report:",
            cr,
            "  Confusion Matrix (rows=true, cols=pred):",
            f"  Labels: {list(le.classes_)}",
        ]
        for i, row in enumerate(cm):
            report_lines.append(f"  {le.classes_[i]:<8} {row}")
        report_lines.append("")

    return results, report_lines


# ── Pick best model (by CV F1, not test F1) ──────────────────────────────────
def save_best(results, report_lines):
    """
    Selection criterion: highest CV F1 mean.
    Tie-break: lower CV std (more stable), then higher Block recall.
    Rationale: a single test split can flatter a model by luck; CV averages
    over 5 splits. Block recall matters because a missed Block is the most
    dangerous clinical error.
    """
    best_name = max(
        results,
        key=lambda n: (
            results[n]["cv_mean"],
            -results[n]["cv_std"],
            results[n]["block_recall"],
        ),
    )
    best = results[best_name]

    print("=" * 60)
    print(f"  Best model (by CV F1): {best_name}")
    print(f"    CV F1        : {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")
    print(f"    Test F1      : {best['test_f1']:.4f}")
    print(f"    Block recall : {best['block_recall']:.4f}")
    print("=" * 60 + "\n")

    with open(MODEL_DIR / "review_classifier_best.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    print("Best model saved -> review_classifier_best.pkl")

    comparison = {
        name: {
            "train_f1":     round(r["train_f1"], 4),
            "test_f1":      round(r["test_f1"], 4),
            "overfit_gap":  round(r["gap"], 4),
            "block_recall": round(r["block_recall"], 4),
            "cv_f1_mean":   round(r["cv_mean"], 4),
            "cv_f1_std":    round(r["cv_std"], 4),
        }
        for name, r in results.items()
    }
    comparison["_best"] = best_name
    comparison["_selection_criterion"] = "cv_f1_mean (tie: cv_std, block_recall)"
    with open(MODEL_DIR / "classifier_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("Comparison saved -> classifier_comparison.json")

    summary = [
        "=" * 60,
        "  ClinOrigin AI -- Review Classifier Training Report",
        "=" * 60,
        f"  Best model (by CV F1) : {best_name}",
        f"  CV F1                 : {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}",
        f"  Test F1               : {best['test_f1']:.4f}",
        f"  Block recall          : {best['block_recall']:.4f}",
        "",
    ] + report_lines

    report_path = REPORT_DIR / "classifier_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(summary))
    print("Full report saved -> reports/classifier_report.txt\n")

    return best_name


# ── Safety threshold tuning (Block recall first) ─────────────────────────────
def tune_block_threshold(results, best_name, X, y, le, target_recall=0.95):
    """
    Clinical rationale: a Block record slipping through as Clean/Review-pass
    is the most dangerous error (bad data enters the database unseen).
    Instead of using argmax over class probabilities, we apply a
    safety-first decision rule:

        if P(Block) >= t  ->  Block
        else              ->  argmax over remaining classes

    The threshold t is tuned on a VALIDATION split (never the test set)
    as the lowest value that achieves the target Block recall.
    Lower t  ->  more records flagged as Block  ->  recall up, precision down.
    In a clinical pipeline this trade-off is correct: false alarms cost
    review time, missed Blocks cost data integrity.
    """
    print("=" * 60)
    print(f"  Safety threshold tuning (target Block recall >= {target_recall})")
    print("=" * 60)

    block_idx = list(le.classes_).index("Block")

    # Fresh split: train / validation / test (60 / 20 / 20)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    # Re-fit the best model class on the reduced train split
    model = get_models()[best_name]
    model.fit(X_tr, y_tr)

    # --- Tune on validation ---
    proba_val = model.predict_proba(X_val)
    is_block_val = (y_val == block_idx)

    best_t = 0.5
    for t in np.arange(0.50, 0.04, -0.01):
        pred_block = proba_val[:, block_idx] >= t
        recall = pred_block[is_block_val].mean()
        if recall >= target_recall:
            best_t = round(float(t), 2)
            break
    else:
        best_t = 0.05  # floor: even max sensitivity misses some

    # --- Evaluate on test: argmax vs thresholded rule ---
    proba_test = model.predict_proba(X_test)

    pred_argmax = proba_test.argmax(axis=1)

    pred_safety = pred_argmax.copy()
    pred_safety[proba_test[:, block_idx] >= best_t] = block_idx

    def block_metrics(y_true, y_pred):
        rec = recall_score(y_true, y_pred, labels=[block_idx], average=None)[0]
        mask = (y_pred == block_idx)
        prec = (y_true[mask] == block_idx).mean() if mask.any() else 0.0
        f1m = f1_score(y_true, y_pred, average="macro")
        return rec, prec, f1m

    rec_a, prec_a, f1_a = block_metrics(y_test, pred_argmax)
    rec_s, prec_s, f1_s = block_metrics(y_test, pred_safety)

    print(f"  Tuned threshold      : P(Block) >= {best_t}")
    print(f"                          {'argmax':>8}   {'safety':>8}")
    print(f"  Block recall          {rec_a:>8.4f}   {rec_s:>8.4f}")
    print(f"  Block precision       {prec_a:>8.4f}   {prec_s:>8.4f}")
    print(f"  F1-macro (overall)    {f1_a:>8.4f}   {f1_s:>8.4f}")
    print()
    print("  Reading: recall rises (fewer missed Blocks), precision falls")
    print("  (more false alarms routed to human review). Clinically correct.")
    print("=" * 60 + "\n")

    # Persist threshold for inference
    with open(MODEL_DIR / "block_threshold.json", "w") as f:
        json.dump({
            "model": best_name,
            "block_threshold": best_t,
            "target_recall": target_recall,
            "decision_rule": "Block if P(Block) >= threshold else argmax",
            "validation_tuned": True,
        }, f, indent=2)
    print("Threshold saved -> block_threshold.json")

    return best_t


# ── Feature importance (tree models) ─────────────────────────────────────────
def print_feature_importance(results):
    print("Feature Importances (tree-based models, normalised):")
    for name, r in results.items():
        model = r["model"]
        # unwrap Pipeline if needed
        est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
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
    print("  ClinOrigin AI -- Review Classifier Training")
    print("=" * 60 + "\n")

    X, y, le = load_data()
    results, report_lines = train_and_evaluate(X, y, le)
    best_name = save_best(results, report_lines)
    tune_block_threshold(results, best_name, X, y, le, target_recall=0.95)
    print_feature_importance(results)

    print("\nTraining complete.")
    print(f"  Models saved to : {MODEL_DIR}")
    print(f"  Report saved to : {REPORT_DIR / 'classifier_report.txt'}")