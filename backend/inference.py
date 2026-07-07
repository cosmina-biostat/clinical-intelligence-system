import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ── Feature order: MUST match engineer_features() output keys, in order ───────
FEATURE_ORDER = [
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

MODEL_DIR = Path(__file__).parent / "models"


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
class _Models:
    def __init__(self):
        self.classifier = None
        self.label_encoder = None
        self.quality = None
        self.block_threshold = 0.5
        self.block_idx = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        # Review classifier
        with open(MODEL_DIR / "review_classifier_best.pkl", "rb") as f:
            self.classifier = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        # Safety threshold (optional -- falls back to argmax if absent)
        try:
            with open(MODEL_DIR / "block_threshold.json") as f:
                cfg = json.load(f)
            self.block_threshold = cfg.get("block_threshold", 0.5)
        except FileNotFoundError:
            self.block_threshold = None  # no safety rule -> pure argmax
        # Quality regressor (optional)
        try:
            with open(MODEL_DIR / "quality_regressor_best.pkl", "rb") as f:
                self.quality = pickle.load(f)
        except FileNotFoundError:
            self.quality = None
        # Cache the Block class index
        classes = list(self.label_encoder.classes_)
        self.block_idx = classes.index("Block") if "Block" in classes else None
        self.loaded = True


_M = _Models()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _vectorize(features: dict) -> np.ndarray:
    """Turn the feature dict into the model's expected row vector."""
    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"features missing keys: {missing}")
    return np.array([[float(features[k]) for k in FEATURE_ORDER]])


# ── Public API ────────────────────────────────────────────────────────────────

# Class ranking used only to compare severity for the guardrail below
# (Clean < Review < Block). Independent of the label encoder's internal order.
_SEVERITY_RANK = {"Clean": 0, "Review": 1, "Block": 2}


def classify_review(features: dict) -> dict:
    """
    Review classification with TWO layers of safety on top of the ML model's
    raw prediction, applied in order:

      1. Probabilistic Block escalation: Block if P(Block) >= block_threshold.
      2. Deterministic critical-field guardrail: if a required field is
         missing (critical_fields_missing > 0), the verdict can never be
         'Clean' -- minimum 'Review' -- REGARDLESS of what the model
         predicts. This exists because the model's training signal for
         extraction_confidence is a constant 1.0 in production (see
         feature_engineering.py TODO), unlike its variable, informative role
         during training -- so the model alone cannot be fully trusted to
         catch every case where a required field is missing. A deterministic
         rule closes that gap rather than relying solely on ML recall.

    Returns:
      {label, probabilities: {Clean, Review, Block}, block_probability,
       safety_rule_applied, guardrail_applied}
    """
    _M.load()
    X = _vectorize(features)

    proba = _M.classifier.predict_proba(X)[0]
    classes = list(_M.label_encoder.classes_)
    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    # Default: argmax
    pred_idx = int(np.argmax(proba))
    safety_applied = False

    # Layer 1: probabilistic Block escalation (existing)
    if _M.block_threshold is not None and _M.block_idx is not None:
        if proba[_M.block_idx] >= _M.block_threshold and pred_idx != _M.block_idx:
            pred_idx = _M.block_idx
            safety_applied = True

    label = str(_M.label_encoder.inverse_transform([pred_idx])[0])

    # Layer 2: deterministic critical-field guardrail (new)
    guardrail_applied = False
    critical_missing = features.get("critical_fields_missing", 0) or 0
    if critical_missing > 0 and _SEVERITY_RANK.get(label, 0) < _SEVERITY_RANK["Review"]:
        label = "Review"
        guardrail_applied = True

    block_prob = float(proba[_M.block_idx]) if _M.block_idx is not None else None

    return {
        "label": label,
        "probabilities": prob_map,
        "block_probability": round(block_prob, 4) if block_prob is not None else None,
        "safety_rule_applied": safety_applied,
        "guardrail_applied": guardrail_applied,
        "block_threshold": _M.block_threshold,
    }


def predict_quality(features: dict) -> Optional[float]:
    """Quality score in [0,1], or None if the model isn't available."""
    _M.load()
    if _M.quality is None:
        return None
    X = _vectorize(features)
    score = float(_M.quality.predict(X)[0])
    return round(min(max(score, 0.0), 1.0), 4)


def assess(features: dict) -> dict:
    """
    Full assessment used by the API: review label + quality + a small
    audit trail explaining WHY (for the dashboard / regulatory traceability).
    """
    review = classify_review(features)
    quality = predict_quality(features)

    # Audit trail: surface the features that drove the decision
    drivers = {
        "completeness_score": features.get("completeness_score"),
        "critical_fields_missing": features.get("critical_fields_missing"),
        "high_severity_flags": features.get("high_severity_flags"),
        "total_flags": features.get("total_flags"),
        "guardrail_applied": review.get("guardrail_applied", False),
    }

    return {
        "review_status": review["label"],
        "review_detail": review,
        "quality_score": quality,
        "decision_drivers": drivers,
    }


def models_status() -> dict:
    """Lightweight status for the /health endpoint."""
    try:
        _M.load()
        return {
            "classifier": _M.classifier is not None,
            "quality_regressor": _M.quality is not None,
            "safety_threshold": _M.block_threshold,
            "labels": list(_M.label_encoder.classes_),
        }
    except Exception as e:
        return {"error": str(e)}