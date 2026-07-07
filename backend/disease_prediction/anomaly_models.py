from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float             # from decision_function; lower = more anomalous
    top_contributors: list[tuple[str, float]]  # only populated if is_anomaly
    model_key: str
    model_name: str


@dataclass
class AnomalyModelCard:
    key: str
    display_name: str
    indication_patterns: list[str]
    feature_order: list[str]
    detect_fn: Callable[[pd.DataFrame], AnomalyResult]


class AnomalyModelRegistry:
    def __init__(self):
        self._cards: dict[str, AnomalyModelCard] = {}

    def register(self, card: AnomalyModelCard):
        self._cards[card.key] = card

    def available(self) -> list[AnomalyModelCard]:
        return list(self._cards.values())

    def match_indication(self, indication: Optional[str]) -> Optional[AnomalyModelCard]:
        if not indication:
            return None
        text = indication.lower()
        for card in self._cards.values():
            for pattern in card.indication_patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return card
        return None

    def get(self, key: str) -> Optional[AnomalyModelCard]:
        return self._cards.get(key)

    def detect(self, key: str, features: dict) -> AnomalyResult:
        card = self.get(key)
        if card is None:
            raise KeyError(f"No anomaly model registered for key '{key}'")
        missing = [f for f in card.feature_order if f not in features]
        if missing:
            raise ValueError(f"Missing required features for '{card.display_name}': {missing}")
        row = pd.DataFrame([[features[f] for f in card.feature_order]],
                           columns=card.feature_order)
        return card.detect_fn(row)


# ── Cardio anomaly model (Cosmina) ────────────────────────────────────────────
_CARDIO_ANOMALY_FEATURES = ["age_years", "gender", "height", "weight", "bmi",
                            "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]


def _make_cardio_anomaly_detect_fn():
    model = joblib.load(MODEL_DIR / "iso_forest_cardio.pkl")

    # SHAP explainer is built once, lazily, only if a flagged patient needs it
    _explainer = {"obj": None}
    def _get_explainer():
        if _explainer["obj"] is None:
            import shap
            _explainer["obj"] = shap.TreeExplainer(model)
        return _explainer["obj"]

    def detect(row: pd.DataFrame) -> AnomalyResult:
        pred = int(model.predict(row)[0])           # -1 = anomaly, 1 = normal
        score = float(model.decision_function(row)[0])
        is_anomaly = pred == -1

        contributors: list[tuple[str, float]] = []
        if is_anomaly:
            explainer = _get_explainer()
            shap_values = explainer.shap_values(row)[0]
            pairs = list(zip(_CARDIO_ANOMALY_FEATURES, [float(v) for v in shap_values]))
            pairs.sort(key=lambda x: x[1])  # most negative first = biggest push toward anomaly
            contributors = pairs[:3]  # top 3 drivers, matches decision_drivers pattern elsewhere

        return AnomalyResult(
            is_anomaly=is_anomaly, anomaly_score=round(score, 4),
            top_contributors=[(f, round(v, 4)) for f, v in contributors],
            model_key="cardio", model_name="Cardiovascular Profile Anomaly (IsolationForest)",
        )
    return detect


def build_anomaly_registry() -> AnomalyModelRegistry:
    registry = AnomalyModelRegistry()
    registry.register(AnomalyModelCard(
        key="cardio",
        display_name="Cardiovascular Profile Anomaly (IsolationForest)",
        indication_patterns=[
            r"cardio", r"cardiovascular", r"heart failure", r"coronary",
            r"hypertension", r"myocardial infarction",
        ],
        feature_order=_CARDIO_ANOMALY_FEATURES,
        detect_fn=_make_cardio_anomaly_detect_fn(),
    ))
    return registry


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reg = build_anomaly_registry()
    print("Registered anomaly models:", [c.key for c in reg.available()])

    normal = {"age_years": 54, "gender": 1, "height": 165, "weight": 72, "bmi": 26.4,
             "ap_hi": 130, "ap_lo": 85, "cholesterol": 2, "gluc": 1,
             "smoke": 0, "alco": 0, "active": 1}
    r1 = reg.detect("cardio", normal)
    print(f"\nNormal patient: {r1}")

    outlier = {"age_years": 22, "gender": 1, "height": 165, "weight": 145, "bmi": 53.2,
              "ap_hi": 195, "ap_lo": 118, "cholesterol": 3, "gluc": 3,
              "smoke": 1, "alco": 1, "active": 0}
    r2 = reg.detect("cardio", outlier)
    print(f"\nOutlier patient: {r2}")
