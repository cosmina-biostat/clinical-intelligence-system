from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"


# ── Model card: one per registered disease model ──────────────────────────────
@dataclass
class ModelCard:
    key: str                          # internal id, e.g. "cardio"
    display_name: str                 # shown to the user
    indication_patterns: list[str]    # regex patterns matched against the
                                       # protocol's `indication` field
    feature_order: list[str]          # exact columns the model expects, in order
    predict_fn: Callable[[pd.DataFrame], "PredictionResult"]
    notes: str = ""


@dataclass
class PredictionResult:
    label: str                 # e.g. "High risk" / "Low risk"
    probability: float         # P(positive class), 0..1
    model_key: str
    model_name: str


class DiseaseModelRegistry:
    def __init__(self):
        self._cards: dict[str, ModelCard] = {}

    def register(self, card: ModelCard):
        self._cards[card.key] = card

    def available(self) -> list[ModelCard]:
        return list(self._cards.values())

    def match_indication(self, indication: Optional[str]) -> Optional[ModelCard]:
        """
        Find the best-matching model for a free-text indication string.
        Returns None if nothing matches -- the caller must then ask the
        user to pick manually (never silently guess).
        """
        if not indication:
            return None
        text = indication.lower()
        for card in self._cards.values():
            for pattern in card.indication_patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return card
        return None

    def get(self, key: str) -> Optional[ModelCard]:
        return self._cards.get(key)

    def predict(self, key: str, features: dict) -> PredictionResult:
        card = self.get(key)
        if card is None:
            raise KeyError(f"No model registered for key '{key}'")
        missing = [f for f in card.feature_order if f not in features]
        if missing:
            raise ValueError(
                f"Missing required features for '{card.display_name}': {missing}"
            )
        row = pd.DataFrame([[features[f] for f in card.feature_order]],
                           columns=card.feature_order)
        return card.predict_fn(row)


# ── Cardio model (Cosmina) ────────────────────────────────────────────────────
# NOTE: we do NOT call pipeline.predict_proba() on the full sklearn Pipeline.
# The saved pipeline was pickled with sklearn 1.2.2; unpickling its
# ColumnTransformer's "passthrough" step under newer sklearn (1.9+) raises
# "'str' object has no attribute 'transform'" -- a known cross-version
# incompatibility in how "passthrough" sentinels are resolved at transform
# time. Rather than pin an old sklearn just for this, we reproduce the exact
# preprocessing manually (verified against the pipeline's own StandardScaler
# and the model's expected column order) and call the inner LGBMClassifier
# directly. This is also more auditable for a clinical system.

_CARDIO_NUMERIC = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
_CARDIO_PASSTHROUGH = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
_CARDIO_FEATURES = _CARDIO_NUMERIC + _CARDIO_PASSTHROUGH  # public feature order


def _load_cardio_parts():
    """Load the trained LGBM model + its StandardScaler, bypassing the
    version-fragile full Pipeline object."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected cross-version pickle warnings
        pipeline = joblib.load(MODEL_DIR / "lgbm_pipeline_cardio.pkl")
        scaler_path = MODEL_DIR / "lgbm_scaler_cardio.pkl"
        scaler = (joblib.load(scaler_path) if scaler_path.exists()
                 else pipeline.named_steps["prep"].named_transformers_["num"])
    model = pipeline.named_steps["model"]
    return model, scaler


def _make_cardio_predict_fn():
    model, scaler = _load_cardio_parts()
    model_feature_order = list(model.feature_name_)  # ["Column_0", ... "Column_11"]

    def predict(row: pd.DataFrame) -> PredictionResult:
        import warnings
        # row columns are in _CARDIO_FEATURES order (public names)
        num_vals = row[_CARDIO_NUMERIC].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scaler expects a plain array here, by design
            num_scaled = scaler.transform(num_vals)[0]
        pass_vals = row[_CARDIO_PASSTHROUGH].to_numpy(dtype=float)[0]

        final_row = list(num_scaled) + list(pass_vals)
        X = pd.DataFrame([final_row], columns=model_feature_order)
        proba = float(model.predict_proba(X)[0, 1])

        label = "Elevated cardiovascular risk" if proba >= 0.5 else "Low cardiovascular risk"
        return PredictionResult(label=label, probability=round(proba, 4),
                                model_key="cardio", model_name="Cardiovascular Risk (LightGBM)")
    return predict


# ── MS model -- PLACEHOLDER ────────────────────────────────────────────────────
# The current trained_model.pkl is corrupted (truncated at byte 732) AND has a
# target-leakage issue (Initial_EDSS is only populated for one class). Do NOT
# wire it in until it has been retrained without the leaking column and saved
# cleanly with joblib.dump(). This placeholder keeps the registry honest:
# asking for "ms" raises a clear NotImplementedError instead of silently
# returning a wrong or leaked prediction.
_MS_FEATURES = [
    "Gender", "Age", "Schooling", "Breastfeeding", "Varicella",
    "Initial_Symptom", "Mono_or_Polysymptomatic", "Oligoclonal_Bands",
    "LLSSEP", "ULSSEP", "VEP", "BAEP",
    "Periventricular_MRI", "Cortical_MRI", "Infratentorial_MRI", "Spinal_Cord_MRI",
]  # NOTE: excludes Initial_EDSS and Final_EDSS deliberately (leakage) -- verify
   # the exact retrained feature list against the new training script.


def _ms_predict_placeholder(row: pd.DataFrame) -> PredictionResult:
    raise NotImplementedError(
        "MS model is not yet wired in: the saved file is corrupted and the "
        "training data has a known leakage issue (Initial_EDSS). Retrain "
        "without the leaking column, save with joblib.dump(), then call "
        "register_ms_model() with the new file."
    )


def register_ms_model(registry: "DiseaseModelRegistry", model_path: Path,
                      feature_order: Optional[list[str]] = None):
    """
    Call this once a clean MS model exists, e.g.:
        register_ms_model(registry, Path("models/ms_model_v2.pkl"))
    """
    pipeline = joblib.load(model_path)
    features = feature_order or _MS_FEATURES

    def predict(row: pd.DataFrame) -> PredictionResult:
        proba = float(pipeline.predict_proba(row)[0, 1])
        label = "Likely MS conversion" if proba >= 0.5 else "Unlikely MS conversion"
        return PredictionResult(label=label, probability=round(proba, 4),
                                model_key="ms", model_name="MS Conversion Risk")

    registry.register(ModelCard(
        key="ms", display_name="MS Conversion Risk",
        indication_patterns=[r"multiple sclerosis", r"\bms\b"],
        feature_order=features, predict_fn=predict,
        notes="Registered from a retrained, leakage-free model.",
    ))


# ── Build the default registry ────────────────────────────────────────────────
# ── Mapping layer: extracted patient data -> model feature contract ──────────
# The extractor produces human-readable fields with names Claude assigned
# while parsing the protocol (e.g. "Age", "Sex": "Female") -- NOT necessarily
# the exact keys/encoding a given model was trained on (e.g. "age_years",
# "gender": 1/2). This layer bridges that gap per model, so the Structured
# table can run predictions automatically from already-extracted data
# instead of requiring manual re-entry.
#
# Design: never guess silently. If a value can't be confidently resolved,
# the field is reported as missing/unmappable rather than defaulted -- the
# caller (UI) must show "insufficient data", not a fabricated prediction.

def _first_present(data: dict, *keys):
    """Case-insensitive lookup across several possible field-name spellings."""
    lower_map = {k.lower(): v for k, v in data.items()}
    for key in keys:
        if key.lower() in lower_map and lower_map[key.lower()] not in (None, ""):
            return lower_map[key.lower()]
    return None


def _to_number(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # strip common units the extractor might keep, e.g. "72 kg", "130 mmHg"
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _sex_to_gender_code(val):
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("1", "female", "f", "woman"):
        return 1.0
    if s in ("2", "male", "m", "man"):
        return 2.0
    return None


def _category_to_code(val, low_terms, high_terms, vhigh_terms):
    """Map a descriptive lab category (or an already-numeric 1/2/3 code) to
    the model's expected 1/2/3 encoding."""
    if val is None:
        return None
    num = _to_number(val)
    if num in (1.0, 2.0, 3.0):
        return num
    s = str(val).strip().lower()
    if any(t in s for t in vhigh_terms):
        return 3.0
    if any(t in s for t in high_terms):
        return 2.0
    if any(t in s for t in low_terms):
        return 1.0
    return None


def _to_binary(val):
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("1", "yes", "true", "y"):
        return 1.0
    if s in ("0", "no", "false", "n"):
        return 0.0
    return None


def resolve_cardio_features(data: dict) -> tuple[Optional[dict], list[str]]:
    """
    Attempt to build the 12 cardio features from an extracted patient record.
    Returns (features_or_None, missing_field_names). If ANY required feature
    cannot be resolved, features is None -- the caller must not predict on
    a partial/guessed input.
    """
    missing = []

    age = _to_number(_first_present(data, "age", "age_years"))
    if age is None: missing.append("age")

    gender = _sex_to_gender_code(_first_present(data, "sex", "gender"))
    if gender is None: missing.append("sex")

    height = _to_number(_first_present(data, "height", "height_cm"))
    if height is None: missing.append("height")

    weight = _to_number(_first_present(data, "weight", "weight_kg"))
    if weight is None: missing.append("weight")

    bmi = _to_number(_first_present(data, "bmi"))
    if bmi is None and height and weight:
        bmi = round(weight / ((height / 100) ** 2), 1)
    if bmi is None: missing.append("bmi")

    ap_hi = _to_number(_first_present(data, "ap_hi", "systolic_bp", "sbp", "blood_pressure_systolic"))
    if ap_hi is None: missing.append("ap_hi (systolic BP)")

    ap_lo = _to_number(_first_present(data, "ap_lo", "diastolic_bp", "dbp", "blood_pressure_diastolic"))
    if ap_lo is None: missing.append("ap_lo (diastolic BP)")

    cholesterol = _category_to_code(
        _first_present(data, "cholesterol", "cholesterol_category"),
        low_terms=["normal"], high_terms=["above normal"], vhigh_terms=["well above"])
    if cholesterol is None: missing.append("cholesterol")

    gluc = _category_to_code(
        _first_present(data, "gluc", "glucose", "glucose_category"),
        low_terms=["normal"], high_terms=["above normal"], vhigh_terms=["well above"])
    if gluc is None: missing.append("glucose")

    smoke = _to_binary(_first_present(data, "smoke", "smoking", "current_smoker"))
    if smoke is None: missing.append("smoking status")

    alco = _to_binary(_first_present(data, "alco", "alcohol"))
    if alco is None: missing.append("alcohol use")

    active = _to_binary(_first_present(data, "active", "physical_activity"))
    if active is None: missing.append("physical activity")

    if missing:
        return None, missing

    return {
        "age_years": age, "height": height, "weight": weight, "bmi": bmi,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "gender": gender,
        "cholesterol": cholesterol, "gluc": gluc,
        "smoke": smoke, "alco": alco, "active": active,
    }, []


# Registry of per-model resolvers (only cardio implemented so far)
FEATURE_RESOLVERS: dict[str, Callable[[dict], tuple[Optional[dict], list[str]]]] = {
    "cardio": resolve_cardio_features,
}


def resolve_features(model_key: str, data: dict) -> tuple[Optional[dict], list[str]]:
    resolver = FEATURE_RESOLVERS.get(model_key)
    if resolver is None:
        return None, [f"no field-mapping defined yet for model '{model_key}'"]
    return resolver(data)


def build_registry() -> DiseaseModelRegistry:
    registry = DiseaseModelRegistry()

    registry.register(ModelCard(
        key="cardio",
        display_name="Cardiovascular Risk (LightGBM)",
        indication_patterns=[
            r"cardio", r"cardiovascular", r"heart failure", r"coronary",
            r"hypertension", r"myocardial infarction",
        ],
        feature_order=_CARDIO_FEATURES,
        predict_fn=_make_cardio_predict_fn(),
        notes="Trained on cardio_train.csv (70k records), no leakage found.",
    ))

    registry.register(ModelCard(
        key="ms",
        display_name="MS Conversion Risk (not yet available)",
        indication_patterns=[r"multiple sclerosis", r"\bms\b"],
        feature_order=_MS_FEATURES,
        predict_fn=_ms_predict_placeholder,
        notes="PLACEHOLDER -- corrupted file + Initial_EDSS leakage. See register_ms_model().",
    ))

    return registry


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reg = build_registry()

    print("Registered models:")
    for card in reg.available():
        print(f"  {card.key:8s} -> {card.display_name}")

    print("\nIndication matching:")
    for ind in ["Cardiovascular Disease", "Multiple Sclerosis (MS)",
               "Advanced Solid Malignancies", None]:
        match = reg.match_indication(ind)
        print(f"  {str(ind):35s} -> {match.key if match else 'NO MATCH (manual pick needed)'}")

    print("\nCardio prediction (sample patient):")
    sample = {
        "age_years": 54, "gender": 1, "height": 165, "weight": 72, "bmi": 26.4,
        "ap_hi": 130, "ap_lo": 85, "cholesterol": 2, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1,
    }
    result = reg.predict("cardio", sample)
    print(f"  {result}")

    print("\nMS prediction (should raise NotImplementedError):")
    try:
        reg.predict("ms", {})
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")