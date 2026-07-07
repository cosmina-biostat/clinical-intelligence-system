"""
Batch Screening — upload patient letters → Claude extracts fields → LightGBM scores.
Uses the CARDIO-SHIELD schema to guide extraction.
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pdfplumber
from pathlib import Path
from anthropic import Anthropic

st.set_page_config(page_title="Batch Screening", page_icon="📋", layout="wide")

MODELS_DIR  = Path("models/saved")
SCHEMA_PATH = Path("backend/schema/CARDIO_SHIELD_SAP_V2_schema.json")

NUMERIC = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CODED   = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
FEATURES = NUMERIC + CODED


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    pkl = MODELS_DIR / "lgbm_model_cardio.pkl"
    return joblib.load(pkl) if pkl.exists() else None

@st.cache_resource(show_spinner=False)
def load_scaler():
    pkl = MODELS_DIR / "lgbm_scaler_cardio.pkl"
    return joblib.load(pkl) if pkl.exists() else None

@st.cache_resource(show_spinner=False)
def load_iso():
    pkl = MODELS_DIR / "iso_forest_cardio.pkl"
    return joblib.load(pkl) if pkl.exists() else None

@st.cache_data(show_spinner=False)
def load_schema():
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text())
    return {}

model  = load_model()
scaler = load_scaler()
iso    = load_iso()
schema = load_schema()

if model is None or scaler is None:
    st.error("Prediction model not found. Run the training script first.")
    st.stop()


# ── Claude extraction ─────────────────────────────────────────────────────────
def extract_patient_fields(text: str, api_key: str) -> dict:
    """Send letter text to Claude Haiku and extract structured patient fields."""
    client = Anthropic(api_key=api_key)

    prompt = """You are a clinical data extractor. Extract patient fields from this letter.

Return ONLY a JSON object with these fields (use null if not found):
{
  "patient_id": "string",
  "age": integer (years),
  "sex": "Female" or "Male",
  "height": float (cm),
  "weight": float (kg),
  "bmi": float (kg/m2),
  "systolic_bp": integer (mmHg),
  "diastolic_bp": integer (mmHg),
  "cholesterol": 1, 2, or 3  (1=normal, 2=above normal, 3=well above normal),
  "glucose": 1, 2, or 3  (1=normal, 2=above normal, 3=well above normal),
  "smoking": 0 or 1,
  "alcohol": 0 or 1,
  "physical_activity": 0 or 1
}

Rules:
- Convert decimal commas to points (13,8 → 13.8)
- If BMI not stated, calculate from height and weight
- Respond ONLY with the JSON object, no other text"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": f"{prompt}\n\nLetter:\n{text}"}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def score_patient(fields: dict) -> dict:
    """Map extracted fields → model features → risk score + anomaly flag."""
    try:
        bmi = fields.get("bmi") or (
            round(fields["weight"] / (fields["height"] / 100) ** 2, 1)
            if fields.get("height") and fields.get("weight") else None
        )
        row = {
            "age_years":   fields.get("age"),
            "gender":      1 if fields.get("sex") == "Female" else 2,
            "height":      fields.get("height"),
            "weight":      fields.get("weight"),
            "bmi":         bmi,
            "ap_hi":       fields.get("systolic_bp"),
            "ap_lo":       fields.get("diastolic_bp"),
            "cholesterol": fields.get("cholesterol", 1),
            "gluc":        fields.get("glucose", 1),
            "smoke":       fields.get("smoking", 0),
            "alco":        fields.get("alcohol", 0),
            "active":      fields.get("physical_activity", 1),
        }

        # Check all required numeric fields are present
        missing = [k for k in FEATURES if row.get(k) is None]
        if missing:
            return {"error": f"Missing: {', '.join(missing)}"}

        df = pd.DataFrame([row])
        num_scaled = scaler.transform(df[NUMERIC])
        X = np.hstack([num_scaled, df[CODED].values])

        prob  = float(model.predict_proba(X)[:, 1][0])
        band  = "Low" if prob < 0.33 else ("Moderate" if prob < 0.66 else "High")

        anomaly_score = None
        anomaly_flag  = None
        if iso is not None:
            anomaly_score = float(iso.decision_function(df[FEATURES])[0])
            anomaly_flag  = "⚠️ Anomalous" if iso.predict(df[FEATURES])[0] == -1 else "✅ Normal"

        return {
            "risk_pct":     f"{prob * 100:.1f}%",
            "risk_band":    band,
            "anomaly_score": f"{anomaly_score:.4f}" if anomaly_score is not None else "—",
            "anomaly_flag":  anomaly_flag or "—",
            "error":        None,
        }
    except Exception as e:
        return {"error": str(e)}


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📋 Batch Patient Screening")
st.caption(
    "Upload patient letters (PDFs) → Claude extracts clinical fields → "
    "LightGBM scores cardiovascular risk · CARDIO-SHIELD protocol"
)

# API key
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
    api_key = ""

if not api_key:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

if not api_key:
    st.warning("ANTHROPIC_API_KEY not set. Add it to your .env file or Streamlit secrets.")
    st.stop()

st.divider()

# Upload
uploaded_files = st.file_uploader(
    "Upload one or more patient letters (PDF)",
    type="pdf",
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload patient PDFs above to begin. The demo letters are in `data/raw/cardio_shield_demo_letters/`.")
    st.stop()

if st.button(f"Extract & Score {len(uploaded_files)} patient(s)", type="primary"):

    results = []
    progress = st.progress(0)
    status   = st.empty()

    for i, f in enumerate(uploaded_files):
        status.text(f"Processing {f.name} …")

        # Extract text from PDF
        try:
            with pdfplumber.open(f) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            results.append({"Patient ID": f.name, "Error": str(e)})
            continue

        # Claude extraction
        try:
            fields = extract_patient_fields(text, api_key)
        except Exception as e:
            results.append({"Patient ID": f.name, "Error": f"Extraction failed: {e}"})
            progress.progress((i + 1) / len(uploaded_files))
            continue

        # Score
        scores = score_patient(fields)

        if scores.get("error"):
            results.append({
                "Patient ID": fields.get("patient_id", f.name),
                "Age": fields.get("age"), "Sex": fields.get("sex"),
                "Error": scores["error"],
            })
        else:
            results.append({
                "Patient ID":    fields.get("patient_id", f.name),
                "Age":           fields.get("age"),
                "Sex":           fields.get("sex"),
                "Height (cm)":   fields.get("height"),
                "Weight (kg)":   fields.get("weight"),
                "BMI":           fields.get("bmi"),
                "Systolic BP":   fields.get("systolic_bp"),
                "Diastolic BP":  fields.get("diastolic_bp"),
                "CVD Risk":      scores["risk_pct"],
                "Risk Level":    scores["risk_band"],
                "Anomaly Score": scores["anomaly_score"],
                "Profile":       scores["anomaly_flag"],
                "Error":         "—",
            })

        progress.progress((i + 1) / len(uploaded_files))

    status.empty()
    progress.empty()

    st.success(f"Processed {len(results)} patient(s)")
    st.divider()

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

    # Highlight high-risk patients
    high_risk = df_results[df_results.get("Risk Level", pd.Series()) == "High"]
    if not high_risk.empty:
        st.warning(f"⚠️ {len(high_risk)} patient(s) flagged as **High Risk** — review recommended.")
        st.dataframe(high_risk[["Patient ID", "Age", "Sex", "CVD Risk", "Profile"]], use_container_width=True)

    # Download
    csv = df_results.to_csv(index=False)
    st.download_button(
        "Download results as CSV",
        data=csv,
        file_name="batch_screening_results.csv",
        mime="text/csv",
    )
