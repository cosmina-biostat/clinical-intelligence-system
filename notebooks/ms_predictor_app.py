import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MS Conversion Predictor",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# Train & cache model (runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv")
    X = df.drop(columns=["Unnamed: 0", "group"])
    y = (df["group"] == 2).astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, imputer, scaler, X.columns.tolist()

model, imputer, scaler, feature_names = load_model()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🧠 MS Conversion Predictor")
st.markdown("Predict whether a **Clinically Isolated Syndrome (CIS)** patient will convert to **Multiple Sclerosis (MS)**.")
st.divider()

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    age = st.number_input("Age (years)", min_value=15, max_value=80, value=34)
    schooling = st.number_input("Schooling (years)", min_value=0, max_value=25, value=12)
    breastfeeding = st.selectbox("Breastfeeding", options=[1, 2, 3],
                                  format_func=lambda x: {1: "Yes", 2: "No", 3: "Unknown"}.get(x))
    varicella = st.selectbox("Varicella (Chickenpox)", options=[1, 2, 3],
                              format_func=lambda x: {1: "Positive", 2: "Negative", 3: "Unknown"}.get(x))
    initial_symptom = st.selectbox("Initial Symptom", options=list(range(1, 16)),
                                    format_func=lambda x: {
                                        1: "1 - Visual", 2: "2 - Sensory", 3: "3 - Motor",
                                        4: "4 - Other", 5: "5 - Visual+Sensory", 6: "6 - Visual+Motor",
                                        7: "7 - Visual+Other", 8: "8 - Sensory+Motor", 9: "9 - Sensory+Other",
                                        10: "10 - Motor+Other", 11: "11 - Visual+Sensory+Motor",
                                        12: "12 - Visual+Sensory+Other", 13: "13 - Visual+Motor+Other",
                                        14: "14 - Sensory+Motor+Other", 15: "15 - All"
                                    }.get(x, str(x)))
    mono_poly = st.selectbox("Mono/Polysymptomatic", options=[1, 2, 3],
                              format_func=lambda x: {1: "Monosymptomatic", 2: "Polysymptomatic", 3: "Unknown"}.get(x))
    oligoclonal = st.selectbox("Oligoclonal Bands", options=[0, 1, 2],
                                format_func=lambda x: {0: "Negative", 1: "Positive", 2: "Unknown"}.get(x))
    initial_edss = st.number_input("Initial EDSS", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

with col2:
    st.markdown("**Evoked Potentials**")
    llssep = st.selectbox("LLSSEP (Lower Limb)", options=[0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
    ulssep = st.selectbox("ULSSEP (Upper Limb)", options=[0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
    vep    = st.selectbox("VEP (Visual)", options=[0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
    baep   = st.selectbox("BAEP (Brainstem)", options=[0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")

    st.markdown("**MRI Findings**")
    periventricular_mri  = st.selectbox("Periventricular MRI", options=[0, 1], format_func=lambda x: "No Lesion" if x == 0 else "Lesion Present")
    cortical_mri         = st.selectbox("Cortical MRI", options=[0, 1], format_func=lambda x: "No Lesion" if x == 0 else "Lesion Present")
    infratentorial_mri   = st.selectbox("Infratentorial MRI", options=[0, 1], format_func=lambda x: "No Lesion" if x == 0 else "Lesion Present")
    spinal_cord_mri      = st.selectbox("Spinal Cord MRI", options=[0, 1], format_func=lambda x: "No Lesion" if x == 0 else "Lesion Present")
    final_edss           = st.number_input("Final EDSS", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Schooling": schooling,
        "Breastfeeding": breastfeeding,
        "Varicella": varicella,
        "Initial_Symptom": initial_symptom,
        "Mono_or_Polysymptomatic": mono_poly,
        "Oligoclonal_Bands": oligoclonal,
        "LLSSEP": llssep,
        "ULSSEP": ulssep,
        "VEP": vep,
        "BAEP": baep,
        "Periventricular_MRI": periventricular_mri,
        "Cortical_MRI": cortical_mri,
        "Infratentorial_MRI": infratentorial_mri,
        "Spinal_Cord_MRI": spinal_cord_mri,
        "Initial_EDSS": initial_edss,
        "Final_EDSS": final_edss,
    }])

    X_imp = imputer.transform(input_data)
    X_sc  = scaler.transform(X_imp)

    prediction   = model.predict(X_sc)[0]
    probability  = model.predict_proba(X_sc)[0]
    cdms_prob    = probability[0]
    npms_prob    = probability[1]

    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.error(f"⚠️ **CDMS — Converted to Multiple Sclerosis**")
        st.markdown(f"The model predicts this patient is likely to **convert to MS**.")
    else:
        st.success(f"✅ **NPMS — No Conversion to Multiple Sclerosis**")
        st.markdown(f"The model predicts this patient is **unlikely to convert to MS**.")

    # Confidence bars
    st.markdown("**Confidence Scores:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("CDMS Probability", f"{cdms_prob:.1%}")
        st.progress(float(cdms_prob))
    with col_b:
        st.metric("NPMS Probability", f"{npms_prob:.1%}")
        st.progress(float(npms_prob))

    st.caption("⚠️ This tool is for research purposes only. Always consult a qualified neurologist for clinical decisions.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:grey; font-size:12px;'>MS Conversion Predictor · Powered by Gradient Boosting · For Research Use Only</div>",
    unsafe_allow_html=True
)