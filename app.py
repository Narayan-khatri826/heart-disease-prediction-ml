import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("heart_disease_scaler.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's demographic, medical, and lifestyle "
    "information to estimate the probability of heart disease."
)

st.info(
    "⚠️ This application provides an ML-based prediction "
    "for research purposes and is not a medical diagnosis."
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("👤 Patient Information")

age = st.number_input(
    "Age",
    min_value=20,
    max_value=80,
    value=50,
    step=1
)

sex = st.selectbox(
    "Sex",
    options=[1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

race_ethnicity = st.selectbox(
    "Race / Ethnicity Code",
    options=[1, 2, 3, 4, 5, 6]
)

education = st.selectbox(
    "Education Level Code",
    options=[1, 2, 3, 4, 5]
)

poverty_income_ratio = st.number_input(
    "Poverty Income Ratio",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
)


# =========================================================
# MEDICAL INFORMATION
# =========================================================

st.header("🏥 Medical Information")


taking_bp_meds = st.selectbox(
    "Taking Blood Pressure Medication?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

taking_cholesterol_meds = st.selectbox(
    "Taking Cholesterol Medication?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

taking_insulin = st.selectbox(
    "Taking Insulin?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

taking_diabetes_pills = st.selectbox(
    "Taking Diabetes Pills?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

told_prediabetes = st.selectbox(
    "Previously Told You Have Prediabetes?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

told_stroke = st.selectbox(
    "History of Stroke?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

family_history_heart_attack = st.selectbox(
    "Family History of Heart Attack?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

diabetes = st.selectbox(
    "Diabetes?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

hypertension = st.selectbox(
    "Hypertension?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

high_cholesterol = st.selectbox(
    "High Cholesterol?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# =========================================================
# LIFESTYLE INFORMATION
# =========================================================

st.header("🏃 Lifestyle Information")


smoking_ever = st.selectbox(
    "Have You Ever Smoked?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

smoking_current = st.selectbox(
    "Currently Smoking?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

physically_active = st.selectbox(
    "Physically Active?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Heart Disease Probability",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create patient dataframe
    patient_data = pd.DataFrame([{
        "taking_bp_meds": taking_bp_meds,
        "taking_cholesterol_meds": taking_cholesterol_meds,
        "age": age,
        "sex": sex,
        "race_ethnicity": race_ethnicity,
        "education": education,
        "poverty_income_ratio": poverty_income_ratio,
        "taking_insulin": taking_insulin,
        "taking_diabetes_pills": taking_diabetes_pills,
        "told_prediabetes": told_prediabetes,
        "told_stroke": told_stroke,
        "family_history_heart_attack": family_history_heart_attack,
        "diabetes": diabetes,
        "smoking_ever": smoking_ever,
        "smoking_current": smoking_current,
        "physically_active": physically_active,
        "hypertension": hypertension,
        "high_cholesterol": high_cholesterol
    }])


    # Scale patient data
    patient_scaled = scaler.transform(patient_data)


    # Prediction probability
    probability = model.predict_proba(patient_scaled)[0][1]


    # Default threshold = 0.5
    prediction = int(probability >= 0.5)


    # =====================================================
    # RESULT
    # =====================================================

    st.header("📊 Prediction Result")

    st.metric(
        "Estimated Heart Disease Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))


    if prediction == 1:

        st.warning(
            "⚠️ Higher Predicted Risk"
        )

        st.write(
            "The model predicts a higher probability of "
            "heart disease based on the entered information."
        )

    else:

        st.success(
            "✅ Lower Predicted Risk"
        )

        st.write(
            "The model predicts a lower probability of "
            "heart disease based on the entered information."
        )


    st.caption(
        "This prediction is generated by a machine learning "
        "model and should not be used as a medical diagnosis."
    )