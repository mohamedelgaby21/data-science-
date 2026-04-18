import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# Load model and feature names
# =========================
model_path = "models/random_forest_model.pkl"
features_path = "models/feature_names.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

if not os.path.exists(features_path):
    st.error(f"Feature names file not found: {features_path}")
    st.stop()

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

st.set_page_config(page_title="Student Risk Prediction", layout="centered")

st.title("🎓 Student Dropout Risk Prediction")
st.write("Enter student data below to predict whether the student is Safe or At-Risk.")

# =========================
# User input fields
# =========================
gender = st.selectbox("Gender", ["M", "F"])

highest_education = st.selectbox(
    "Highest Education",
    [
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification"
    ]
)

imd_band = st.selectbox(
    "IMD Band",
    [
        "0-10%",
        "10-20",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%"
    ]
)

age_band = st.selectbox("Age Band", ["0-35", "35-55", "55<="])
disability = st.selectbox("Disability", ["N", "Y"])

num_of_prev_attempts = st.number_input("Number of Previous Attempts", min_value=0, step=1)
studied_credits = st.number_input("Studied Credits", min_value=0, step=1)
total_early_clicks = st.number_input("Total Early Clicks", min_value=0, step=1)
GDP_per_capita = st.number_input("GDP per Capita", min_value=0.0, step=0.01)

# =========================
# Build input dataframe
# =========================
input_data = pd.DataFrame([{
    "gender": gender,
    "highest_education": highest_education,
    "imd_band": imd_band,
    "age_band": age_band,
    "disability": disability,
    "num_of_prev_attempts": num_of_prev_attempts,
    "studied_credits": studied_credits,
    "total_early_clicks": total_early_clicks,
    "GDP_per_capita": GDP_per_capita
}])

# =========================
# Apply same encoding structure
# =========================
input_encoded = pd.get_dummies(input_data)

for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[feature_names]

# =========================
# Prediction
# =========================
if st.button("Predict Risk"):
    prediction = model.predict(input_encoded)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 This student is likely At-Risk.")
    else:
        st.success("✅ This student is likely Safe.")