# app.py
import streamlit as st
import joblib
import pandas as pd
import os

# Title
st.title("🩺 Blood Pressure Prediction App")
st.markdown("Predict systolic BP from age, gender, BMI, and ethnicity.")

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "bp_model_improved.pkl")
model = joblib.load(model_path)
# Load model (relative path)
# model = joblib.load("models/bp_model_improved.pkl")

# Input form
st.sidebar.header("Patient Input")
age = st.sidebar.slider("Age (years)", 18, 100, 40)
gender = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI (kg/m²)", 15.0, 50.0, 25.0)
ethnicity = st.sidebar.selectbox("Ethnicity", [
    "Mexican American", "Other Hispanic", "Non-Hispanic White", 
    "Non-Hispanic Black", "Other Race - Including Multi-Racial"
])

# Encode inputs
gender_code = 1 if gender == "Male" else 2
ethnicity_map = {
    "Mexican American": 1, "Other Hispanic": 2, "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4, "Other Race - Including Multi-Racial": 5
}
eth_code = ethnicity_map[ethnicity]

# Create input DataFrame
input_data = pd.DataFrame({
    "RIDAGEYR_scaled": [(age - 45.5) / 18.5], # Mean=45.5, std=18.5 (from training)
    "RIAGENDR": [gender_code],
    "BMXBMI_scaled": [(bmi - 28.0) / 6.0],  # Mean=28.0, std=6.0
    "eth_2.0": [1 if eth_code == 2 else 0],
    "eth_3.0": [1 if eth_code == 3 else 0],
    "eth_4.0": [1 if eth_code == 4 else 0],
    "eth_5.0": [1 if eth_code == 5 else 0]
})

# Predict
prediction_scaled = model.predict(input_data)[0]
prediction_raw = prediction_scaled * 15.0 + 120.0 # Reverse scale: mean=120, std=15

# Display result
st.subheader("Prediction")
st.metric("Predicted Systolic BP (mmHg)", f"{prediction_raw:.1f}")

# Add info
st.markdown("---")
st.caption("Model trained on NHANES August 2021-August 2023 data. Not for medical diagnosis.")