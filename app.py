# app.py
import streamlit as st
import joblib
import pandas as pd

# Title
st.title("🩺 Blood Pressure Prediction App")
st.markdown("Predict systolic BP from age, gender, and BMI.")

# Load model
model = joblib.load("models/bp_model.pkl")

# Input form
st.sidebar.header("Patient Input")
age = st.sidebar.slider("Age (years)", 18, 100, 40)
gender = st.sidebar.selectbox("Sex", ["Male", "Female"])

# for improved model:
# bmi = st.sidebar.number_input("BMI (kg/m2)", 15.0, 50.0, 25.0)

# Encode gender
gender_code = 1 if gender == "Male" else 2

# Create input DataFrame
input_data = pd.DataFrame({
    "RIDAGEYR_scaled": [(age - 45.5) / 18.5], # Mean=45.5, std=18.5 (from training)
    "RIAGENDR": [gender_code]
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