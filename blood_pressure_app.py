# ------------- IMPORTS -------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


# ------------- PAGE CONFIG -------------

st.set_page_config(
    page_title="BP Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ------------- LOAD FILES -------------

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load model
pipeline_path = os.path.join(MODELS_DIR, "bp_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

# Load confidence interval data 
sigma_residuals_path = os.path.join(MODELS_DIR, "sigma_residuals.pkl")
sigma_residuals = joblib.load(sigma_residuals_path)

# Load Mahalanobis data
mean_vector_path = os.path.join(MODELS_DIR, "mean_vector.pkl")
mean_vector = joblib.load(mean_vector_path)

cov_matrix_inv_path = os.path.join(MODELS_DIR, "cov_matrix_inv.pkl")
cov_matrix_inv = joblib.load(cov_matrix_inv_path)

# Load training info
training_info_path = os.path.join(MODELS_DIR, "training_info.pkl")
training_info = joblib.load(training_info_path)
n_train = training_info["n"]
p_features = training_info["p"]


# ------------- STREAMLIT SETUP & INPUT -------------

# Title
st.title("Blood Pressure Prediction App")
st.markdown("Predict systolic BP from age, gender, BMI, and ethnicity.")

# Author info
st.markdown("---")
st.markdown(
    "Developed by **Ezra Guerrero González** | "
    "[GitHub](https://github.com/EzraGuerrero) | "
    "[LinkedIn](https://www.linkedin.com/in/ezraguerrero/)",
    help="View source code and documentation"
)

# Input form (Top of page)
st.markdown("## Subject Information")
st.markdown("Enter demographic and anthropometric data below:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age, (years)", 18, 100, 40)
    gender = st.selectbox("Sex", ["Male", "Female"])

with col2:
    bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 25.0, step=0.5)
    ethnicity = st.selectbox("Ethnicity", [
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


# ------------- PREDICTION -------------

# Wrap into Predict Button
st.markdown("---")

if st.button("Predict"):
    
    # =========== Prepare input data (input DataFrame) ===========

    input_data = pd.DataFrame({
        "RIDAGEYR": [age],          # Raw age (e.g., 50)
        "BMXBMI": [bmi],            # Raw BMI (e.g., 28.5)
        "RIAGENDR": [gender_code],  # 1 or 2
        "eth_2.0": [1 if eth_code == 2 else 0],
        "eth_3.0": [1 if eth_code == 3 else 0],
        "eth_4.0": [1 if eth_code == 4 else 0],
        "eth_5.0": [1 if eth_code == 5 else 0]
    })

    
    # =========== Prediction ===========
    
    prediction_raw = pipeline.predict(input_data)[0]


    # =========== Confidence Interval ===========
    
    # Calculate t-critical value for 95% confidence
    df = n_train - p_features -1 # degrees of freedom
    t_crit = stats.t.ppf(0.975, df)

    # Margin of error (in raw BP units)
    margin = t_crit * sigma_residuals

    # Confidence interval
    ci_lower = prediction_raw - margin
    ci_upper = prediction_raw + margin

    
    # =========== Mahalanobis Distance ===========
    
    # Get the scaler from the pipeline to transform input consistently
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
    
    # Transform the raw input using the SAME scaler the pipeline uses
    age_scaled = scaler.transform([[age, bmi]])[0][0]  # Scaled age
    bmi_scaled = scaler.transform([[age, bmi]])[0][1]  # Scaled BMI

    # Convert input to numpy array (same order as training)
    input_array = np.array([
        age_scaled,    # Scaled using pipeline's scaler
        bmi_scaled,    # Scaled using pipeline's scaler
        gender_code,
        1 if eth_code == 2 else 0,
        1 if eth_code == 3 else 0,
        1 if eth_code == 4 else 0,
        1 if eth_code == 5 else 0
    ])
    
    # Compute Mahalanobis distance
    diff = input_array - mean_vector
    mahalanobis_dist = np.sqrt(diff @ cov_matrix_inv @ diff.T)

    # Set threshold (95% confidence)
    threshold = np.sqrt(chi2.ppf(0.95, df=p_features))

    # Check if OOD
    is_ood = mahalanobis_dist > threshold

    
    # =========== Display Results ===========

    # 1) Prediction
    st.subheader("Prediction")
    st.metric("Predicted Systolic BP (mmHg)", f"{prediction_raw:.1f}")
    
    # 2) Confidence Interval
    st.subheader("Prediction Confidence")
    st.markdown(f"**95% Confidence Interval**: {ci_lower:.1f} – {ci_upper:.1f} mmHg")
    st.markdown(f"*(±{margin:.1f} mmHg)*")
    
    # 3) Health Status
    if prediction_raw < 120:
        status = "Normal"
        advice = "Good news! Your predicted systolic BP is below 120 mm Hg — within the normal range. Maintain healthy habits for long-term wellness."
        color = "#2E7D32"
    elif 120 <= prediction_raw < 130:
        status = "Elevated"
        advice = "Your predicted systolic BP is 120–129 mm Hg. This may signal early-stage risk; consider regular monitoring and lifestyle adjustments."
        color = "#B8860B"
    elif 130 <= prediction_raw < 140:
        status = "Stage 1 Hypertension"
        advice = "Your predicted systolic BP is 130–139 mm Hg. A healthcare provider should evaluate your risk factors."
        color = "#E65100"
    elif 140 <= prediction_raw < 180:
        status = "Stage 2 Hypertension"
        advice = "Your predicted systolic BP is 140 mm Hg or higher. Please schedule a medical checkup to discuss next steps."
        color = "#C62828"
    else:  # >= 180
        status = "Hypertensive Crisis"
        advice = "Your predicted systolic BP is above 180 mm Hg. This requires urgent medical evaluation — do not delay care!"
        color = "#6A1B9A"

    st.subheader("Health Status")
    st.markdown(f"<span style='font-size: 1.5em; color: {color};'>{status}</span>", unsafe_allow_html=True)
    st.info(advice)
    
    # 4) OOD Warning
    st.subheader("Input Reliability")
    if is_ood:
        st.warning(f"**Out-of-Distribution Input Detected**")
        st.markdown(f"*This input is unusual compared to the training data (Mahalanobis distance: {mahalanobis_dist:.2f}, threshold: {threshold:.2f}). The prediction may be less reliable.*")
    else:
        st.success(f"Input is within normal range (Mahalanobis distance: {mahalanobis_dist:.2f}, threshold: {threshold:.2f})")

    
    # 5) Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This tool is for educational and informational purposes only. "
        "It is not a medical device and should not be used for clinical diagnosis. "
        "Always consult a qualified healthcare provider for medical advice. "
        "Model trained on NHANES data (August 2021 - August 2023)."
    )

else:
    st.info("Click 'Predict' to get a result.")