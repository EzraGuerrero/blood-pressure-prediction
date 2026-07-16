# 🩺 Blood Pressure Prediction App

A real-world data science project that predicts systolic blood pressure using NHANES 2021–2022 data.

## 📌 Features
- Pulls public health data from NHANES
- Cleans and normalizes data (age, gender, BMI)
- Trains a linear regression model (R²: ~0.40)
- Deploys a live Streamlit web app

## 🚀 Live Demo
👉 [View App](https://ezraguerrero-blood-pressure-prediction.streamlit.app)

## 📦 Setup
1. Clone repo:
   ```bash
   git clone https://github.com/ezraguerrero/blood-pressure-prediction.git
   cd blood-pressure-prediction

2. Create environment:
    ```bash
    conda env create -f blood_pressure_env.yml

3. Run app:
    ```bash
    streamlit run app.py

### Model Performance:

- **R²**: 0.40
- **RMSE**: 0.78 (scaled)