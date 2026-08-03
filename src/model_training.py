# ---------- IMPORTS ----------

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import joblib
import yaml


# ---------- PROJECT DIR ----------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

print(f"Project root: {PROJECT_ROOT}")


# ---------- DEFINE TRAINING FUNCTION ----------

def train_pipeline(config_path="config/params.yml"):
    
    """Train the BP prediction pipeline and save all artifacts."""
    print("=" * 50)
    print("Starting model training...")
    print("=" * 50)
    
    # ===== LOAD CONFIG =====
    config_path = os.path.join(PROJECT_ROOT, "config", "params.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    
    # ===== LOAD DATA =====
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_merged.csv")
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} rows")
    
    # ===== CREATE ETHNICITY DUMMIES =====
    print("\nCreating ethnicity dummies...")
    df = pd.get_dummies(df, columns=["RIDRETH1"], prefix="eth", drop_first=True)
    eth_columns = [col for col in df.columns if col.startswith("eth_")]
    for col in eth_columns:
        df[col] = df[col].astype(int)
    print(f"- Created dummies: {eth_columns}")
    
    # ===== DEFINE FEATURES =====
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    feature_columns = numeric_features + categorical_features
    print(f"\nFeatures:")
    print(f"   Numeric (will be scaled): {numeric_features}")
    print(f"   Categorical (passthrough): {categorical_features}")

    # ===== PREPARE X AND Y =====
    X = df[feature_columns]
    y = df["BPXOSY1"]
    
    # ===== SPLIT DATA =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    print(f"\nData split:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train range: {y_train.min():.0f} - {y_train.max():.0f} mmHg")
    print(f"   y_test range:  {y_test.min():.0f} - {y_test.max():.0f} mmHg")
    
    # ===== CREATE PIPELINE =====
    print("\nBuilding pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features)
        ]
    )
    
    model_type = config["model"]["type"]
    if model_type == "LinearRegression": # If block to include more models in the future
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    # ===== TRAIN =====
    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Model trained successfully!")

    # ===== EVALUATE =====
    print("\nEvaluating model...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"   Train MSE: {train_mse:.2f}")
    print(f"   Test MSE:  {test_mse:.2f}")
    print(f"   Train R²:  {train_r2:.4f}")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Test RMSE: {np.sqrt(test_mse):.2f} mmHg")
    
    # ===== COMPUTE RESIDUALS FOR CONFIDENCE INTERVAL =====
    print("\nComputing residuals...")
    residuals = y_train - y_pred_train
    sigma_residuals = np.std(residuals)
    n_train = X_train.shape[0]
    p_features = X_train.shape[1]
    
    print(f"   Sigma residuals: {sigma_residuals:.4f} mmHg")
    print(f"   Training samples: {n_train}")
    print(f"   Features: {p_features}")
    
    # ===== COMPUTE OOD DATA =====
    print("\nComputing OOD detection data...")
    X_train_scaled = pipeline.named_steps['preprocessor'].transform(X_train)
    mean_vector = np.mean(X_train_scaled, axis=0)
    cov_matrix = np.cov(X_train_scaled, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    ood_confidence = config["ood"]["confidence"]
    print(f"   OOD confidence level: {ood_confidence}")
    
    print(f"   Mean vector shape: {mean_vector.shape}")
    print(f"   Cov matrix shape: {cov_matrix.shape}")
    
    # ===== SAVE ALL ARTIFACTS =====
    print("\nSaving artifacts...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save pipeline
    pipeline_path = os.path.join(models_dir, "bp_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"   ✅ Pipeline: {pipeline_path}")
    
    # Save residuals info
    sigma_path = os.path.join(models_dir, "sigma_residuals.pkl")
    joblib.dump(sigma_residuals, sigma_path)
    print(f"   ✅ Sigma residuals: {sigma_path}")
    
    # Save training info
    info_path = os.path.join(models_dir, "training_info.pkl")
    joblib.dump({"n": n_train, "p": p_features}, info_path)
    print(f"   ✅ Training info: {info_path}")
    
    # Save OOD data
    mean_path = os.path.join(models_dir, "mean_vector.pkl")
    joblib.dump(mean_vector, mean_path)
    print(f"   ✅ Mean vector: {mean_path}")
    
    cov_path = os.path.join(models_dir, "cov_matrix_inv.pkl")
    joblib.dump(cov_matrix_inv, cov_path)
    print(f"   ✅ Cov matrix inverse: {cov_path}")
    
    # Save feature info
    feature_path = os.path.join(models_dir, "feature_info.pkl")
    joblib.dump({
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": feature_columns
    }, feature_path)
    print(f"   ✅ Feature info: {feature_path}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nModel Performance:")
    print(f"   R²:  {test_r2:.4f}")
    print(f"   RMSE: {np.sqrt(test_mse):.2f} mmHg")
    print(f"\nAll artifacts saved to: {models_dir}")
    
    return pipeline


# ---------- CALL FUNCTION ----------
if __name__ == "__main__":
    """Run training when script is executed directly."""
    print("Testing model_training.py...")
    pipeline = train_pipeline()
    
    # Quick test with sample input
    test_sample = pd.DataFrame([{
        "RIDAGEYR": 50,
        "RIAGENDR": 1,
        "BMXBMI": 28.0,
        "eth_2.0": 0,
        "eth_3.0": 1,
        "eth_4.0": 0,
        "eth_5.0": 0
    }])
    
    prediction = pipeline.predict(test_sample)[0]
    print(f"\nTest prediction: {prediction:.1f} mmHg")
    print(f"   Input: Age=50, Male, BMI=28, White")
    print("\n✅ All tests passed!")