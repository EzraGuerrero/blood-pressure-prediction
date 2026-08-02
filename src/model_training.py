# ---------- IMPORTS ----------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import joblib # For saving the model
import os
import yaml

# ---------- PROJECT DIR ----------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)   

# ---------- DEFINE FUNCTION ----------
def train_pipeline(config_path="config/params.yaml"):
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_csv("data/processed/cleaned_merged.csv")
    
    # Create dummies
    df = pd.get_dummies(df, columns=["RIDRETH1"], prefix="eth", drop_first=True)
    eth_cols = [c for c in df.columns if c.startswith("eth_")]
    for col in eth_cols:
        df[col] = df[col].astype(int)
    
    # Features
    feature_cols = config["features"]["numeric"] + config["features"]["categorical"]
    X = df[feature_cols]
    y = df["BPXOSY1"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    # Pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), config["features"]["numeric"]),
        ("cat", "passthrough", config["features"]["categorical"])
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/bp_pipeline.pkl")
    
    # Compute and save OOD data
    X_train_scaled = pipeline.named_steps['preprocessor'].transform(X_train)
    mean_vector = np.mean(X_train_scaled, axis=0)
    cov_matrix_inv = np.linalg.inv(np.cov(X_train_scaled, rowvar=False))
    
    joblib.dump(mean_vector, "models/ood_mean.joblib")
    joblib.dump(cov_matrix_inv, "models/ood.joblib")
    
    print("✅ Model and OOD files saved")
    return pipeline


# ---------- CALL FUNCTION ----------
if __name__ == "__main__":
    train_pipeline()