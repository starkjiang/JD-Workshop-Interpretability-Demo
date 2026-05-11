"""
model.py — Train and cache the Random Forest model for the JD demo.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "ai4i2020.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cache.pkl")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FEATURE_LABELS = {
    "Air temperature [K]": "Air Temperature (K)",
    "Process temperature [K]": "Process Temperature (K)",
    "Rotational speed [rpm]": "Rotational Speed (rpm)",
    "Torque [Nm]": "Torque (Nm)",
    "Tool wear [min]": "Tool Wear (min)",
}

TARGET = "Machine failure"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, df


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


def get_shap_explainer(model, X_train):
    explainer = shap.TreeExplainer(model, X_train)
    return explainer


def get_feature_ranges(X):
    ranges = {}
    for col in FEATURES:
        ranges[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "p25": float(X[col].quantile(0.25)),
            "p75": float(X[col].quantile(0.75)),
        }
    return ranges
