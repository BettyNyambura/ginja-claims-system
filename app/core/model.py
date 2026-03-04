import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.core.features import engineer_features, FEATURE_COLUMNS

MODEL_PATH  = "models/claims_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def train(df: pd.DataFrame) -> dict:
    """
    Train an XGBoost classifier on labelled claims data.

    Why XGBoost:
    - Handles tabular data extremely well
    - Robust to feature scale differences
    - Produces probability scores (0-1) natively
    - Built-in feature importance for explainability
    - Fast to train even on modest hardware

    Args:
        df: Raw claims DataFrame with is_fraud label column.

    Returns:
        Dictionary with evaluation metrics and feature importances.
    """
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features — helps XGBoost converge faster
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred      = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc         = roc_auc_score(y_test, y_pred_prob)
    report      = classification_report(y_test, y_pred, output_dict=True)

    # Persist model and scaler
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))

    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "auc_roc":             round(auc, 4),
        "classification_report": report,
        "feature_importances": importances,
    }


def load_model():
    """Load persisted model and scaler from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python -m app.core.model` to train first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(claim: dict) -> dict:
    """
    Run inference on a single claim dictionary.

    Args:
        claim: Raw claim fields as a dictionary.

    Returns:
        Dictionary with risk_score, decision, confidence, and top reasons.
    """
    model, scaler = load_model()

    df = pd.DataFrame([claim])
    df = engineer_features(df)

    X       = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    risk_score  = float(model.predict_proba(X_scaled)[0][1])
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))

    # Top 2 features driving this prediction
    top_reasons = sorted(importances, key=importances.get, reverse=True)[:2]

    return {
        "risk_score": round(risk_score, 4),
        "top_reasons": top_reasons,
    }


if __name__ == "__main__":
    from app.data.generator import generate_dataset

    print("Generating synthetic training data...")
    df = generate_dataset()

    print("Training model...")
    metrics = train(df)

    print("\nFeature Importances:")
    for feat, score in sorted(metrics["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat:30s} {score:.4f}")