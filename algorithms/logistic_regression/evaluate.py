import os

import joblib
import pandas as pd
from config import (
    DATASET_PATH,
    LR_MODEL_SAVE_PATH,
    LR_RANDOM_STATE,
    LR_SCALER_SAVE_PATH,
    LR_SPLIT_RATIO,
    LR_THRESHOLD_SAVE_PATH,
    LR_TUNED_MODEL_SAVE_PATH,
    LR_TUNED_SCALER_SAVE_PATH,
    LR_TUNED_THRESHOLD_SAVE_PATH,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def evaluate(tuned=False):
    model_path = LR_TUNED_MODEL_SAVE_PATH if tuned else LR_MODEL_SAVE_PATH
    scaler_path = LR_TUNED_SCALER_SAVE_PATH if tuned else LR_SCALER_SAVE_PATH
    threshold_path = LR_TUNED_THRESHOLD_SAVE_PATH if tuned else LR_THRESHOLD_SAVE_PATH

    for path in [model_path, scaler_path, threshold_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run train first.")

    variant = "tuned" if tuned else "simple"
    print(f"Loading {variant} model, scaler, and threshold...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    threshold = joblib.load(threshold_path)
    print(f"Using decision threshold: {threshold:.4f}")

    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=LR_SPLIT_RATIO, random_state=LR_RANDOM_STATE, stratify=y
    )

    X_val = X
    y_val = y
    
    assert isinstance(X_val, pd.DataFrame)

    X_val[["Amount"]] = scaler.transform(X_val[["Amount"]])

    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print("\n\n" + "=" * 100)
    print(f"VAL SET RESULTS LOGISTIC REGRESSION ({variant.upper()})")
    print("=" * 100 + "\n\n")

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred))

    auprc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"--- AUPRC: {auprc:.4f}  |  ROC-AUC: {roc_auc:.4f} ---")
