import os

import joblib
import pandas as pd
from config import (
    DATASET_PATH,
    RF_MODEL_SAVE_PATH,
    RF_RANDOM_STATE,
    RF_SPLIT_RATIO,
    RF_THRESHOLD_SAVE_PATH,
    RF_TUNED_MODEL_SAVE_PATH,
    RF_TUNED_THRESHOLD_SAVE_PATH,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def evaluate(tuned=False):
    model_path = RF_TUNED_MODEL_SAVE_PATH if tuned else RF_MODEL_SAVE_PATH
    threshold_path = RF_TUNED_THRESHOLD_SAVE_PATH if tuned else RF_THRESHOLD_SAVE_PATH

    for path in [model_path, threshold_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run train first.")

    variant = "tuned" if tuned else "simple"
    print(f"Loading {variant} model and threshold...")
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    print(f"Using decision threshold: {threshold:.4f}")

    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=RF_SPLIT_RATIO, random_state=RF_RANDOM_STATE, stratify=y
    )

    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print("\n==========================================")
    print(f"VAL SET RESULTS RANDOM FOREST ({variant.upper()})")
    print("==========================================\n")

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred))

    auprc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"--- AUPRC: {auprc:.4f}  |  ROC-AUC: {roc_auc:.4f} ---")
