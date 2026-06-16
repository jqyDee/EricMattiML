import os

import joblib
import pandas as pd
from config import (
    DATASET_PATH,
    LR_MODEL_SAVE_PATH,
    LR_SCALER_SAVE_PATH,
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

    print(f"Loading test dataset from {DATASET_PATH}...")
    df_test = pd.read_csv(DATASET_PATH)

    X_test = df_test.drop("Class", axis=1)
    y_test = df_test["Class"]

    cols_to_scale = ["Time", "Amount"]
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print("\n==========================================")
    print(f"     FINAL TEST RESULTS ({variant.upper()})      ")
    print("==========================================\n")

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    auprc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"--- AUPRC: {auprc:.4f}; ROC AUC: {roc_auc:.4f} ---")
