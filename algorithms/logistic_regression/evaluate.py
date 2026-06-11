import os

import joblib
import pandas as pd
from config import DATASET_PATH, MODEL_SAVE_PATH, SCALER_SAVE_PATH
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def evaluate():
    # ==========================================
    # 1. Configuration
    # ==========================================
    # In your final grading, the TAs will likely provide a separate hidden test file.
    # For now, you can test this by pointing it to a dummy test file.
    #
    # ==========================================
    # 2. Safety Checks & Loading Artifacts
    # ==========================================
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(SCALER_SAVE_PATH):
        raise FileNotFoundError(
            "Model or scaler file not found! Please run train.py first."
        )

    print("Loading saved model and scaler...")
    model = joblib.load(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)

    # ==========================================
    # 3. Loading Test Data
    # ==========================================
    print(f"Loading test dataset from {DATASET_PATH}...")
    df_test = pd.read_csv(DATASET_PATH)

    X_test = df_test.drop("Class", axis=1)
    y_test = df_test["Class"]

    # ==========================================
    # 4. Preprocessing Test Data
    # ==========================================
    print("Applying saved scaler to test data...")
    cols_to_scale = ["Time", "Amount"]

    # Notice we ONLY use .transform() here. NEVER use .fit() on test data!
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # ==========================================
    # 5. Prediction & Evaluation
    # ==========================================
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n==========================================")
    print("          FINAL TEST RESULTS              ")
    print("==========================================\n")

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"--- AUPRC (Area Under Precision-Recall Curve): {auprc:.4f} ---")
