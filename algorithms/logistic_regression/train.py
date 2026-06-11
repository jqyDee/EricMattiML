import pickle

import pandas as pd
from config import DATASET_PATH, MODEL_SAVE_PATH, SCALER_SAVE_PATH
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def process_dataset():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # stratify=y is CRITICAL here to ensure the 1% fraud rate is preserved in both sets
    print("Splitting data into Train and Validation sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_data(X_train, X_val):
    print("Scaling Time and Amount features...")
    scaler = RobustScaler()

    cols_to_scale = ["Time", "Amount"]

    # Fit the scaler ONLY on the training data to prevent data leakage, then transform
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return X_train, X_val, scaler


def train():
    X_train, X_val, y_train, y_val = process_dataset()

    # shutup type checking
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)

    X_train, X_val, scaler = scale_data(X_train, X_val)

    # ==========================================
    # 4. Model Training
    # ==========================================
    print("Training Logistic Regression model...")
    # class_weight='balanced' forces the model to pay heavy attention to the minority (fraud) class
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ==========================================
    # 5. Validation Evaluation
    # ==========================================
    print("\nEvaluating on Validation Set:")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred))

    auprc = average_precision_score(y_val, y_pred_proba)
    print(f"--- AUPRC (Area Under Precision-Recall Curve): {auprc:.4f} ---")

    # ==========================================
    # 6. Saving Artifacts (Using Pickle)
    # ==========================================
    print("\nSaving model and scaler to disk using pickle...")

    # Write in binary mode ('wb')
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Training complete! Artifacts saved successfully.")


if __name__ == "__main__":
    train()
