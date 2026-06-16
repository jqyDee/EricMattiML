# Project Features: Transactions (Fraud Detection) Dataset

Binary classification on financial transactions dataset. Distinguish legitimate vs. fraudulent transactions. Model size hard limit: **50 MB**.

## 1. Dataset Overview
* **Domain:** Financial Transactions / Fraud Detection
* **Instances:** 227,845 rows
* **Features:** 31 columns
  * `Time`: Seconds elapsed since first transaction in dataset.
  * `Feature0`–`Feature27`: Anonymized numerical features (PCA-transformed, pre-scaled).
  * `Amount`: Monetary value of transaction.
* **Target Variable:** `Class` (0 = Legitimate, 1 = Fraudulent)
* **Class imbalance:** 394 fraud cases out of 227,845 total (0.17% fraud rate). Accuracy meaningless — use AUPRC/F1/Recall.

## 2. Data Preprocessing Pipeline
* **Scaling:** `RobustScaler` applied to `Time` and `Amount` only. `Feature0`–`Feature27` are already PCA-transformed and left as-is.
* **Imbalance Handling:** `class_weight='balanced'` on model — forces attention to minority fraud class without resampling.
* **Validation Split:** 80/20 train/val split with `stratify=y` and `random_state=42` to preserve 0.17% fraud rate in both sets. Scaler fit on train only, transform applied to val/test (no data leakage).

## 3. Implemented Machine Learning Methods

### Method 1: Logistic Regression ✅ (implemented in `algorithms/logistic_regression/`)
* *Rationale:* Strong interpretable baseline for binary classification. Explicitly models fraud probability. Tiny model size (well under 50 MB). Handles imbalance via `class_weight='balanced'`.
* *Hyperparameters:* `class_weight='balanced'`, `max_iter=1000`, `random_state=42`. Regularization strength (`C`) and penalty type (`l1` vs `l2`) candidates for tuning.
* *Artifacts:* Model → `data/lr_fraud_model.pkl`, Scaler → `data/lr_robust_scaler.pkl`

### Method 2: Neural Network 🚧 (stub in `algorithms/neural_network/`)
* *Rationale:* Can capture non-linear relationships between PCA features. Flexible architecture allows tuning capacity to stay within 50 MB limit.
* *Hyperparameters to tune:* Layer count/width, activation function, learning rate, dropout, batch size.
* *Status:* `train()` and `evaluate()` not yet implemented.

## 4. Evaluation & Results
* **Primary Metrics:** AUPRC (`average_precision_score`), F1-Score, Recall for class 1.
* **Why not accuracy:** 0.17% fraud rate → 99.83% accuracy achievable by predicting all-legitimate. Useless.
* **Validation Performance:**
  * Logistic Regression: [Insert F1/Recall/AUPRC]
  * Neural Network: [Insert F1/Recall/AUPRC]
* **Final Model Selection:** [State best model by Recall/F1/AUPRC]

## 5. Deployment Constraints
* **Model size limit:** 50 MB (hard limit for submission).
* **Runtime:** Must run on course JupyterHub (NumPy, scikit-learn, PyTorch available).
* **Reproducibility:** Training disabled by default in test notebook. Models saved to `data/` and loaded via `joblib` for inference without retraining.
* **Submission:** Training + eval scripts as ZIP on OLAT. Test notebook on JupyterHub (do not rename/replace).
