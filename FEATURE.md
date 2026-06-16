# Project Features: Transactions (Fraud Detection) Dataset

Binary classification on financial transactions dataset. Distinguish legitimate vs. fraudulent transactions. Model size hard limit: **50 MB**.

## 1. Dataset Overview
* **Domain:** Financial Transactions / Fraud Detection
* **Instances:** 227,845 rows
* **Features:** 31 columns
  * `Time`: Seconds elapsed since first transaction. Dropped — no predictive signal (confirmed: +0.0005 AUPRC, not worth the leakage risk).
  * `Feature0`–`Feature27`: Anonymized numerical features (PCA-transformed, pre-scaled).
  * `Amount`: Monetary value of transaction.
* **Target Variable:** `Class` (0 = Legitimate, 1 = Fraudulent)
* **Class imbalance:** 394 fraud cases out of 227,845 total (0.17% fraud rate). Accuracy meaningless — use AUPRC/F1/Recall.

## 2. Data Preprocessing Pipeline
* **Scaling:** `RobustScaler` applied to `Amount` only (LR). `Feature0`–`Feature27` pre-scaled; `Time` dropped. RF requires no scaling — tree splits on rank, not magnitude.
* **Imbalance Handling:** `class_weight='balanced'` on both models — forces attention to minority fraud class without resampling.
* **Validation Split:** 80/20 train/val split with `stratify=y` and `random_state=42` to preserve 0.17% fraud rate in both sets. Scaler fit on train only, transform applied to val/test (no data leakage).

## 3. Implemented Machine Learning Methods

### Method 1: Logistic Regression (implemented in `algorithms/logistic_regression/`)
* **Rationale:** Strong interpretable baseline for binary classification. Explicitly models fraud probability. Tiny model size (well under 50 MB). Handles imbalance via `class_weight='balanced'`.
* **Hyperparameter search:** `GridSearchCV` over `C ∈ {0.1, 1, 10}` × `penalty ∈ {l1, l2}` × `class_weight ∈ {balanced, None}`, 3-fold CV, scored on `average_precision`. Best: L1, C=0.1, balanced.
* **Artifacts:** `models/lr_simple_model.pkl`, `models/lr_tuned_model.pkl`, `models/lr_*_scaler.pkl`, `models/lr_*_threshold.pkl`

### Method 2: Random Forest (implemented in `algorithms/random_forest/`)
* **Rationale:** Ensemble of decision trees — captures non-linear interactions between PCA features without feature scaling. Bootstrap sampling + feature randomness reduce variance. Strong on tabular fraud data.
* **Hyperparameter search:** 5 rounds of `GridSearchCV` / `RandomizedSearchCV` over `n_estimators`, `max_depth`, `max_features`, `class_weight`, `min_samples_split/leaf`, `max_samples`. Settled params: `n_estimators=500`, `max_depth=None` (full trees), `max_features='log2'`, `class_weight='balanced'`.
* **Key finding:** `max_depth=None` consistently wins (unlimited depth + 500-tree averaging prevents overfitting). `log2` beats `sqrt` at 500 trees.
* **Artifacts:** `models/rf_simple_model.pkl`, `models/rf_tuned_model.pkl`, `models/rf_*_threshold.pkl` (no scaler needed)

## 4. Evaluation & Results
* **Primary Metric:** AUPRC (`average_precision_score`) — not inflated by true negatives, directly measures precision/recall tradeoff on fraud class.
* **Secondary Metric:** ROC-AUC — used by grader leaderboard (`leader_board_predict_fn`).
* **Why not accuracy:** 0.17% fraud rate → 99.83% accuracy achievable by predicting all-legitimate. Useless.
* **Decision threshold:** F1-maximizing threshold found via PR curve sweep (saved per model). Default 0.5 is suboptimal for imbalanced data.
* **Validation Performance (80/20 stratified val set):**
  * Logistic Regression Simple: AUPRC ~0.787
  * Logistic Regression Tuned (L1, C=0.1): AUPRC ~0.807
  * Random Forest Simple (100 trees): AUPRC ~0.850
  * Random Forest Tuned (500 trees, log2, balanced): AUPRC 0.9001, ROC-AUC 0.9707
* **Best model:** RF Tuned — highest AUPRC and ROC-AUC on validation set.

## 5. Deployment Constraints
* **Model size limit:** 50 MB (hard limit for submission). Both models well within limit.
* **Runtime:** Must run on course JupyterHub (NumPy, scikit-learn available).
* **Reproducibility:** Training disabled by default in test notebook. Models saved to `models/` and loaded via `joblib` for inference without retraining.
* **Submission:** Training + eval scripts as ZIP on OLAT. Test notebook on JupyterHub (do not rename/replace).
