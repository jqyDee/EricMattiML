# Project Features: Transactions (Fraud Detection) Dataset

This project focuses on solving a binary classification problem using a financial transactions dataset. The primary objective is to accurately distinguish between legitimate transactions and fraudulent ones while adhering to a strict 50 MB model size limit for deployment.

## 1. Dataset Overview
* **Domain:** Financial Transactions / Fraud Detection
* **Features:** 30 predictive features
  * `Time`: Seconds elapsed between the transaction and the first transaction in the dataset.
  * `Feature0` to `Feature27`: Anonymized numerical features (PCA transformed).
  * `Amount`: The monetary value of the transaction.
* **Target Variable:** `Class` (0 = Legitimate Transaction, 1 = Fraudulent Transaction).
* **Data Challenges:** Extreme class imbalance (frauds are highly infrequent compared to normal transactions) and mixed feature scales.

## 2. Data Preprocessing Pipeline
* **Scaling:** Applied `RobustScaler` (or `StandardScaler`) to the `Time` and `Amount` columns to bring them onto the same scale as the PCA-transformed features without being skewed by massive transaction outliers.
* **Imbalance Handling:** [Insert your chosen method here, e.g., Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training set to create synthetic examples of the minority fraud class / Used penalization via `class_weight='balanced'`].
* **Validation Split:** The dataset was split into an 80/20 training and validation set before any oversampling was applied to prevent data leakage.

## 3. Implemented Machine Learning Methods
* **Method 1: [e.g., Logistic Regression]**
  * *Rationale:* Serves as a strong, interpretable, and lightweight baseline for binary classification. It explicitly models the probability of fraud and easily adheres to the model size constraint.
  * *Hyperparameters Tuned:* Regularization strength (`C`), penalty type (`l1` vs `l2`).
* **Method 2: [e.g., Random Forest Classifier / XGBoost]**
  * *Rationale:* Highly effective on tabular datasets. It captures complex, non-linear relationships between the PCA features without requiring extensive feature engineering.
  * *Hyperparameters Tuned:* Number of estimators, maximum tree depth (constrained to prevent exceeding the 50 MB memory limit), and minimum samples per split.

## 4. Evaluation & Results
* **Primary Metrics:** Due to the severe class imbalance, accuracy was discarded in favor of **F1-Score**, **Recall** (to ensure maximum fraud detection), and the **Area Under the Precision-Recall Curve (AUPRC)**.
* **Validation Performance:**
  * Method 1: [Insert F1/Recall/AUPRC Score]
  * Method 2: [Insert F1/Recall/AUPRC Score]
* **Final Model Selection:** [State which model performed best regarding Recall/F1 and was chosen for the final submission].

## 5. Deployment Constraints
* **Model Size:** [e.g., 15 MB] (Successfully kept strictly under the 50 MB limitation).
* **Reproducibility:** Code is fully runnable on JupyterHub, complete with a flag to disable training during test-set evaluation. The final model parameters are saved to disk and loaded dynamically for inference.