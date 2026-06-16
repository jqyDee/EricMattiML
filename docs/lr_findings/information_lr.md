# Report Outline

IEEE 2-page format. 5 required sections. Notes on what goes where.

---

## 1. Introduction
- Task: binary classification — fraud vs. legitimate transaction
- Dataset: 227,845 transactions, 31 columns (Time, Amount, Feature0–Feature27, Class)
- Feature0–Feature27 are PCA-transformed from original (anonymized) features; pre-scaled
- Extreme imbalance: 394 fraud / 227,451 legit = 0.17% positive rate
- Accuracy useless → primary metric: AUPRC (average precision); secondary: ROC-AUC, F1

## 2. Implementation / ML Process
- **Methods**: Logistic Regression (LR) + [second method: TBD — Random Forest recommended]
- **Preprocessing**:
  - Time dropped — no predictive signal (absorbed into PCA components; see findings)
  - Amount scaled with RobustScaler fit on train only (prevents data leakage)
  - Feature0–Feature27 left as-is (already PCA-transformed)
  - 80/20 stratified train/val split (random_state=42) to preserve 0.17% fraud rate
- **LR hyperparameter search**:
  - GridSearchCV, 3-fold CV, scoring=average_precision (not roc_auc — see findings)
  - Two grids: L2/lbfgs (fast) and L1/saga; C ∈ {0.1, 1, 10}, class_weight ∈ {None, balanced}
  - Winner: L1, C=0.1, class_weight=None
  - Why average_precision scoring: roc_auc inflated by true negatives on imbalanced data; optimizing it selects worse hyperparams on both metrics (AUPRC 0.777 vs 0.807)
- **Decision threshold**: PR curve sweep → F1-maximizing threshold (0.0659) instead of default 0.5

## 3. Results
- LR tuned val: AUPRC 0.807, ROC-AUC 0.983, F1 0.827, Recall 0.85 (67/79 fraud caught)
- LR simple val: AUPRC 0.787, ROC-AUC 0.981, F1 comparable
- [Second method results — TBD]
- **Visualizations** (≥2 required):
  - PR curves simple vs tuned with operating point marked (`lr_pr_curves.png`)
  - Confusion matrices (`lr_confusion_matrices.png`)
  - Threshold analysis (`lr_threshold_analysis.png`)
  - GridSearch C × class_weight × penalty (`lr_gridsearch_results.png`)

## 4. Discussion
- **Worked**: AUPRC-optimized GridSearch, custom threshold sweep, L1 regularization
- **Discarded / tried but dropped**:
  - Time column — +0.0005 AUPRC, identical confusion matrix, not worth including
  - roc_auc as GridSearch scoring — selected class_weight=balanced, AUPRC dropped to 0.778
  - Default threshold 0.5 — suboptimal F1 for imbalanced data
- **Didn't work / open**: class_weight=balanced in LR pushes threshold to 1.0 (degenerate); using custom threshold without balanced weighting works better
- **Potential improvements**: ensemble methods (RF/XGBoost), SMOTE oversampling, probability calibration

## 5. Conclusion
- Test-set performance: [fill after JupyterHub eval]
- LR with L1 regularization + AUPRC-tuned GridSearch + custom threshold achieves strong fraud detection (AUPRC ~0.807 on val)
- Main takeaway: metric choice during hyperparameter search matters as much as the algorithm

---

# Dropping Time Column

## Time Column Dropped
```
L1 best: C=0.1  |  CV score: 0.7450
Selected: L1  |  CV score: 0.7450
Best params: {'C': 0.1, 'class_weight': None}

Evaluating on Validation Set:

--- Best threshold: 0.0659 (F1=0.8272) ---

--- Confusion Matrix ---
[[45474    16]
 [   12    67]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.81      0.85      0.83        79

    accuracy                           1.00     45569
   macro avg       0.90      0.92      0.91     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8074  |  ROC-AUC: 0.9825 ---
```

## Time Column Present
```
L1 best: C=0.1  |  CV score: 0.7445
Selected: L1  |  CV score: 0.7445
Best params: {'C': 0.1, 'class_weight': None}

Evaluating on Validation Set:

--- Best threshold: 0.0633 (F1=0.8272) ---

--- Confusion Matrix ---
[[45474    16]
 [   12    67]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.81      0.85      0.83        79

    accuracy                           1.00     45569
   macro avg       0.90      0.92      0.91     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8069  |  ROC-AUC: 0.9821 ---
```

## Analysis

Dropping `Time` has negligible effect:

| | Time dropped | Time present |
|---|---|---|
| AUPRC | 0.8074 | 0.8069 |
| ROC-AUC | 0.9825 | 0.9821 |
| F1 | 0.8272 | 0.8272 |
| Confusion matrix | identical | identical |
| Best params | C=0.1, no weighting | C=0.1, no weighting |

Difference is +0.0005 AUPRC — within noise, not meaningful. `Time` carries no predictive signal for this model. Since `Feature0`–`Feature27` are PCA-transformed from the original features, any temporal patterns were likely already absorbed into the PCA components. Moser et al.'s intuition was correct — `Time` is effectively a sequence index here. Keeping or dropping it produces identical fraud detection performance.

# Different Scoring

## average_precision
```
L1 best: C=0.1  |  CV score: 0.7450 | Class weight: None
Selected: L1  |  CV score: 0.7450
Best params: {'C': 0.1, 'class_weight': None}

Evaluating on Validation Set:

--- Best threshold: 0.0659 (F1=0.8272) ---

--- Model Parameters ---
{'C': 0.1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': 1, 'max_iter': 5000, 'n_jobs': None, 'penalty': 'deprecated', 'random_state': 42, 'solver': 'saga', 'tol': 0.001, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45474    16]
 [   12    67]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.81      0.85      0.83        79

    accuracy                           1.00     45569
   macro avg       0.90      0.92      0.91     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8074  |  ROC-AUC: 0.9825 ---
```

## roc_auc
```
L1 best: C=0.1  |  CV score: 0.9791 | Class weight: balanced
Selected: L1  |  CV score: 0.9791
Best params: {'C': 0.1, 'class_weight': 'balanced'}

Evaluating on Validation Set:

--- Best threshold: 1.0000 (F1=0.8516) ---

--- Model Parameters ---
{'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': 1, 'max_iter': 5000, 'n_jobs': None, 'penalty': 'deprecated', 'random_state': 42, 'solver': 'saga', 'tol': 0.001, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45480    10]
 [   13    66]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.87      0.84      0.85        79

    accuracy                           1.00     45569
   macro avg       0.93      0.92      0.93     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.7783  |  ROC-AUC: 0.9786 ---
```

## Analysis

GridSearch scoring metric directly determines which hyperparameters get selected — and the choice matters significantly here.

| | `average_precision` | `roc_auc` |
|---|---|---|
| Selected params | C=0.1, no weighting | C=0.1, balanced |
| Threshold | 0.0659 | 1.0000 |
| Fraud caught | 67/79 | 66/79 |
| False positives | 16 | 10 |
| AUPRC | **0.8074** | 0.7783 |
| ROC-AUC | **0.9825** | 0.9786 |

**Key finding:** `average_precision` scoring found a better model on *both* metrics (+0.029 AUPRC, +0.004 ROC-AUC). It also wins on fraud recall (67 vs 66 caught), at the cost of 6 more false positives.

**Why the difference:** ROC-AUC scoring favoured `class_weight='balanced'` because balanced weighting inflates ROC-AUC on imbalanced data — the massive true negative count (45k legit transactions) makes FPR trivially small regardless of fraud detection quality. AUPRC scoring ignores true negatives entirely, forcing GridSearch to optimize purely for fraud detection precision and recall.

**Conclusion:** `average_precision` is the correct scoring choice for this problem. Optimizing ROC-AUC during hyperparameter search leads to a model that looks similar on paper but is measurably worse at the actual task of finding fraud.
