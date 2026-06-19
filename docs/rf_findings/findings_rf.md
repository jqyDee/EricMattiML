# Tuned Random/Grid Search

> Important Note: These benchmarks were run on a machine with Python 3.13.x. The
  scores can therefore differ a tiny bit.

## Grid Search
Input:
```python
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 4],
    "class_weight": [None, "balanced"],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=RF_RANDOM_STATE, n_jobs=-1),
    param_grid,
    scoring=scoring,
    cv=3,
    n_jobs=1,
    verbose=2,
)
```

Output:
```
Best params: {'class_weight': 'balanced', 'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 300}
CV score: 0.8243

Evaluating on Validation Set:

--- Best threshold: 0.3100 (F1=0.9032) ---

--- Model Parameters ---
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 300, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45484     6]
 [    9    70]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.92      0.89      0.90        79

    accuracy                           1.00     45569
   macro avg       0.96      0.94      0.95     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8992  |  ROC-AUC: 0.9714 ---
```

## Random Search - Test 1
Input:
```python
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
    "max_features": ["sqrt", "log2", None],
}

# 4×5×3×3×1×3 = 540 combinations - use RandomizedSearchCV
grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=50,  # Test 50 random combinations
    scoring=,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
```

Output:
```
Best params: {'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 20, 'class_weight': 'balanced'}
CV score: 0.8249

Evaluating on Validation Set:

--- Best threshold: 0.3729 (F1=0.9045) ---

--- Model Parameters ---
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 150, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45483     7]
 [    8    71]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.91      0.90      0.90        79

    accuracy                           1.00     45569
   macro avg       0.96      0.95      0.95     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8886  |  ROC-AUC: 0.9648 ---
```

## Random Search - Test 2
Input:
```python
param_grid = {
  "n_estimators": [150, 200, 300],  # removed 50 and 100 trees for now
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2],
  "max_features": ["sqrt", "log2"],
  # "max_depth": [5, 10, 15, 20, None],  # Previous tuning revealed max_depth of None is best
  # "class_weight": ["balanced"],  # Previous tuning revealed `balanced` is best
}

# 4×5×3×3×1×3 = 540 combinations - use RandomizedSearchCV
grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=50,  # Test 50 random combinations
    scoring=,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
```

Output:
```
Best params: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2'}
CV score: 0.8206

Evaluating on Validation Set:

--- Best threshold: 0.3150 (F1=0.9067) ---

--- Model Parameters ---
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 200, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45487     3]
 [   11    68]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.96      0.86      0.91        79

    accuracy                           1.00     45569
   macro avg       0.98      0.93      0.95     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8900  |  ROC-AUC: 0.9599 ---
```

## Random Search - Test 3
Input:
```python
param_grid = {
    "n_estimators": [100, 200, 300, 500],  # removed 50 and 100 trees for now
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["log2"],
    "class_weight": ["balanced"],
}

grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=50,  # Test 50 random combinations
    scoring=scoring,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
```

Output:
```
Best params: {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced'}
CV score: 0.8243

Evaluating on Validation Set:

--- Best threshold: 0.3100 (F1=0.9032) ---

--- Model Parameters ---
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 300, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45484     6]
 [    9    70]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.92      0.89      0.90        79

    accuracy                           1.00     45569
   macro avg       0.96      0.94      0.95     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.8992  |  ROC-AUC: 0.9714 ---
```

## Random Search - Test 4
Input:
```python
param_grid = {
    "n_estimators": [150, 200, 300, 500],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["log2"],
    "max_depth": [None],
    "class_weight": ["balanced"],
}

grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=50,  # Test 50 random combinations
    scoring=scoring,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
```

Output:
```
Best params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'class_weight': 'balanced'}
CV score: 0.8252

Evaluating on Validation Set:

--- Best threshold: 0.2880 (F1=0.8974) ---

--- Model Parameters ---
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 500, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45483     7]
 [    9    70]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.91      0.89      0.90        79

    accuracy                           1.00     45569
   macro avg       0.95      0.94      0.95     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.9001  |  ROC-AUC: 0.9707 ---
```

## Random Search - Test 5
Input:
```python
param_grid = {
    "n_estimators": [300, 500],  # removed 50 and 100 trees for now
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["log2"],
    "max_samples": [None, 0.7, 0.8],
    "max_depth": [None],
    "class_weight": ["balanced"],
}

grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=50,  # Test 50 random combinations
    scoring=scoring,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
```

Output:
```
Best params: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_samples': None, 'max_features': 'log2', 'max_depth': None, 'class_weight': 'balanced'}
CV score: 0.8252

Evaluating on Validation Set:

--- Best threshold: 0.2880 (F1=0.8974) ---

--- Model Parameters ---
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 500, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}

--- Confusion Matrix ---
[[45483     7]
 [    9    70]]

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45490
           1       0.91      0.89      0.90        79

    accuracy                           1.00     45569
   macro avg       0.95      0.94      0.95     45569
weighted avg       1.00      1.00      1.00     45569

--- AUPRC: 0.9001  |  ROC-AUC: 0.9707 ---
```

## Analysis

All six runs used `average_precision` scoring — directly comparable.

| | Grid Search | RS1 | RS2 | RS3 | RS4 | RS5 |
|---|---|---|---|---|---|---|
| n_estimators | 300 | 150 | 200 | 300 | **500** | **500** |
| max_depth | None | 20 | None | None | None | None |
| max_features | sqrt | sqrt | log2 | sqrt | log2 | log2 |
| max_samples | None | None | None | None | None | None |
| class_weight | balanced | balanced | None (accident) | balanced | balanced | balanced |
| CV score | 0.8243 | 0.8249 | 0.8206 | 0.8243 | **0.8252** | **0.8252** |
| Threshold | 0.3100 | 0.3729 | 0.3150 | 0.3100 | 0.2880 | 0.2880 |
| Fraud caught | 70/79 | **71/79** | 68/79 | 70/79 | 70/79 | 70/79 |
| False positives | 6 | 7 | **3** | 6 | 7 | 7 |
| F1 | 0.9032 | 0.9045 | **0.9067** | 0.9032 | 0.8974 | 0.8974 |
| AUPRC | 0.8992 | 0.8886 | 0.8900 | 0.8992 | **0.9001** | **0.9001** |
| ROC-AUC | **0.9714** | 0.9648 | 0.9599 | **0.9714** | 0.9707 | 0.9707 |

**Test 5 reproduced Test 4 exactly** — `max_samples=None` won over 0.7 and 0.8. Reducing the bootstrap sample size did not help: full dataset per tree is already optimal. `max_samples` is settled at default (None).

**Test 4 remains the best run** — AUPRC 0.9001, CV score 0.8252, `n_estimators=500 + log2`. No further improvement found.

**RS Test 2 accidentally dropped `class_weight='balanced'`** — defaulted to `None`, producing highest F1 and fewest FP but worst recall (68/79). Wrong tradeoff for fraud detection.

**Model is converged.** Five parameters are now fully settled across all runs:
- `max_depth=None` — full trees always win
- `n_estimators=500` — more than 300 helps, more than 500 unlikely to
- `max_features='log2'` — more diverse trees than sqrt
- `max_samples=None` — full bootstrap sample optimal
- `class_weight='balanced'` — critical for recall on 0.17% minority class

**Conclusion:** AUPRC 0.9001 is the final RF result. Beats tuned LR (0.8074) by +0.0927. Stop tuning.
