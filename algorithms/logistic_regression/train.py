import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        DATASET_PATH,
        LR_MODEL_SAVE_PATH,
        LR_RANDOM_STATE,
        LR_SCALER_SAVE_PATH,
        LR_SPLIT_RATIO,
        LR_THRESHOLD_SAVE_PATH,
        LR_TUNED_CV_RESULTS_PATH,
        LR_TUNED_MODEL_SAVE_PATH,
        LR_TUNED_SCALER_SAVE_PATH,
        LR_TUNED_THRESHOLD_SAVE_PATH,
    )
except ModuleNotFoundError:
    DATASET_PATH = None
    LR_SPLIT_RATIO = 0.2
    LR_RANDOM_STATE = 42
    LR_MODEL_SAVE_PATH = None
    LR_SCALER_SAVE_PATH = None
    LR_THRESHOLD_SAVE_PATH = None
    LR_TUNED_MODEL_SAVE_PATH = None
    LR_TUNED_SCALER_SAVE_PATH = None
    LR_TUNED_THRESHOLD_SAVE_PATH = None
    LR_TUNED_CV_RESULTS_PATH = None
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler


def process_dataset(dataset_path=None):
    """
    Load and split the dataset into training and validation sets.

    Args:
        dataset_path: Path to the dataset CSV file. If None, uses the default path.

    Returns:
        tuple: The training and validation sets (X_train, X_val, y_train, y_val).
    """

    path = Path(dataset_path) if dataset_path else DATASET_PATH
    assert isinstance(path, Path)
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)

    # Drop the Class column for training/testing data respectively
    X = df.drop(
        ["Class", "Time"], axis=1
    )  # dropping the Time column, seems to yield no change, meaning Time
    # features are not influencing the model
    y = df["Class"]

    print("Splitting data into Train and Validation sets...")
    return train_test_split(
        X, y, test_size=LR_SPLIT_RATIO, random_state=LR_RANDOM_STATE, stratify=y
    )


def scale_data(X_train, X_val):
    """
    Scale the Time and Amount features using RobustScaler.

    Args:
        X_train: training features
        X_val: validation features

    Returns:
        X_train: scaled training features
        X_val: scaled validation features
        scaler: the fitted RobustScaler instance
    """

    print("Scaling Time and Amount features...")
    scaler = RobustScaler()

    # cols_to_scale = ["Time", "Amount"]
    cols_to_scale = ["Amount"]  # tried dropping the Time column

    # Fit ONLY on train to prevent data leakage
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return X_train, X_val, scaler


def find_best_threshold(y_val, y_pred_proba):
    """
    Find the best threshold for the given precision-recall curve.

    Args:
        y_val: true labels
        y_pred_proba: predicted probabilities

    Returns:
        best_threshold: the threshold that maximizes the F1 score
    """

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # [:-1] drops the last point sklearn appends (precision=1, recall=0, no threshold).
    # np.where guards against the separate case where precision+recall==0 (would cause 0/0).
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) == 0,  # avoid division by zero
        0,  # fallback to 0 if precision + recall are 0
        2
        * precisions[:-1]
        * recalls[:-1]
        / (precisions[:-1] + recalls[:-1]),  # compute F1 score
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(
        f"\n--- Best threshold: {best_threshold:.20f} (F1={f1_scores[best_idx]:.4f}) ---"
    )

    return best_threshold


def validate(model, X_val, y_val):
    """
    Validate the model on the validation set and compute metrics.

    Args:
        model: the trained logistic regression model
        X_val: validation features
        y_val: validation labels

    Returns:
        best_threshold: the threshold that maximizes the F1 score
    """

    print("\nEvaluating on Validation Set:")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(
        int
    )  # cutoff any predictions below the threshold

    print("\n--- Model Parameters ---")
    print(model.get_params())

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred))

    auprc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"--- AUPRC: {auprc:.4f}  |  ROC-AUC: {roc_auc:.4f} ---")

    return best_threshold


def _artifact_paths(tuned, output_dir=None):
    """
    Returns the paths for saving model artifacts based on the tuning status and output directory.

    Args:
        tuned: Whether the model is tuned or not.
        output_dir: The directory to save the artifacts. If None, uses the default paths.

    Returns:
        tuple: The paths for the model, scaler, threshold, and CV results.
    """

    prefix = "lr_tuned" if tuned else "lr_simple"
    if output_dir:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        return (
            base / f"{prefix}_model.pkl",
            base / f"{prefix}_scaler.pkl",
            base / f"{prefix}_threshold.pkl",
            base / f"{prefix}_cv_results.pkl",
        )
    return (
        LR_TUNED_MODEL_SAVE_PATH if tuned else LR_MODEL_SAVE_PATH,
        LR_TUNED_SCALER_SAVE_PATH if tuned else LR_SCALER_SAVE_PATH,
        LR_TUNED_THRESHOLD_SAVE_PATH if tuned else LR_THRESHOLD_SAVE_PATH,
        LR_TUNED_CV_RESULTS_PATH,
    )


def save(model, scaler, threshold, tuned, output_dir=None):
    """
    Save the trained model, scaler, and threshold to disk.

    Args:
        model: The trained model.
        scaler: The fitted scaler.
        threshold: The threshold for classification.
        tuned: Whether the model was tuned or not.
        output_dir: The directory to save the artifacts to.
    """

    model_path, scaler_path, threshold_path, _ = _artifact_paths(tuned, output_dir)
    assert isinstance(model_path, Path)
    assert isinstance(scaler_path, Path)
    assert isinstance(threshold_path, Path)

    print(f"\nSaving artifacts to {model_path.parent} ...")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(threshold_path, "wb") as f:
        pickle.dump(threshold, f)

    print("Artifacts saved successfully.")


def train_simple(X_train, y_train):
    """
    Train a simple Logistic Regression model with default parameters for quick
    training and evaluation.

    Args:
        X_train: The training features.
        y_train: The training labels.

    Returns:
        The trained model.
    """

    print("Training Logistic Regression (Simple) ...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=LR_RANDOM_STATE,
        verbose=2,
    )
    model.fit(X_train, y_train)
    return model


def train_tuned(
    X_train,
    y_train,
    output_dir=None,
    scoring="average_precision",
    create_cv_results=False,
):
    """
    Train a tuned Logistic Regression model using GridSearchCV.

    Args:
        X_train: The training features.
        y_train: The training labels.
        output_dir: The directory to save the artifacts to.
        scoring: The scoring metric to use for tuning.

    Returns:
        The trained model.
    """

    print("Training Logistic Regression with GridSearchCV...")
    param_grid = {
        "C": [0.1, 1, 10],
        "class_weight": [None, "balanced"],
    }

    # Check which solver is creates the best scoring result. LBFGS is the
    # default for Logistic Regression in scikit-learn. Saga is slower but only
    # solver supports L1 regularization.

    # lbfgs for L2, as this converges much faster than saga on large datasets
    grid_l2 = GridSearchCV(
        LogisticRegression(
            max_iter=1000,
            random_state=LR_RANDOM_STATE,
            solver="lbfgs",
        ),
        param_grid,
        scoring=scoring,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )
    # saga required for L1
    # lowering tol to 1e-3 from default 1e-4 reduces iterations by 10x. This is
    # necessary to keep the computation time reasonable.
    grid_l1 = GridSearchCV(
        LogisticRegression(
            max_iter=5000,
            random_state=LR_RANDOM_STATE,
            solver="saga",
            penalty="l1",
            tol=1e-3,
        ),
        param_grid,
        scoring=scoring,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    print("--- Grid search: L2 (lbfgs) ---")
    grid_l2.fit(X_train, y_train)
    print(
        f"L2 best: C={grid_l2.best_params_['C']}  |  CV score: {grid_l2.best_score_:.4f} | Class weight: {grid_l2.best_params_['class_weight']}"
    )

    print("--- Grid search: L1 (saga) ---")
    grid_l1.fit(X_train, y_train)
    print(
        f"L1 best: C={grid_l1.best_params_['C']}  |  CV score: {grid_l1.best_score_:.4f} | Class weight: {grid_l1.best_params_['class_weight']}"
    )

    best = grid_l2 if grid_l2.best_score_ >= grid_l1.best_score_ else grid_l1
    print(
        f"Selected: {'L2' if best is grid_l2 else 'L1'}  |  CV score: {best.best_score_:.4f}"
    )
    print(f"Best params: {best.best_params_}")

    # This is only for internal visualization and not necessary for the model itself
    if create_cv_results:
        _, _, _, cv_results_path = _artifact_paths(tuned=True, output_dir=output_dir)
        assert isinstance(cv_results_path, Path)

        with open(cv_results_path, "wb") as f:
            pickle.dump({"l2": grid_l2.cv_results_, "l1": grid_l1.cv_results_}, f)

    return best.best_estimator_


def run(
    scoring,
    create_cv_results,
    tuned=False,
    dataset_path=None,
    output_dir=None,
):
    """
    Run logistic regression training.

    Args:
        tuned (bool): whether to use tuned hyperparameters
        dataset_path (str): path to the dataset
        output_dir (str): directory to save artifacts
        create_cv_results (bool): whether to save cross-validation results
        scoring (str): scoring metric to use for hyperparameter tuning
    """

    variant = "tuned" if tuned else "simple"
    print(f"Running logistic regression training ({variant})...")

    X_train, X_val, y_train, y_val = process_dataset(dataset_path)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)

    X_train, X_val, scaler = scale_data(X_train, X_val)
    model = (
        train_tuned(
            X_train,
            y_train,
            output_dir,
            scoring=scoring,
            create_cv_results=create_cv_results,
        )
        if tuned
        else train_simple(X_train, y_train)
    )
    threshold = validate(model, X_val, y_val)
    save(model, scaler, threshold, tuned, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Logistic Regression fraud detection model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset CSV. Defaults to config path.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_false",
        default=True,
        help="Use GridSearchCV hyperparameter tuning.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="average_precision",
        help='Scoring metric for GridSearchCV. Defaults to "average_precision".',
    )
    parser.add_argument(
        "--create-cv-results",
        action="store_true",
        default=False,
        help="Save cross-validation results from GridSearchCV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save model artifacts. Defaults to ./models",
    )
    args = parser.parse_args()

    run(
        tuned=args.no_tune,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        create_cv_results=args.create_cv_results,
        scoring=args.scoring,
    )
