import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        DATASET_PATH,
        RF_MODEL_SAVE_PATH,
        RF_RANDOM_STATE,
        RF_SPLIT_RATIO,
        RF_THRESHOLD_SAVE_PATH,
        RF_TUNED_CV_RESULTS_PATH,
        RF_TUNED_MODEL_SAVE_PATH,
        RF_TUNED_THRESHOLD_SAVE_PATH,
    )
except ModuleNotFoundError:
    DATASET_PATH = None
    RF_SPLIT_RATIO = 0.2
    RF_RANDOM_STATE = 42
    RF_MODEL_SAVE_PATH = None
    RF_THRESHOLD_SAVE_PATH = None
    RF_TUNED_MODEL_SAVE_PATH = None
    RF_TUNED_THRESHOLD_SAVE_PATH = None
    RF_TUNED_CV_RESULTS_PATH = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split


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

    # Drop the Class and Time column for training/testing data respectively
    X = df.drop(
        ["Class", "Time"], axis=1
    )  # dropping the Time column, seems to yield no change, meaning Time
    # features are not influencing the model
    y = df["Class"]

    print("Splitting data into Train and Validation sets...")
    return train_test_split(
        X, y, test_size=RF_SPLIT_RATIO, random_state=RF_RANDOM_STATE, stratify=y
    )


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
        f"\n--- Best threshold: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f}) ---"
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
    Returns the paths for saving model artifacts based on the tuning status and
    output directory.

    Args:
        tuned: Whether the model is tuned or not.
        output_dir: The directory to save the artifacts. If None, uses the
                    default paths.

    Returns:
        tuple: The paths for the model, scaler, threshold, and CV results.
    """

    prefix = "rf_tuned" if tuned else "rf_simple"
    if output_dir:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        return (
            base / f"{prefix}_model.pkl",
            base / f"{prefix}_threshold.pkl",
            base / f"{prefix}_cv_results.pkl",
        )
    return (
        RF_TUNED_MODEL_SAVE_PATH if tuned else RF_MODEL_SAVE_PATH,
        RF_TUNED_THRESHOLD_SAVE_PATH if tuned else RF_THRESHOLD_SAVE_PATH,
        RF_TUNED_CV_RESULTS_PATH,
    )


def save(model, threshold, tuned, output_dir=None):
    """
    Save the trained model, scaler, and threshold to disk.

    Args:
        model: The trained model.
        scaler: The fitted scaler.
        threshold: The threshold for classification.
        tuned: Whether the model was tuned or not.
        output_dir: The directory to save the artifacts to.
    """

    model_path, threshold_path, _ = _artifact_paths(tuned, output_dir)
    assert isinstance(model_path, Path)
    assert isinstance(threshold_path, Path)

    print(f"\nSaving artifacts to {model_path.parent} ...")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(threshold_path, "wb") as f:
        pickle.dump(threshold, f)

    print("Artifacts saved successfully.")


def train_simple(X_train, y_train):
    """
    Train a simple Random Forest model with default parameters for quick
    training and evaluation.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        model: The trained Random Forest model.
    """

    print("Training Random Forest (Simple) ...")
    model = RandomForestClassifier(
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
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
    Train a tuned Random Forest model using GridSearchCV.

    Args:
        X_train: The training features.
        y_train: The training labels.
        output_dir: The directory to save the artifacts to.
        scoring: The scoring metric to use for tuning.

    Returns:
        The trained model.
    """

    print("Training Random Forest with GridSearchCV...")

    # GridSearchCV with parameters gathered from tuning with RandomizedSearchCV
    # to keep computation time manageable
    #
    # Change to False and modify param_grid of else branch to tune further with
    # RandomizedSearchCV
    fixed_params_enabled = True

    if fixed_params_enabled:
        print("Using fixed params for Random Forest; cv_results not updated!")

        # here we used grid search before, but noticed computation time was too
        # long -> switched to fixed params and RandomForestClassifier.fit() for
        # finding the best hyperparameters
        model = RandomForestClassifier(
            n_estimators=500,  # number of trees
            min_samples_split=2,  # minimum number of samples required to split a node
            min_samples_leaf=1,  # minimum number of samples required at a leaf node
            max_features="log2",  # number of features to consider when looking for the best split
            max_depth=None,  # maximum depth of the tree
            class_weight="balanced",  # class weight to handle class imbalance
            random_state=RF_RANDOM_STATE,  # random state for reproducibility
            n_jobs=-1,  # use all available CPU cores
            verbose=1,  # verbose output during training
        )

        model.fit(X_train, y_train)

        return model
    else:
        param_grid = {
            "n_estimators": [300, 500],  # removed 50 and 100 trees for now
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "max_features": ["log2"],
            "max_depth": [None],
            "class_weight": ["balanced"],
            # --- Previously tuned ---
            # "n_estimators": [50, 100, 150, 200],  # removed 50 and 100 trees for now, added 300
            # "min_samples_split": [2, 5, 10],  # Previous tuning revealed min_samples_split=2 is best
            # "min_samples_leaf": [1, 2],  # Previous tuning revealed min_samples_leaf=1 is best
            # "max_depth": [5, 10, 15, 20, None],  # Previous tuning revealed max_depth of None is best
            # "max_features": ["log2", "sqrt"],  # Previous tuning revealed "log2" is best
            # "class_weight": ["balanced", None],  # Previous tuning revealed "balanced" is best
            # "max_samples": [None, 0.7, 0.8],  # Previous tuning revealed None is best
        }

        # Use RandomizedSearchCV to test 50 random combinations of hyperparameters
        # This is done to keep computation time reasonable, while still
        # exploring a wide range of hyperparameters, by drawing 50 random
        # samples from the parameter space.
        grid = RandomizedSearchCV(
            RandomForestClassifier(random_state=RF_RANDOM_STATE, n_jobs=-1),
            param_grid,
            n_iter=50,  # Test 50 random combinations
            scoring=scoring,
            cv=3,
            n_jobs=-1,
            random_state=RF_RANDOM_STATE,
            verbose=2,
        )

    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"CV score: {grid.best_score_:.4f}")

    if create_cv_results:
        _, _, cv_results_path = _artifact_paths(tuned=True, output_dir=output_dir)
        assert isinstance(cv_results_path, Path)
        with open(cv_results_path, "wb") as f:
            pickle.dump(grid.cv_results_, f)

    return grid.best_estimator_


def run(
    create_cv_results,
    scoring,
    tuned,
    dataset_path=None,
    output_dir=None,
):
    """
    Run Random Forest training.

    Args:
        tuned (bool): whether to use tuned hyperparameters
        dataset_path (str): path to the dataset
        output_dir (str): directory to save artifacts
        create_cv_results (bool): whether to save cross-validation results
        scoring (str): scoring metric to use for hyperparameter tuning
    """

    variant = "tuned" if tuned else "simple"
    print(f"Running Random Forest training ({variant})...")

    X_train, X_val, y_train, y_val = process_dataset(dataset_path)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)

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
    save(model, threshold, tuned, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Random Forest fraud detection model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset. Defaults to config path.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_false",
        default=True,  # this is inverted, so default (True) means tuning is enabled
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
