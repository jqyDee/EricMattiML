import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import (
    DATASET_PATH,
    RF_MODEL_SAVE_PATH,
    RF_PLOTS_PATH,
    RF_RANDOM_STATE,
    RF_SPLIT_RATIO,
    RF_THRESHOLD_SAVE_PATH,
    RF_TUNED_MODEL_SAVE_PATH,
    RF_TUNED_THRESHOLD_SAVE_PATH,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def _load_val_set():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=RF_SPLIT_RATIO, random_state=RF_RANDOM_STATE, stratify=y
    )
    return X_val, y_val


def _load_artifacts(tuned):
    model = joblib.load(RF_TUNED_MODEL_SAVE_PATH if tuned else RF_MODEL_SAVE_PATH)
    threshold = joblib.load(
        RF_TUNED_THRESHOLD_SAVE_PATH if tuned else RF_THRESHOLD_SAVE_PATH
    )
    return model, threshold


def plot_pr_curves():
    X_val, y_val = _load_val_set()

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, tuned in [
        ("Simple (100 trees)", False),
        ("Tuned (500 trees, log2, balanced)", True),
    ]:
        path = RF_TUNED_MODEL_SAVE_PATH if tuned else RF_MODEL_SAVE_PATH
        if not os.path.exists(path):
            print(f"Skipping {label}: artifacts not found.")
            continue

        model, threshold = _load_artifacts(tuned)
        proba = model.predict_proba(X_val)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        auprc = average_precision_score(y_val, proba)
        roc_auc = roc_auc_score(y_val, proba)

        (line,) = ax.plot(
            recall, precision, label=f"{label}  (AUPRC={auprc:.4f}, ROC-AUC={roc_auc:.4f})", lw=1.5
        )

        op_idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(
            recall[op_idx], precision[op_idx], color=line.get_color(), s=80, zorder=5
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Simple vs Tuned RF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = RF_PLOTS_PATH / "rf_pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_confusion_matrices():
    X_val, y_val = _load_val_set()

    models_to_plot = []
    for label, tuned in [("Simple", False), ("Tuned", True)]:
        path = RF_TUNED_MODEL_SAVE_PATH if tuned else RF_MODEL_SAVE_PATH
        if not os.path.exists(path):
            print(f"Skipping {label}: artifacts not found.")
            continue
        model, threshold = _load_artifacts(tuned)
        proba = model.predict_proba(X_val)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        models_to_plot.append((label, cm, threshold))

    if not models_to_plot:
        print("No artifacts found.")
        return

    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(10, 5))
    if len(models_to_plot) == 1:
        axes = [axes]

    for ax, (label, cm, threshold) in zip(axes, models_to_plot):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=ax,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(f"Confusion Matrix — {label}\n(threshold={threshold:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    out = RF_PLOTS_PATH / "rf_confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_feature_importance():
    path = RF_TUNED_MODEL_SAVE_PATH
    if not os.path.exists(path):
        print("Skipping feature importance: tuned model not found.")
        return

    model, _ = _load_artifacts(tuned=True)
    importances = model.feature_importances_

    df = pd.read_csv(DATASET_PATH)
    feature_names = df.drop(["Class", "Time"], axis=1).columns.tolist()

    indices = np.argsort(importances)[::-1]
    top_n = 20

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices[:top_n]], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] for i in indices[:top_n]], rotation=45, ha="right"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance (mean impurity decrease)")
    ax.set_title(f"RF Feature Importance (Tuned Model) — Top {top_n}")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = RF_PLOTS_PATH / "rf_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_threshold_analysis():
    X_val, y_val = _load_val_set()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (label, tuned) in zip(axes, [("Simple", False), ("Tuned", True)]):
        path = RF_TUNED_MODEL_SAVE_PATH if tuned else RF_MODEL_SAVE_PATH
        if not os.path.exists(path):
            ax.set_visible(False)
            continue

        model, threshold = _load_artifacts(tuned)
        proba = model.predict_proba(X_val)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
        f1s = np.where(
            (precisions[:-1] + recalls[:-1]) == 0,
            0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        )

        ax.plot(thresholds, precisions[:-1], label="Precision", lw=1)
        ax.plot(thresholds, recalls[:-1], label="Recall", lw=1)
        ax.plot(thresholds, f1s, label="F1", lw=1.5, color="black")
        ax.axvline(
            threshold, color="red", linestyle="--", label=f"Chosen ({threshold:.4f})"
        )

        ax.set_title(f"Threshold Analysis — {label}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.05)

    fig.tight_layout()
    out = RF_PLOTS_PATH / "rf_threshold_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def visualize():
    RF_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    print("Generating RF visualizations...")
    plot_pr_curves()
    plot_confusion_matrices()
    plot_feature_importance()
    plot_threshold_analysis()
    print(f"\nAll plots saved to {RF_PLOTS_PATH}")
