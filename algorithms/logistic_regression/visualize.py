import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import (
    DATASET_PATH,
    LR_MODEL_SAVE_PATH,
    LR_PLOTS_PATH,
    LR_RANDOM_STATE,
    LR_SCALER_SAVE_PATH,
    LR_SPLIT_RATIO,
    LR_THRESHOLD_SAVE_PATH,
    LR_TUNED_CV_RESULTS_PATH,
    LR_TUNED_MODEL_SAVE_PATH,
    LR_TUNED_SCALER_SAVE_PATH,
    LR_TUNED_THRESHOLD_SAVE_PATH,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def _load_val_set():
    """Reproduce the same train/val split used during training."""
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(["Class", "Time"], axis=1)  # Time dropped — no predictive signal
    y = df["Class"]
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=LR_SPLIT_RATIO, random_state=LR_RANDOM_STATE, stratify=y
    )
    return X_val, y_val


def _load_artifacts(tuned):
    model = joblib.load(LR_TUNED_MODEL_SAVE_PATH if tuned else LR_MODEL_SAVE_PATH)
    scaler = joblib.load(LR_TUNED_SCALER_SAVE_PATH if tuned else LR_SCALER_SAVE_PATH)
    threshold = joblib.load(
        LR_TUNED_THRESHOLD_SAVE_PATH if tuned else LR_THRESHOLD_SAVE_PATH
    )
    return model, scaler, threshold


def _apply_scaler(scaler, X_val):
    X = X_val.copy()
    X[["Amount"]] = scaler.transform(X[["Amount"]])  # scaler fit on Amount only
    return X


def plot_pr_curves():
    """PR curves for simple and tuned models overlaid."""
    X_val_raw, y_val = _load_val_set()

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, tuned in [
        ("Simple (L2, class_weight=balanced)", False),
        ("Tuned (L1, C=0.1)", True),
    ]:
        path = LR_TUNED_MODEL_SAVE_PATH if tuned else LR_MODEL_SAVE_PATH
        if not os.path.exists(path):
            print(f"Skipping {label}: artifacts not found.")
            continue

        model, scaler, threshold = _load_artifacts(tuned)
        X_val = _apply_scaler(scaler, X_val_raw)
        proba = model.predict_proba(X_val)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        auprc = average_precision_score(y_val, proba)
        roc_auc = roc_auc_score(y_val, proba)

        (line,) = ax.plot(
            recall,
            precision,
            label=f"{label}  (AUPRC={auprc:.4f}, ROC-AUC={roc_auc:.4f})",
            lw=1.5,
        )

        op_idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(
            recall[op_idx],
            precision[op_idx],
            color=line.get_color(),
            s=80,
            zorder=5,
            label=f"_op_{label}",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Simple vs Tuned LR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    LR_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = LR_PLOTS_PATH / "lr_pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_confusion_matrices():
    """Side-by-side confusion matrices for simple and tuned."""
    X_val_raw, y_val = _load_val_set()

    models_to_plot = []
    for label, tuned in [("Simple", False), ("Tuned", True)]:
        path = LR_TUNED_MODEL_SAVE_PATH if tuned else LR_MODEL_SAVE_PATH
        if not os.path.exists(path):
            print(f"Skipping {label}: artifacts not found.")
            continue
        model, scaler, threshold = _load_artifacts(tuned)
        X_val = _apply_scaler(scaler, X_val_raw)
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
            cmap="Blues",
            ax=ax,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(f"Confusion Matrix — {label}\n(threshold={threshold:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    out = LR_PLOTS_PATH / "lr_confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_gridsearch_results():
    """CV AUPRC per C × class_weight for L1 and L2 grids."""
    if not os.path.exists(LR_TUNED_CV_RESULTS_PATH):
        print("No CV results found. Run: python main.py --algo lr --mode train --tune")
        return

    with open(LR_TUNED_CV_RESULTS_PATH, "rb") as f:
        cv_data = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (key, title) in zip(
        axes, [("l2", "L2 regularization (lbfgs)"), ("l1", "L1 regularization (saga)")]
    ):
        results = cv_data[key]
        params = results["params"]
        scores = results["mean_test_score"]
        stds = results["std_test_score"]

        C_vals = sorted(set(p["C"] for p in params))
        cw_vals = sorted(set(str(p["class_weight"]) for p in params))
        x = np.arange(len(C_vals))
        width = 0.35

        colors = ["steelblue", "coral"]
        for i, cw in enumerate(cw_vals):
            means = []
            errs = []
            for C in C_vals:
                idx = next(
                    j
                    for j, p in enumerate(params)
                    if p["C"] == C and str(p["class_weight"]) == cw
                )
                means.append(scores[idx])
                errs.append(stds[idx])
            ax.bar(
                x + i * width,
                means,
                width,
                yerr=errs,
                label=f"class_weight={cw}",
                color=colors[i % len(colors)],
                capsize=4,
            )

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([str(c) for c in C_vals])
        ax.set_xlabel("C  (higher = less regularization)")
        ax.set_ylabel("CV AUPRC (mean ± std)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("GridSearch Results: C × class_weight", fontsize=13)
    fig.tight_layout()
    out = LR_PLOTS_PATH / "lr_gridsearch_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_threshold_analysis():
    """F1 / Precision / Recall vs threshold for both models."""
    X_val_raw, y_val = _load_val_set()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (label, tuned) in zip(axes, [("Simple", False), ("Tuned", True)]):
        path = LR_TUNED_MODEL_SAVE_PATH if tuned else LR_MODEL_SAVE_PATH
        if not os.path.exists(path):
            ax.set_visible(False)
            continue

        model, scaler, threshold = _load_artifacts(tuned)
        X_val = _apply_scaler(scaler, X_val_raw)
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
    out = LR_PLOTS_PATH / "lr_threshold_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def visualize():
    LR_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    print("Generating LR visualizations...")
    plot_pr_curves()
    plot_confusion_matrices()
    plot_gridsearch_results()
    plot_threshold_analysis()
    print(f"\nAll plots saved to {LR_PLOTS_PATH}")
