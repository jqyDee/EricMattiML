import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import (
    COMPARE_PLOTS_PATH,
    DATASET_PATH,
    LR_MODEL_SAVE_PATH,
    LR_RANDOM_STATE,
    LR_SCALER_SAVE_PATH,
    LR_SPLIT_RATIO,
    LR_THRESHOLD_SAVE_PATH,
    LR_TUNED_MODEL_SAVE_PATH,
    LR_TUNED_SCALER_SAVE_PATH,
    LR_TUNED_THRESHOLD_SAVE_PATH,
    RF_MODEL_SAVE_PATH,
    RF_THRESHOLD_SAVE_PATH,
    RF_TUNED_MODEL_SAVE_PATH,
    RF_TUNED_THRESHOLD_SAVE_PATH,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def _load_val_set():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=LR_SPLIT_RATIO, random_state=LR_RANDOM_STATE, stratify=y
    )
    return X_val, y_val


def _load_all_models(X_val_raw):
    models = []

    configs = [
        (
            "LR Simple",
            LR_MODEL_SAVE_PATH,
            LR_SCALER_SAVE_PATH,
            LR_THRESHOLD_SAVE_PATH,
            True,
        ),
        (
            "LR Tuned",
            LR_TUNED_MODEL_SAVE_PATH,
            LR_TUNED_SCALER_SAVE_PATH,
            LR_TUNED_THRESHOLD_SAVE_PATH,
            True,
        ),
        ("RF Simple", RF_MODEL_SAVE_PATH, None, RF_THRESHOLD_SAVE_PATH, False),
        (
            "RF Tuned",
            RF_TUNED_MODEL_SAVE_PATH,
            None,
            RF_TUNED_THRESHOLD_SAVE_PATH,
            False,
        ),
    ]

    for label, model_path, scaler_path, threshold_path, needs_scaler in configs:
        if not os.path.exists(model_path):
            print(f"Skipping {label}: {model_path} not found.")
            continue

        model = joblib.load(model_path)
        threshold = joblib.load(threshold_path)

        X_val = X_val_raw.copy()
        if needs_scaler and scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_val[["Amount"]] = scaler.transform(X_val[["Amount"]])

        models.append((label, model, threshold, X_val))

    return models


def plot_pr_curves_combined():
    X_val_raw, y_val = _load_val_set()
    models = _load_all_models(X_val_raw)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors  = ["steelblue", "royalblue", "coral", "firebrick"]
    styles  = ["--", "-", "--", "-"]
    alphas  = [0.4, 1.0, 0.4, 1.0]

    for (label, model, threshold, X_val), color, style, alpha in zip(models, colors, styles, alphas):
        proba = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        auprc = average_precision_score(y_val, proba)

        (line,) = ax.plot(
            recall, precision,
            label=f"{label}  (AUPRC={auprc:.4f})",
            lw=1.5, color=color, linestyle=style, alpha=alpha,
        )

        op_idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(recall[op_idx], precision[op_idx], color=color, s=80, zorder=5, alpha=alpha)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — LR vs RF (Simple and Tuned)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = COMPARE_PLOTS_PATH / "compare_pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_metrics_comparison():
    X_val_raw, y_val = _load_val_set()
    models = _load_all_models(X_val_raw)

    labels, auprcs, roc_aucs, f1s = [], [], [], []

    for label, model, threshold, X_val in models:
        proba = model.predict_proba(X_val)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        labels.append(label)
        auprcs.append(average_precision_score(y_val, proba))
        roc_aucs.append(roc_auc_score(y_val, proba))
        f1s.append(f1_score(y_val, y_pred))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, auprcs, width, label="AUPRC", color="steelblue")
    ax.bar(x, roc_aucs, width, label="ROC-AUC", color="coral")
    ax.bar(x + width, f1s, width, label="F1", color="seagreen")

    for i, (a, r, f) in enumerate(zip(auprcs, roc_aucs, f1s)):
        ax.text(i - width, a + 0.002, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, r + 0.002, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width, f + 0.002, f"{f:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — AUPRC / ROC-AUC / F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = COMPARE_PLOTS_PATH / "compare_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_confusion_matrices_tuned():
    X_val_raw, y_val = _load_val_set()

    configs = [
        (
            "LR Tuned",
            LR_TUNED_MODEL_SAVE_PATH,
            LR_TUNED_SCALER_SAVE_PATH,
            LR_TUNED_THRESHOLD_SAVE_PATH,
            True,
        ),
        (
            "RF Tuned",
            RF_TUNED_MODEL_SAVE_PATH,
            None,
            RF_TUNED_THRESHOLD_SAVE_PATH,
            False,
        ),
    ]

    to_plot = []
    for label, model_path, scaler_path, threshold_path, needs_scaler in configs:
        if not os.path.exists(model_path):
            print(f"Skipping {label}: not found.")
            continue

        model = joblib.load(model_path)
        threshold = joblib.load(threshold_path)
        X_val = X_val_raw.copy()
        if needs_scaler and scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
            assert isinstance(X_val, pd.DataFrame)
            X_val[["Amount"]] = scaler.transform(X_val[["Amount"]])

        proba = model.predict_proba(X_val)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        to_plot.append((label, confusion_matrix(y_val, y_pred), threshold))

    if not to_plot:
        print("No tuned artifacts found.")
        return

    fig, axes = plt.subplots(1, len(to_plot), figsize=(10, 5))
    if len(to_plot) == 1:
        axes = [axes]

    cmaps = ["Blues", "Greens"]
    for ax, (label, cm, threshold), cmap in zip(axes, to_plot, cmaps):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(f"{label}\n(threshold={threshold:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Tuned Models", fontsize=13)
    fig.tight_layout()
    out = COMPARE_PLOTS_PATH / "compare_confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_roc_curves_combined():
    X_val_raw, y_val = _load_val_set()
    models = _load_all_models(X_val_raw)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["steelblue", "royalblue", "coral", "firebrick"]
    styles = ["--", "-", "--", "-"]
    alphas = [0.4, 1.0, 0.4, 1.0]

    for (label, model, threshold, X_val), color, style, alpha in zip(models, colors, styles, alphas):
        proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc = roc_auc_score(y_val, proba)

        ax.plot(
            fpr, tpr,
            label=f"{label}  (ROC-AUC={auc:.4f})",
            lw=1.5, color=color, linestyle=style, alpha=alpha,
        )

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — LR vs RF (Simple and Tuned)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = COMPARE_PLOTS_PATH / "compare_roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def compare():
    COMPARE_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    print("Generating comparison visualizations...")
    plot_pr_curves_combined()
    plot_roc_curves_combined()
    plot_metrics_comparison()
    plot_confusion_matrices_tuned()
    print(f"\nAll plots saved to {COMPARE_PLOTS_PATH}")
