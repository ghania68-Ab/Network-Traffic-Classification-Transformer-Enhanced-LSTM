"""Evaluation and visualization utilities for the proposed model only."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

MODEL_NAME = "Transformer-Enhanced LSTM"
<<<<<<< HEAD


def _safe_auc(y_test: np.ndarray, probabilities: np.ndarray, average: str) -> float | None:
    """Return one-vs-rest multiclass AUC when all classes are present."""

    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_test, probabilities[:, 1]))
        return float(roc_auc_score(y_test, probabilities, multi_class="ovr", average=average))
    except ValueError:
        return None


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    tta_passes: int = 1,
    tta_noise: float = 0.004,
) -> dict[str, object]:
=======


def _safe_auc(y_test: np.ndarray, probabilities: np.ndarray, average: str) -> float | None:
    """Return one-vs-rest multiclass AUC when all classes are present."""

    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_test, probabilities[:, 1]))
        return float(roc_auc_score(y_test, probabilities, multi_class="ovr", average=average))
    except ValueError:
        return None


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> dict[str, object]:
>>>>>>> e38505b (Update project with latest changes)
    """Evaluate the proposed model on the held-out test split."""

    probabilities = model.predict(X_test, verbose=0)
    if tta_passes > 1:
        rng = np.random.default_rng(42)
        for _ in range(tta_passes - 1):
            noisy = X_test + rng.normal(0.0, tta_noise, X_test.shape).astype(np.float32)
            probabilities += model.predict(noisy, verbose=0)
        probabilities /= float(tta_passes)
    y_pred = np.argmax(probabilities, axis=1)
    labels = np.arange(len(class_names))

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )

    return {
        "model_name": MODEL_NAME,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_auc": _safe_auc(y_test, probabilities, average="weighted"),
        "macro_auc": _safe_auc(y_test, probabilities, average="macro"),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "y_pred": y_pred,
    }


def _distribution_lines(class_distribution: Mapping[str, Mapping[str, int]]) -> list[str]:
    lines = ["Class distribution:"]
    for split, counts in class_distribution.items():
        rendered = ", ".join(f"{label}: {count}" for label, count in counts.items())
        lines.append(f"- {split}: {rendered}")
    return lines


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Create a readable ASCII table for metrics files."""

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_line = "|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, widths)) + "|"
    table = [border, header_line, border]
    for row in rows:
        table.append("|" + "|".join(f" {str(value):<{width}} " for value, width in zip(row, widths)) + "|")
    table.append(border)
    return table


def _score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def save_metrics(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save proposed-model metrics for supplied dataset experiment(s)."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Dataset",
        "Model",
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted AUC",
        "Macro AUC",
    ]
    rows = []
    for experiment in experiments.values():
        result = experiment["metrics"]
        rows.append([
            str(experiment["dataset_name"]),
            MODEL_NAME,
            f"{result['accuracy']:.4f}",
            f"{result['weighted_precision']:.4f}",
            f"{result['weighted_recall']:.4f}",
            f"{result['weighted_f1']:.4f}",
            f"{result['macro_precision']:.4f}",
            f"{result['macro_recall']:.4f}",
            f"{result['macro_f1']:.4f}",
            _score(result["weighted_auc"]),
            _score(result["macro_auc"]),
        ])

    lines = [
        "Proposed Transformer-Enhanced LSTM Metrics",
        "",
        "Performance Summary",
        *_format_table(headers, rows),
    ]

    for experiment in experiments.values():
        result = experiment["metrics"]
        lines.extend(["", "=" * 100, f"Dataset: {experiment['dataset_name']}"])
        lines.append(f"Label column: {experiment['label_column']}")
        if "class_weight_mode" in experiment:
            lines.append(f"Class weight mode: {experiment['class_weight_mode']}")
        lines.extend(_distribution_lines(experiment["class_distribution"]))
        lines.extend([
            "",
            f"Classification Report: {MODEL_NAME}",
            str(result["classification_report"]),
        ])

    output.write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy_curve(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save training and validation accuracy curves."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 6), dpi=170)
    for experiment in experiments.values():
        values = experiment["history"].history
        epochs = range(1, len(values["accuracy"]) + 1)
        dataset_name = str(experiment["dataset_name"])
        plt.plot(epochs, values["accuracy"], label=f"{dataset_name} train")
        plt.plot(epochs, values["val_accuracy"], linestyle="--", label=f"{dataset_name} validation")

    plt.title("Transformer-Enhanced LSTM Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.30)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_loss_curve(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save training and validation loss curves."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 6), dpi=170)
    for experiment in experiments.values():
        values = experiment["history"].history
        epochs = range(1, len(values["loss"]) + 1)
        dataset_name = str(experiment["dataset_name"])
        plt.plot(epochs, values["loss"], label=f"{dataset_name} train")
        plt.plot(epochs, values["val_loss"], linestyle="--", label=f"{dataset_name} validation")

    plt.title("Transformer-Enhanced LSTM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.30)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save normalized confusion matrix/matrices for the proposed model."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = list(experiments.values())
    cols = 1 if len(panels) == 1 else 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.8 * rows), dpi=160)
    axes = np.atleast_1d(axes).ravel()

    for ax, experiment in zip(axes, panels):
        matrix = np.asarray(experiment["metrics"]["confusion_matrix"])
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(matrix, dtype=float)
        np.divide(matrix, row_sums, out=normalized, where=row_sums != 0)
        display = ConfusionMatrixDisplay(normalized, display_labels=experiment["class_names"])
        display.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title(f"{experiment['dataset_name']} - {MODEL_NAME}")
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Normalized Confusion Matrix", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

def plot_dataset_comparison(experiments: Mapping[str, Mapping[str, object]], output_path: str | Path) -> None:
    """Save proposed-model performance comparison across datasets."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels, accuracy, weighted_f1, macro_f1, auc_scores = [], [], [], [], []
    for experiment in experiments.values():
        result = experiment["metrics"]
        labels.append(str(experiment["dataset_name"]))
        accuracy.append(float(result["accuracy"]))
        weighted_f1.append(float(result["weighted_f1"]))
        macro_f1.append(float(result["macro_f1"]))
        auc_value = result.get("weighted_auc")
        auc_scores.append(np.nan if auc_value is None else float(auc_value))

    x = np.arange(len(labels))
    width = 0.20
    plt.figure(figsize=(10.5, 6), dpi=170)
    bars1 = plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
    bars2 = plt.bar(x - 0.5 * width, weighted_f1, width, label="Weighted F1")
    bars3 = plt.bar(x + 0.5 * width, macro_f1, width, label="Macro F1")
    bars4 = plt.bar(x + 1.5 * width, auc_scores, width, label="Weighted AUC")

    for bars in (bars1, bars2, bars3, bars4):
        plt.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    plt.title("Performance Comparison of Proposed Model Across Datasets")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()