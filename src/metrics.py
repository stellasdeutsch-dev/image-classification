"""Classification metrics — pure NumPy (no sklearn), so they're transparent and
testable: confusion matrix, per-class precision/recall/F1, accuracy, and a
report. Used by eval and error analysis."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, cols = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def per_class_prf(cm: np.ndarray):
    """Returns (precision, recall, f1) arrays, one entry per class."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def macro_f1(cm: np.ndarray) -> float:
    return float(per_class_prf(cm)[2].mean())


def classification_report(cm: np.ndarray, class_names: list[str]) -> dict:
    precision, recall, f1 = per_class_prf(cm)
    support = cm.sum(axis=1).astype(int)
    total = int(support.sum())
    per_class = {
        class_names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    w = support / max(total, 1)
    return {
        "accuracy": round(float(np.trace(cm) / max(total, 1)), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float((f1 * w).sum()), 4),
        "n_samples": total,
        "per_class": per_class,
    }
