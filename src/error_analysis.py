"""Error analysis — the heart of the project. Pure-NumPy functions over
prediction arrays so the analysis is testable and reproducible:

- most-confident mistakes (where the model is confidently wrong)
- most-confused class pairs (off-diagonal of the confusion matrix)
- per-class error rate (find the weak classes)
- confidence calibration (reliability bins)

Plotting helpers (confusion matrix, misclassified gallery) are matplotlib and
guarded so the core stays import-light."""

from __future__ import annotations

import numpy as np


def top_misclassified(y_true, y_pred, confidence, k: int = 20) -> np.ndarray:
    """Indices of wrong predictions, sorted by confidence descending
    (the model's most confident mistakes — the most informative to inspect)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    wrong = np.flatnonzero(y_true != y_pred)
    order = wrong[np.argsort(-confidence[wrong])]
    return order[:k]


def most_confused_pairs(cm: np.ndarray, k: int = 10) -> list[tuple[int, int, int]]:
    """Largest off-diagonal confusion entries as (true, pred, count)."""
    cm = np.asarray(cm, dtype=int)
    n = cm.shape[0]
    pairs = [(i, j, int(cm[i, j])) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda x: -x[2])[:k]


def per_class_error_rate(cm: np.ndarray) -> np.ndarray:
    """Fraction of each true class that is misclassified."""
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    errors = support - np.diag(cm)
    return errors / np.maximum(support, 1.0)


def calibration_bins(confidence, correct, n_bins: int = 10):
    """Reliability data: per-bin (mean_confidence, accuracy, count) plus ECE.
    `correct` is a 0/1 array of whether each prediction was right."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece_sum, n_binned = [], 0.0, 0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence >= lo) & (confidence <= hi)
        cnt = int(mask.sum())
        if cnt:
            conf_mean = float(confidence[mask].mean())
            acc = float(correct[mask].mean())
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "confidence": round(conf_mean, 4),
                         "accuracy": round(acc, 4), "count": cnt})
            ece_sum += cnt * abs(acc - conf_mean)
            n_binned += cnt
    # weight by the samples actually binned (in-range), not the raw input length
    ece = ece_sum / n_binned if n_binned else 0.0
    return bins, round(float(ece), 4)


# ---------- plotting (optional) ----------

def plot_confusion(cm: np.ndarray, class_names: list[str], path: str, normalize: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(1.2 + len(class_names), 1.0 + len(class_names)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > (0.5 if normalize else cm.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
