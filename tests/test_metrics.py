"""Metrics — pure NumPy, no torch/sklearn."""

import numpy as np

from src.metrics import accuracy, classification_report, confusion_matrix, macro_f1, per_class_prf


def test_confusion_matrix_counts():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert cm.sum() == 6


def test_accuracy():
    assert accuracy([0, 1, 2, 2], [0, 1, 2, 0]) == 0.75
    assert accuracy([], []) == 0.0


def test_perfect_classifier():
    cm = confusion_matrix([0, 1, 2], [0, 1, 2], 3)
    p, r, f1 = per_class_prf(cm)
    assert np.allclose(p, 1.0) and np.allclose(r, 1.0) and np.allclose(f1, 1.0)
    assert macro_f1(cm) == 1.0


def test_precision_recall_values():
    # class 0: TP=1, FP=1 (one '2' predicted as 0) -> P=0.5; FN=1 -> R=0.5
    cm = confusion_matrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], 3)
    p, r, _ = per_class_prf(cm)
    assert abs(p[0] - 0.5) < 1e-6
    assert abs(r[0] - 0.5) < 1e-6


def test_report_structure():
    cm = confusion_matrix([0, 1, 1], [0, 1, 0], 2)
    rep = classification_report(cm, ["a", "b"])
    assert rep["n_samples"] == 3
    assert set(rep["per_class"]) == {"a", "b"}
    assert 0.0 <= rep["macro_f1"] <= 1.0
