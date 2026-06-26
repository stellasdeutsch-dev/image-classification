"""Error-analysis helpers — pure NumPy."""

import numpy as np

from src.error_analysis import (
    calibration_bins,
    most_confused_pairs,
    per_class_error_rate,
    top_misclassified,
)
from src.metrics import confusion_matrix


def test_top_misclassified_orders_by_confidence():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([1, 1, 0, 2])      # wrong at idx 0, 2, 3
    conf = np.array([0.9, 0.5, 0.7, 0.99])
    idx = top_misclassified(y_true, y_pred, conf, k=2)
    assert idx.tolist() == [3, 0]        # most-confident mistakes first


def test_top_misclassified_none_when_perfect():
    idx = top_misclassified([0, 1], [0, 1], [0.9, 0.8], k=5)
    assert idx.tolist() == []


def test_most_confused_pairs():
    cm = confusion_matrix([0, 0, 1, 1, 1], [1, 1, 0, 0, 0], 2)  # 0->1 x2, 1->0 x3
    pairs = most_confused_pairs(cm, k=5)
    assert pairs[0] == (1, 0, 3)
    assert (0, 1, 2) in pairs


def test_per_class_error_rate():
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2)   # class0: 1 wrong/2; class1: 0/2
    er = per_class_error_rate(cm)
    assert abs(er[0] - 0.5) < 1e-6
    assert er[1] == 0.0


def test_calibration_ece():
    conf = np.array([0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 1, 0, 0])     # 0.9-bin perfectly calibrated-ish, 0.6-bin overconfident
    bins, ece = calibration_bins(conf, correct, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert sum(b["count"] for b in bins) == 4


def test_calibration_ece_weights_by_binned_count():
    # the 1.5 confidence is out of [0,1] -> excluded from bins; ECE must divide by
    # the binned count (2), not the raw input length (3).
    conf = np.array([0.9, 0.9, 1.5])
    correct = np.array([1, 1, 0])
    bins, ece = calibration_bins(conf, correct, n_bins=10)
    assert sum(b["count"] for b in bins) == 2
    assert abs(ece - 0.1) < 1e-9         # |1.0 - 0.9| over the 2 binned samples
