"""Explicit inter-stage data contracts.

Column-presence AND value checks so a malformed artifact fails fast at the
producing stage with a clear message instead of a downstream KeyError or silently
wrong metrics. The column-set constants are the single source of truth for the
predictions table and the persisted test-split manifest."""

from __future__ import annotations

import numpy as np

PREDICTION_COLUMNS = ["path", "y_true", "y_pred", "confidence"]
TEST_MANIFEST_COLUMNS = ["path", "y_true"]


def missing_columns(present, required) -> list:
    """Pure helper: which required columns are absent from `present`."""
    have = set(present)
    return [c for c in required if c not in have]


def validate_columns(df, required, name: str):
    miss = missing_columns(list(df.columns), required)
    if miss:
        raise ValueError(f"{name} is missing columns {miss}; has {list(df.columns)}")
    return df


def validate_predictions(df):
    """Predictions must have non-negative integer labels and finite confidences
    in [0, 1] (so the metrics never silently consume out-of-range values)."""
    validate_columns(df, PREDICTION_COLUMNS, "predictions")
    if len(df):
        if (df["y_true"].to_numpy() < 0).any() or (df["y_pred"].to_numpy() < 0).any():
            raise ValueError("predictions have negative class labels")
        conf = df["confidence"].to_numpy(dtype=float)
        if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
            raise ValueError("predictions have confidences outside [0, 1] or non-finite")
    return df


def validate_test_manifest(df):
    return validate_columns(df, TEST_MANIFEST_COLUMNS, "test_manifest")
