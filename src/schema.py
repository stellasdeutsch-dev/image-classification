"""Explicit inter-stage data contracts.

Lightweight, dependency-free column checks so a malformed artifact fails fast at
the producing stage with a clear message instead of a downstream KeyError. The
column-set constants are the single source of truth for the predictions table and
the persisted test-split manifest."""

from __future__ import annotations

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
    return validate_columns(df, PREDICTION_COLUMNS, "predictions")


def validate_test_manifest(df):
    return validate_columns(df, TEST_MANIFEST_COLUMNS, "test_manifest")
