"""Data-contract validators — pure (duck-typed on .columns, no pandas needed)."""

import pytest

from src.schema import (
    PREDICTION_COLUMNS,
    TEST_MANIFEST_COLUMNS,
    missing_columns,
    validate_predictions,
    validate_test_manifest,
)


class _DF:
    def __init__(self, cols):
        self.columns = cols


def test_missing_columns():
    assert missing_columns(["path", "y_true"], PREDICTION_COLUMNS) == ["y_pred", "confidence"]
    assert missing_columns(PREDICTION_COLUMNS, PREDICTION_COLUMNS) == []


def test_validate_predictions():
    validate_predictions(_DF(PREDICTION_COLUMNS))              # ok
    with pytest.raises(ValueError):
        validate_predictions(_DF(["path", "y_true"]))


def test_validate_test_manifest():
    validate_test_manifest(_DF(TEST_MANIFEST_COLUMNS))         # ok
    with pytest.raises(ValueError):
        validate_test_manifest(_DF(["path"]))
