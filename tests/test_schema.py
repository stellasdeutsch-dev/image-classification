"""Data-contract validators — column presence AND value checks."""

import pandas as pd
import pytest

from src.schema import (
    PREDICTION_COLUMNS,
    missing_columns,
    validate_predictions,
    validate_test_manifest,
)


def _preds(n=3):
    return pd.DataFrame({"path": [f"{i}.png" for i in range(n)], "y_true": list(range(n)),
                         "y_pred": list(range(n)), "confidence": [0.9, 0.8, 0.7][:n]})


def test_missing_columns_helper():
    assert missing_columns(["path", "y_true"], PREDICTION_COLUMNS) == ["y_pred", "confidence"]


def test_predictions_ok_and_missing_column():
    validate_predictions(_preds())
    with pytest.raises(ValueError):
        validate_predictions(_preds().drop(columns=["confidence"]))


def test_predictions_confidence_out_of_range_rejected():
    p = _preds()
    p.loc[0, "confidence"] = 1.5
    with pytest.raises(ValueError):
        validate_predictions(p)


def test_predictions_negative_label_rejected():
    p = _preds()
    p.loc[0, "y_pred"] = -1
    with pytest.raises(ValueError):
        validate_predictions(p)


def test_test_manifest():
    validate_test_manifest(pd.DataFrame({"path": ["a"], "y_true": [0]}))
    with pytest.raises(ValueError):
        validate_test_manifest(pd.DataFrame({"path": ["a"]}))
