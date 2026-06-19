"""Evaluate predictions: classification report + confusion matrix + the error
breakdown (most-confused pairs, weakest classes, calibration).

Usage:  python -m eval.evaluate --predictions data/run/predictions.parquet --classes data/run/classes.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import load_json, save_json, setup_logging
from src.error_analysis import calibration_bins, most_confused_pairs, per_class_error_rate
from src.metrics import classification_report, confusion_matrix

log = logging.getLogger("imgcls")


def evaluate(predictions_path: str, classes_path: str, out_dir: str = "data/run", plot: bool = False) -> dict:
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(predictions_path)
    class_names = load_json(classes_path)
    n = len(class_names)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    conf = df["confidence"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, n)
    report = classification_report(cm, class_names)
    err_rate = per_class_error_rate(cm)
    _, ece = calibration_bins(conf, (y_true == y_pred).astype(float))

    report["expected_calibration_error"] = ece
    report["most_confused"] = [
        {"true": class_names[i], "pred": class_names[j], "count": c}
        for i, j, c in most_confused_pairs(cm, k=10)
    ]
    report["worst_classes"] = [
        {"class": class_names[i], "error_rate": round(float(err_rate[i]), 4)}
        for i in np.argsort(-err_rate)[:5]
    ]

    save_json(report, Path(out_dir) / "report.json")
    save_json(cm.tolist(), Path(out_dir) / "confusion_matrix.json")
    if plot:
        from src.error_analysis import plot_confusion
        try:
            plot_confusion(cm, class_names, str(Path(out_dir) / "confusion.png"))
        except Exception as e:  # pragma: no cover
            log.warning("confusion plot skipped (%s)", e)

    log.info("accuracy=%.4f  macro_f1=%.4f  ECE=%.4f", report["accuracy"], report["macro_f1"], ece)
    return report


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Evaluate predictions + error breakdown.")
    ap.add_argument("--predictions", default="data/run/predictions.parquet")
    ap.add_argument("--classes", default="data/run/classes.json")
    ap.add_argument("--out-dir", default="data/run")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    report = evaluate(args.predictions, args.classes, args.out_dir, args.plot)
    import json
    print(json.dumps({k: report[k] for k in ("accuracy", "macro_f1", "weighted_f1",
                                             "expected_calibration_error")}, indent=2))


if __name__ == "__main__":
    main()
