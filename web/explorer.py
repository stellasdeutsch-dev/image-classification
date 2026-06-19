"""Streamlit error explorer: confusion matrix, per-class metrics, most-confused
pairs, and a gallery of the model's most confident mistakes.

Run:  streamlit run web/explorer.py
Point at a run dir:  IMGCLS_OUT=data/run streamlit run web/explorer.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.error_analysis import most_confused_pairs, top_misclassified  # noqa: E402
from src.metrics import classification_report, confusion_matrix  # noqa: E402

OUT = Path(os.environ.get("IMGCLS_OUT", "data/run"))

st.set_page_config(page_title="Error Explorer", layout="wide")
st.title("🧪 Image Classification — Error Explorer")
st.caption(f"Reading: `{OUT}`")

pred_path = OUT / "predictions.parquet"
classes_path = OUT / "classes.json"
if not pred_path.exists() or not classes_path.exists():
    st.warning(f"Need {pred_path} and {classes_path}. Run predict (or `make sample-preds`) first.")
    st.stop()

df = pd.read_parquet(pred_path)
classes = json.loads(classes_path.read_text())
y_true, y_pred, conf = df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["confidence"].to_numpy()
cm = confusion_matrix(y_true, y_pred, len(classes))
report = classification_report(cm, classes)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{report['accuracy']:.3f}")
c2.metric("Macro F1", f"{report['macro_f1']:.3f}")
c3.metric("Samples", report["n_samples"])

st.subheader("Per-class metrics")
st.dataframe(pd.DataFrame(report["per_class"]).T, use_container_width=True)

col_a, col_b = st.columns(2)
col_a.subheader("Confusion matrix (row-normalized)")
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
col_a.dataframe(pd.DataFrame(cm_norm.round(2), index=classes, columns=classes), use_container_width=True)

col_b.subheader("Most confused pairs")
pairs = [{"true": classes[i], "pred": classes[j], "count": c} for i, j, c in most_confused_pairs(cm, 10)]
col_b.dataframe(pd.DataFrame(pairs), use_container_width=True)

st.subheader("Most confident mistakes")
idx = top_misclassified(y_true, y_pred, conf, k=12)
if len(idx) == 0:
    st.success("No misclassifications 🎉")
else:
    cols = st.columns(6)
    for n, i in enumerate(idx):
        row = df.iloc[int(i)]
        cap = f"{classes[int(row.y_true)]} → {classes[int(row.y_pred)]} ({row.confidence:.2f})"
        with cols[n % 6]:
            if Path(row.path).exists():
                st.image(row.path, caption=cap, use_container_width=True)
            else:
                st.write(cap)
