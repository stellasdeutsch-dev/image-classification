# 🧪 Image Classification with Error Analysis

> Train an image classifier on your own dataset — and then actually understand *how it fails*. The emphasis here is the **error analysis**: confusion matrices, most-confident mistakes, most-confused class pairs, per-class error rates, confidence calibration, and Grad-CAM.

<p>
  <a href="https://github.com/stellasdeutsch-dev/image-classification/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/stellasdeutsch-dev/image-classification/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-portfolio%20project-orange">
</p>

Anyone can call `model.fit`. This project is built around the part that separates an engineer from a tutorial: a reproducible pipeline, sensible handling of class imbalance, and a rigorous, quantitative **error analysis**. Training runs on a GPU cluster via **SLURM**; the metrics + error-analysis core is **pure NumPy** and runs (and is tested) anywhere.

📄 Full design & phased build plan: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**

---

## ✨ Features

- **Any dataset** — folder-per-class layout (`data/dataset/<class>/<image>`), so it fits defect detection, land-use, document types, etc.
- **Modern backbones** — `timm` (ResNet, ConvNeXt, ViT, …) with transfer learning.
- **Imbalance-aware** — inverse-frequency weighted loss out of the box.
- **Error analysis (the point):**
  - confusion matrix (+ row-normalized)
  - **most-confident mistakes** (where the model is confidently wrong)
  - **most-confused class pairs**
  - per-class precision / recall / F1 and error rate
  - **confidence calibration** (reliability bins + ECE)
  - **Grad-CAM** to check the model looks at the object, not the background
- **Interactive explorer** — Streamlit app to browse failures.
- **Tested** — pure-NumPy metrics/error-analysis with unit tests; CI on every push.

---

## 🏗️ Pipeline

```
  data/dataset/<class>/*  ──►  train (timm, GPU)  ──►  best.pt
                                      │
                                      ▼
                              predict (test split)  ──►  predictions.parquet [path, y_true, y_pred, confidence]
                                      │
                                      ▼
                          eval + error_analysis  ──►  report.json · confusion · calibration · most-confused
                                      │
                                      ▼
                              Streamlit error explorer
```

---

## 🚀 Quickstart (local error-analysis demo — no GPU, no training)

> Python **3.10–3.12**. This path uses synthetic predictions to exercise the whole metrics + error-analysis stack instantly.

```bash
git clone <your-repo-url> && cd image-classification
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

make sample-preds     # synthetic predictions with a built-in confusion pattern
make eval             # classification report + error breakdown -> data/run/report.json
make explorer         # Streamlit error explorer at http://localhost:8501
```

Run the tests (pure NumPy — no GPU/torch):

```bash
make test
```

To run the *real* model locally on a tiny synthetic dataset (CPU/MPS, slow):

```bash
make sample           # generate data/dataset/{circle,square,triangle}/
python -m src.train --config configs/train.yaml      # a few epochs on MPS for smoke
python -m src.predict --config configs/predict.yaml
make eval && make explorer
```

---

## 🖥️ On the GPU cluster (real dataset)

```bash
# point configs/*.yaml:data_root at your dataset, pick a model (e.g. convnext_tiny), then:
sbatch slurm/train.slurm     # train -> predict -> evaluate (+ confusion.png)
# copy data/run back to your Mac, then explore + write up the error analysis
```

See **[RUNBOOK.md](RUNBOOK.md)** for the full cluster walkthrough. Example domains: manufacturing **defect detection**, **land-use** (satellite), **document/receipt types**, or audio→spectrogram classification.

---

## 📁 Repo structure

```
configs/      data / train / predict YAML
src/          data · model (timm) · train · predict · metrics · error_analysis · gradcam · common
web/          Streamlit error explorer
slurm/        hello + train SLURM jobs (GPU)
eval/         report + confusion + calibration + most-confused
tests/        metrics & error-analysis unit tests (no GPU)
scripts/      synthetic dataset + synthetic predictions generators
notebooks/    error_analysis.ipynb (the deep dive)
PROJECT_PLAN.md
```

---

## 📊 Results

Fill in after training on your dataset:

| Dataset | Model | Accuracy | Macro F1 | Weakest class | ECE | Notes / fix |
|---|---|---|---|---|---|---|
| _e.g. defects_ | _convnext_tiny_ | _—_ | _—_ | _—_ | _—_ | _—_ |

---

## 🗺️ Roadmap

- [ ] Cross-validation + confidence intervals
- [ ] Test-time augmentation
- [ ] Mixup / label smoothing and re-measure calibration
- [ ] Active-learning loop on the most-confident mistakes
- [ ] Export to ONNX for serving

---

## 📜 License

MIT — see [LICENSE](LICENSE).
