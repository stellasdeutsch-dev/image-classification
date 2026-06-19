# Image Classification with Error Analysis

> Classify images on a custom dataset, then rigorously analyze the errors — confusion, most-confident mistakes, confused pairs, per-class weakness, calibration, Grad-CAM.

This document is **both** the build brief (hand each phase to Claude Code) **and** the portfolio writeup.

---

## 1. What it does

Train a transfer-learning classifier on a folder-per-class dataset, predict on a held-out test split, then produce a quantitative **error-analysis report** + an interactive explorer. The differentiator vs a tutorial is the depth of the error analysis and the engineering around it (reproducible, imbalance-aware, tested).

## 2. Why this is a strong AI / Data Engineer project (CV signal)

- **Reproducible pipeline** — config-driven stages, fixed seeds, deterministic splits.
- **Handles real-world data** — class imbalance via weighted loss; any folder-per-class dataset.
- **Rigorous evaluation** — pure-NumPy metrics (no sklearn black box), confusion matrix, per-class P/R/F1, **calibration/ECE**.
- **Deep error analysis** — most-confident mistakes, most-confused pairs, weakest classes, Grad-CAM (is the model looking at the object?). This is the part interviewers probe.
- **Tested + CI** — the analysis core has unit tests that run on every push.

## 3. Pipeline

```
  data/dataset/<class>/*  ──►  train (timm, GPU)  ──►  best.pt
        ──►  predict (test split)  ──►  predictions.parquet [path, y_true, y_pred, confidence]
        ──►  eval + error_analysis  ──►  report.json · confusion · calibration · most-confused
        ──►  Streamlit error explorer
```

## 4. Tech stack (chosen, with rationale)

| Layer | Choice | Why |
|---|---|---|
| Backbone | **timm** (ResNet/ConvNeXt/ViT) | Huge model zoo, transfer learning, one-line swap. |
| Training | **PyTorch** | Standard; explicit loop = full control (imbalance, logging). |
| Metrics | **pure NumPy** | Transparent + testable; no sklearn dependency. |
| Explorer | **Streamlit** | Fast interactive failure browsing. |
| Tracking | **Weights & Biases** | Log loss / val macro-F1 per epoch. |
| Tooling | `pip`, `ruff`, `pytest`, YAML | Reproducible, config-driven. |

## 5. Repo structure

```
configs/   data.yaml · train.yaml · predict.yaml
src/       data · model · train · predict · metrics · error_analysis · gradcam · common
web/       explorer.py (Streamlit)
slurm/     hello · train
eval/      evaluate.py (report + error breakdown)
tests/     test_metrics · test_error_analysis
scripts/   make_sample_dataset · make_sample_predictions
notebooks/ error_analysis.ipynb
```

## 6. Infrastructure & workflow (Mac M3 + ITEC GPU)

- Edit + test on the Mac (`pytest`, the synthetic-prediction smoke, the explorer).
- Train on the GPU via SLURM: `ssh -p 2022 USER@gpu6.itec.aau.at`, `conda activate imgcls`, `sbatch slurm/train.slurm`.
- Data on `/shares/datasets`; stage to `/fastlocal/$USER` (not backed up → copy `data/run` back to `~`).
- Loop: edit → `pytest` → push → pull on GPU → `sbatch` → monitor (`squeue`, W&B) → copy `data/run` back → error analysis.

## 7. SLURM job template

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate imgcls
srun python -m src.train   --config configs/train.yaml
srun python -m src.predict --config configs/predict.yaml
srun python -m eval.evaluate --predictions data/run/predictions.parquet --classes data/run/classes.json --plot
```

## 8. Dataset options (pick a domain)

| Domain | Idea | Error-analysis angle |
|---|---|---|
| **Defect detection** | parts: normal vs scratch/chip | imbalance, rare-class recall, Grad-CAM on the defect |
| **Land use** | satellite tiles: forest/water/urban/field | borderline classes (field vs meadow), season/lighting |
| **Document types** | scans: invoice/form/receipt/ID | scan-quality effect, visually similar classes |
| **Audio** | mel-spectrograms of environmental sounds | which sounds confuse, clip length / noise |

## 9. Phased execution plan

- **Phase 0 — Scaffold** (local): repo, env, configs, CI, W&B; `make test` green, hello GPU job runs.
- **Phase 1 — Data** (local): assemble folder-per-class dataset; class counts + imbalance check. Done when `make sample` (or your data) loads and class weights compute.
- **Phase 2 — Train** (GPU): transfer-learn a backbone, weighted loss, track val macro-F1, checkpoint best. Done when a model trains with a logged curve.
- **Phase 3 — Predict** (GPU): write `predictions.parquet` on the test split. Done when predictions exist with paths + confidences.
- **Phase 4 — Evaluate** (local): classification report, confusion matrix, calibration/ECE. Done when `report.json` is produced.
- **Phase 5 — Error analysis** (local): most-confident mistakes, confused pairs, weakest classes, Grad-CAM; the explorer. Done when you can list failure categories with examples.
- **Phase 6 — Iterate**: act on findings (rebalance, clean labels, augment, bigger model) and measure the lift. Done when a change is shown to move macro-F1/calibration and you know why.
- **Phase 7 — Ship**: README with results table, confusion image, Grad-CAM examples, and the written error-analysis story.

## 10. Evaluation & metrics

Accuracy, macro/weighted F1, per-class P/R/F1, per-class error rate, **ECE** (calibration), confusion matrix. Report which intervention moved the metric and why.

## 11. Stretch goals

Cross-validation + CIs, TTA, mixup/label-smoothing (+ recalibration), active learning on confident mistakes, ONNX export.

## 12. Deliverables checklist (for the CV)

- [ ] Public repo, clean README, reproducible `make` targets, green CI.
- [ ] Results table + confusion image + Grad-CAM examples.
- [ ] **Written error-analysis section** (the headline): failure taxonomy → fix → measured lift.
- [ ] Résumé bullet: *"Trained an imbalance-aware image classifier (timm), macro-F1 X; error analysis (confusion/calibration/Grad-CAM) drove a +Y point lift by Z."*

## 13. Quickstart commands

```bash
make setup
make smoke && make explorer          # local error analysis on synthetic predictions
# real run on the cluster:
sbatch slurm/train.slurm
```

---

*Build order: Phase 0 → 5 to get the analysis loop working, then 6 → 7 to make it portfolio-grade. Each phase is self-contained.*
