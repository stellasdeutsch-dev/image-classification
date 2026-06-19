# CLAUDE.md — build guide for this repo

This repo implements **image classification with a deep focus on error analysis**.
Full plan in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Conventions
- Python ≥3.10. Code in `src/`, configs in `configs/` (YAML), tests in `tests/`.
- Stages: `data` → `train` → `predict` → `eval` / `error_analysis`. Each heavy
  stage has a CLI: `python -m src.<stage> --config configs/<stage>.yaml`.
- `src/metrics.py` and `src/error_analysis.py` are **pure NumPy** (no sklearn/torch)
  and must stay testable without the ML stack.
- Predictions contract (parquet): `path, y_true, y_pred, confidence` + `classes.json`.
- Config-driven; handle class imbalance via the weighted-loss option.

## Where things run
- **Training/prediction (torch/timm) run on the uni GPU via SLURM** (`slurm/`).
  Local Mac = code + the synthetic-prediction smoke + unit tests + the explorer.
- Loop: edit on Mac → `pytest -q` → push → pull on GPU → `sbatch slurm/train.slurm`
  → monitor (`squeue`, W&B) → copy `data/run` back → `make explorer` + error analysis.

## Don't commit
datasets, checkpoints, predictions, runs, `.env` — see `.gitignore`.
