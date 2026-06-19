CONFIG_DIR ?= configs

.PHONY: help setup sample sample-preds train predict eval explorer test lint smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package with all extras
	python -m pip install -e ".[all]"

sample:  ## Generate a synthetic folder-per-class image dataset
	python scripts/make_sample_dataset.py --out data/dataset --per-class 60

sample-preds:  ## Generate synthetic predictions (no training needed)
	python scripts/make_sample_predictions.py --out data/run

train:  ## Train the classifier (GPU)
	python -m src.train --config $(CONFIG_DIR)/train.yaml

predict:  ## Predict on the test split (GPU)
	python -m src.predict --config $(CONFIG_DIR)/predict.yaml

eval:  ## Evaluate predictions + error breakdown
	python -m eval.evaluate --predictions data/run/predictions.parquet --classes data/run/classes.json

explorer:  ## Launch the Streamlit error explorer
	IMGCLS_OUT=data/run streamlit run web/explorer.py

test:  ## Run unit tests (pure NumPy — no GPU)
	pytest -q

lint:  ## Lint with ruff
	ruff check .

smoke: sample-preds eval  ## Local error-analysis on synthetic predictions (no GPU)
	@echo "Smoke complete — try: make explorer"

clean:  ## Remove generated artifacts
	rm -rf data/run data/dataset
