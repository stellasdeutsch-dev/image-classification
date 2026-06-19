#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1            # gpu:2 on dual-GPU servers
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Train the classifier on the GPU, then predict + evaluate.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate imgcls

# Optional: stage dataset onto fast local NVMe (NOT backed up)
# rsync -a /shares/datasets/<your-set>/ /fastlocal/$USER/dataset/

srun python -m src.train   --config configs/train.yaml
srun python -m src.predict --config configs/predict.yaml
srun python -m eval.evaluate --predictions data/run/predictions.parquet \
     --classes data/run/classes.json --plot
