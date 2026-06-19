#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate imgcls
srun python -c "import torch; print('CUDA:', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
