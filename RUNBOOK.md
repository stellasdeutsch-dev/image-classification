# Runbook — running Image Classification on the ITEC GPU cluster

Take it from scaffold to **real metrics + an error-analysis story in the README**.

## 0. One-time setup (on the cluster)
```bash
ssh -p 2022 USER@gpu6.itec.aau.at            # VPN required
conda create -n imgcls python=3.11 -y && conda activate imgcls
git clone https://github.com/stellasdeutsch-dev/image-classification ~/imgcls
cd ~/imgcls
pip install -r requirements.txt              # torch from the CUDA index if needed
mkdir -p logs
sbatch slurm/hello.slurm && squeue -u $USER  # confirm GPU
```

## 1. Prepare the dataset (folder-per-class)
```
data/dataset/
  <class_a>/img001.jpg ...
  <class_b>/...
```
- Use `/shares/datasets` if present, else copy your images in. Stage to NVMe for speed: `rsync -a /shares/datasets/<set>/ /fastlocal/$USER/dataset/` and set `configs/*.yaml:data_root` accordingly (NVMe not backed up).
- Check imbalance: counts per class drive the weighted loss (on by default).

## 2. Train + predict + evaluate (GPU)
```bash
# pick a backbone in configs/train.yaml (resnet50 / convnext_tiny / vit_small_patch16_224)
sbatch slurm/train.slurm                     # train -> predict -> evaluate (+ confusion.png)
squeue -u $USER                              # logs/train_<jobid>.out
```
Outputs in `data/run/`: `best.pt`, `predictions.parquet`, `classes.json`, `report.json`, `confusion_matrix.json`, `confusion.png`.

## 3. Bring results back & analyze (Mac)
```bash
rsync -avP -e "ssh -p 2022" USER@gpu6.itec.aau.at:~/imgcls/data/run ./data/
make explorer                                # browse failures in Streamlit
jupyter lab notebooks/error_analysis.ipynb   # deep dive
```

## 4. Iterate (the part that matters)
Read `report.json` → `worst_classes`, `most_confused`, `expected_calibration_error`. Then:
- rare class missed → stronger augmentation / oversampling / focal loss
- two classes confused → more data, higher resolution, bigger backbone
- confidently wrong / high ECE → label smoothing, recalibrate (temperature scaling)
- Grad-CAM shows background focus → cleaner crops / better augmentation
Re-run and record the **lift**.

## 5. Fill the README
- Results table: accuracy, macro-F1, weakest class, ECE.
- Add `confusion.png` and a couple of Grad-CAM example images.
- Write the **error-analysis story**: failure taxonomy → what you changed → measured improvement and why.
