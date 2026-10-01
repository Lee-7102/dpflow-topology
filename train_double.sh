#!/bin/bash

#SBATCH --job-name=dpflow_chairs_2xa6000
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=medium

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

#SBATCH --output=logs/dpflow_2xa6000_%j.out
#SBATCH --error=logs/dpflow_2xa6000_%j.err

set -eo pipefail

module load cuda/12.4

source /fs/nexus-scratch/liyiqian/miniconda3/etc/profile.d/conda.sh
conda activate /fs/nexus-scratch/liyiqian/miniconda3/envs/ptlflow

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "========================================"

nvidia-smi

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i}:",
        torch.cuda.get_device_name(i),
        round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        "GB"
    )
PY

echo "========================================"
echo "Starting DPFlow training"
echo "========================================"

srun python train.py \
    --config ptlflow/models/dpflow/configs/dpflow-train1-chairs.yaml \
    --trainer.devices 2 \
    --trainer.num_nodes 1 \
    --trainer.strategy ddp \
    --trainer.precision bf16-mixed \
    --data.train_dataset chairs-train \
    --data.val_dataset chairs-val \
    --data.train_batch_size 10