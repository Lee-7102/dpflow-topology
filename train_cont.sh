#!/bin/bash

#SBATCH --job-name=dpflow_chairs_h200
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dpflow_h200_%j.out
#SBATCH --error=logs/dpflow_h200_%j.err
#SBATCH --requeue

set -eo pipefail

module load cuda/12.4

source /fs/nexus-scratch/liyiqian/miniconda3/etc/profile.d/conda.sh
conda activate /fs/nexus-scratch/liyiqian/miniconda3/envs/ptlflow

cd /fs/nexus-scratch/liyiqian/ptlflow
mkdir -p logs

# Keep this experiment's checkpoints under one root
LOG_ROOT="/fs/nexus-scratch/liyiqian/ptlflow/ptlflow_logs/dpflow_chairs_h200"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "========================================"

nvidia-smi

# Find newest FULL training checkpoint.
# IMPORTANT: use dpflow_train_*.ckpt, not dpflow_last_*.ckpt.
CKPT="$(
    find "${LOG_ROOT}" \
        -type f \
        -name 'dpflow_train_*.ckpt' \
        -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

TRAIN_ARGS=(
    --config ptlflow/models/dpflow/configs/dpflow-train1-chairs.yaml
    --trainer.devices 1
    --trainer.precision bf16-mixed
    --data.train_dataset chairs-train
    --data.val_dataset chairs-val
    --data.train_batch_size 10
    --log_dir "${LOG_ROOT}"
)

if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
    echo "========================================"
    echo "Resuming from checkpoint:"
    echo "${CKPT}"
    echo "========================================"

    TRAIN_ARGS+=(--ckpt_path "${CKPT}")
else
    echo "========================================"
    echo "No existing checkpoint found."
    echo "Starting training from scratch."
    echo "========================================"
fi

srun python train.py "${TRAIN_ARGS[@]}"