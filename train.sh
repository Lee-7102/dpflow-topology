#!/bin/bash

#SBATCH --job-name=dpflow_chairs_a6000
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dpflow_a6000_%j.out
#SBATCH --error=logs/dpflow_a6000_%j.err


module load cuda/12.4

source /fs/nexus-scratch/liyiqian/miniconda3/etc/profile.d/conda.sh
conda activate /fs/nexus-scratch/liyiqian/miniconda3/envs/ptlflow

cd /fs/nexus-scratch/liyiqian/ptlflow

echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "======================================"

python train.py --config ptlflow/models/dpflow/configs/dpflow-train1-chairs.yaml \
                --trainer.devices 1 