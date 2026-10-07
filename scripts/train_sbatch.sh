#!/bin/bash
#SBATCH --job-name=PuttingCupintotheDish_demo100
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/meat124/openpi/logs/%x_%j.out
#SBATCH --error=/lustre/meat124/openpi/logs/%x_%j.err

# ---------------------------------------------------------------
# Usage:
#   sbatch train_sbatch.sh                         # default args
#   sbatch train_sbatch.sh --exp_name my_run       # extra args forwarded to train.py
#   sbatch --gres=gpu:1 train_sbatch.sh            # override GPU count
# ---------------------------------------------------------------

set -euo pipefail

# --- paths ---
OPENPI_DIR=/lustre/meat124/openpi
cd "$OPENPI_DIR"

# create log dir if needed
mkdir -p "$OPENPI_DIR/logs"

echo "=============================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "=============================="

# --- train ---
# All extra arguments passed to sbatch after -- are forwarded to train.py.
# Default config: pi05_rby1 / exp_name must be set.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/train.py \
    pi05_rby1 \
    --exp-name ${SLURM_JOB_NAME} \
    --batch_size 16 \
    --data.num-episodes 100 \
    "$@"

echo "=============================="
echo "End          : $(date)"
echo "=============================="
