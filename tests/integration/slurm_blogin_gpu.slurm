#!/bin/bash -l
#SBATCH -J gkmx_resources_gpu
#SBATCH -o /home/bepshuoz/diagnose_resources_blogin_gpu.out
#SBATCH -e /home/bepshuoz/diagnose_resources_blogin_gpu.err
#SBATCH --partition=gpu-a100:test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -t 00:30:00

set -euo pipefail

GKMX_DIR="/home/bepshuoz/Dev/gkmx"
micromamba activate gkmx
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$GKMX_DIR"
python -u tests/integration/diagnose_resources.py
