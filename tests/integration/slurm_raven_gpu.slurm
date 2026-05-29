#!/bin/bash -l
#SBATCH -J gkmx_resources_gpu
#SBATCH -o /u/shuoz/diagnose_resources_raven_gpu.out
#SBATCH -e /u/shuoz/diagnose_resources_raven_gpu.err
#SBATCH --partition=gpu
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=00:10:00

set -euo pipefail

GKMX_DIR="/u/shuoz/Dev/gkmx"
# jax[cuda12] bundles its own CUDA libs — do NOT module load cuda.
export MAMBA_EXE='/raven/u/shuoz/bin/micromamba'
export MAMBA_ROOT_PREFIX='/raven/u/shuoz/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate gkmx

# Required on Raven A100: without this jax grabs 90% of GPU at import time
# and the diagnostic's pynvml/jax_memstats numbers won't reflect true free.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$GKMX_DIR"
python -u tests/integration/diagnose_resources.py
