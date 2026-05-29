#!/bin/bash -l
#SBATCH -J gkmx_resources_cpu
#SBATCH -o /u/shuoz/diagnose_resources_raven_cpu.out
#SBATCH -e /u/shuoz/diagnose_resources_raven_cpu.err
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=00:10:00

set -euo pipefail

GKMX_DIR="/u/shuoz/Dev/gkmx"
export MAMBA_EXE='/raven/u/shuoz/bin/micromamba'
export MAMBA_ROOT_PREFIX='/raven/u/shuoz/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate gkmx

cd "$GKMX_DIR"
python -u tests/integration/diagnose_resources.py
