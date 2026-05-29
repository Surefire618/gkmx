#!/bin/bash -l
#SBATCH -J gkmx_resources_cpu
#SBATCH -o /home/bepshuoz/diagnose_resources_blogin_cpu.out
#SBATCH -e /home/bepshuoz/diagnose_resources_blogin_cpu.err
#SBATCH --partition=cpu-clx:test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH -t 00:30:00

set -euo pipefail

GKMX_DIR="/home/bepshuoz/Dev/gkmx"
micromamba activate gkmx

cd "$GKMX_DIR"
python -u tests/integration/diagnose_resources.py
