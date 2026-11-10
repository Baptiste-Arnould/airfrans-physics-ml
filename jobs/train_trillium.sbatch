#!/bin/bash
#SBATCH --job-name=fb-train
#SBATCH --account=def-laurende
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/baarn45/FLUID_BENCH/logs/%x-%j.out
# Trillium: no --mem / --partition (1 GPU => 24 cores + 187.5 GB). Jobs start with an empty env (--export=NONE).
# Usage, from the trillium-gpu login:  sbatch jobs/train_trillium.sbatch configs/a.toml [configs/b.toml ...]
# Several configs run one after the other (partition debug: 1 job per user): sbatch -p debug --time=00:30:00 ...
# Resubmitting the same config resumes from runs/<config>-<task>/last.pt.
set -e
module --force purge
module load StdEnv/2023 python/3.12.4
source /scratch/baarn45/FLUID_BENCH/venv/bin/activate
cd /scratch/baarn45/FLUID_BENCH/airfrans-physics-ml
nvidia-smi -L
status=0
for c in "$@"; do python -u src/train.py "$c" || status=1; done  # one failure does not skip the next config
exit $status
