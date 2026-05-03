#!/usr/bin/env bash
# One-time local preprocessing: split AFDB CIFs into chunk subdirs and convert
# them to PDB. This intentionally does not use Slurm.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

N_CHUNKS="${N_CHUNKS:-8}"
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"
DATA_PATH_ARG="${DATA_PATH_ARG:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate proteina_env

echo "Repo       : $REPO_DIR"
echo "Node       : $(hostname)"
echo "Chunks     : $N_CHUNKS"
echo "Workers    : $NUM_WORKERS"

cmd=(
    python "$REPO_DIR/script_utils/prepare_afdb_chunks.py"
    --n_chunks "$N_CHUNKS"
    --num_workers "$NUM_WORKERS"
)

if [[ -n "$DATA_PATH_ARG" ]]; then
    cmd+=(--data_path "$DATA_PATH_ARG")
fi

"${cmd[@]}"
