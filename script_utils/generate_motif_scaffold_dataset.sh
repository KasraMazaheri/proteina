#!/usr/bin/env bash
# Launch motif-scaffold dataset generation locally: one tmux session per GPU,
# one AFDB chunk per process. This intentionally does not use Slurm.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-/homes/kasram/broteina/dataset/motif_scaffold/raw}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_CHUNKS="${N_CHUNKS:-8}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
SEED_BASE="${SEED_BASE:-}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
AMP_DTYPE="${AMP_DTYPE:-config}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
SESSION_PREFIX="${SESSION_PREFIX:-motif_scaffold_n${NODE_RANK}}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/motif_scaffold_dataset}"
DATASET_CONFIG="${DATASET_CONFIG:-d_FS_chunk}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-inference_motif}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not available in PATH." >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"

echo "Repo       : $REPO_DIR"
echo "Output     : $OUTPUT_DIR"
echo "GPUs       : $GPUS"
echo "Chunks     : $N_CHUNKS"
echo "Nodes      : $NUM_NODES"
echo "Node rank  : $NODE_RANK"
echo "Seed base  : ${SEED_BASE:-config_default}"
echo "Seed stride: $SEED_STRIDE"
echo "AMP dtype  : $AMP_DTYPE"
echo "Batch size : $BATCH_SIZE"
echo "Logs       : $LOG_DIR"

chunk_id=0
for gpu in $GPUS; do
    if (( chunk_id >= N_CHUNKS )); then
        break
    fi

    chunk_name="$(printf 'chunk_%02d' "$chunk_id")"
    session="${SESSION_PREFIX}_${chunk_name}_gpu${gpu}_${timestamp}"
    log_file="$LOG_DIR/${session}.log"

    tmux new-session -d -s "$session" "bash -lc '
        set -eo pipefail
        exec > >(tee -a \"$log_file\") 2>&1
        cd \"$REPO_DIR\"
        echo \"[\$(date)] starting session=$session gpu=$gpu chunk=$chunk_id output=$OUTPUT_DIR\"
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate proteina_env
        export CUDA_VISIBLE_DEVICES=\"$gpu\"
        echo \"[\$(date)] activated proteina_env; CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES node_rank=$NODE_RANK/$NUM_NODES\"
        set +e
        cmd=(
            python \"$REPO_DIR/script_utils/generate_motif_scaffold_dataset.py\"
            --dataset_config \"$DATASET_CONFIG\"
            --inference_config \"$INFERENCE_CONFIG\"
            --output_dir \"$OUTPUT_DIR\"
            --chunk_id \"$chunk_id\"
            --n_chunks \"$N_CHUNKS\"
            --num_nodes \"$NUM_NODES\"
            --node_rank \"$NODE_RANK\"
            --seed_stride \"$SEED_STRIDE\"
            --amp_dtype \"$AMP_DTYPE\"
            --batch_size \"$BATCH_SIZE\"
        )
        if [[ -n \"$SEED_BASE\" ]]; then
            cmd+=(--seed_base \"$SEED_BASE\")
        fi
        \"\${cmd[@]}\"
        status=\${PIPESTATUS[0]}
        set -e
        echo \"[\$(date)] finished with status=\$status\"
        if [[ \$status -ne 0 ]]; then
            exec bash
        fi
    '"

    echo "Started $session -> $log_file"
    chunk_id=$((chunk_id + 1))
done

if (( chunk_id < N_CHUNKS )); then
    echo "WARNING: launched $chunk_id chunks but N_CHUNKS=$N_CHUNKS. Add more GPUs via GPUS=\"...\" or launch remaining chunks manually." >&2
fi

echo ""
echo "Monitor:"
echo "  tmux ls | grep '$SESSION_PREFIX'"
echo "  tail -f '$LOG_DIR/${SESSION_PREFIX}_chunk_00_gpu0_${timestamp}.log'"
