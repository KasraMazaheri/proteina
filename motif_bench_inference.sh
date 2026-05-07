#!/usr/bin/env bash

# Local attached-node launcher for MotifBench inference.
# Starts one tmux session per task/GPU pair.
#
# Defaults:
#   TASKS="4 5 6 9"
#   GPUS="0 1 2 3"
#
# Example:
#   CONFIG_NAME=inference_motif \
#   GPUS="1 3 5 7" \
#   CKPT_NAME=proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt \
#   SC_SCALE_NOISE=1.0 \
#   SESSION_PREFIX=motifbench_base \
#   ./motif_bench_inference.sh
#
# Parallel launches for different configs are safe as long as you:
#   1. choose disjoint GPUS sets, and
#   2. use different SESSION_PREFIX values or CONFIG_NAME values.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

TASKS_STR="${TASKS:-4 5 6 9}"
GPUS_STR="${GPUS:-0 1 2 3}"
CONFIG_NAME="${CONFIG_NAME:-inference_motif}"
CONFIG_SUBDIR="${CONFIG_SUBDIR:-}"
CONDA_ENV="${CONDA_ENV:-proteina_env}"
SESSION_PREFIX="${SESSION_PREFIX:-motifbench}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/motif_bench_inference}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$REPO_DIR/inference_runs}"
CKPT_NAME="${CKPT_NAME:-}"
SC_SCALE_NOISE="${SC_SCALE_NOISE:-}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT_BASE"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not available in PATH." >&2
    exit 1
fi

read -r -a TASKS <<< "$TASKS_STR"
read -r -a GPUS <<< "$GPUS_STR"

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "ERROR: TASKS is empty." >&2
    exit 1
fi

if [[ ${#TASKS[@]} -ne ${#GPUS[@]} ]]; then
    echo "ERROR: TASKS count (${#TASKS[@]}) must equal GPUS count (${#GPUS[@]})." >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
safe_config="$(printf '%s' "$CONFIG_NAME" | tr '/ ' '__')"

echo "Repo        : $REPO_DIR"
echo "Config      : $CONFIG_NAME"
echo "Config dir  : ${CONFIG_SUBDIR:-<default>}"
echo "Conda env   : $CONDA_ENV"
echo "Tasks       : ${TASKS[*]}"
echo "GPUs        : ${GPUS[*]}"
echo "Ckpt name   : ${CKPT_NAME:-<config default>}"
echo "sc_noise    : ${SC_SCALE_NOISE:-<config default>}"
echo "Output base : $OUTPUT_ROOT_BASE"
echo "Logs        : $LOG_DIR"
echo "Prefix      : $SESSION_PREFIX"

for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    gpu="${GPUS[$idx]}"
    session="${SESSION_PREFIX}_${safe_config}_t${task}_g${gpu}_${timestamp}"
    log_file="$LOG_DIR/${session}.log"

    tmux new-session -d -s "$session" "bash -lc '
        set -eo pipefail
        exec > >(tee -a \"$log_file\") 2>&1
        cd \"$REPO_DIR\"
        echo \"[\$(date)] session=$session task=$task gpu=$gpu config=$CONFIG_NAME\"
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate \"$CONDA_ENV\"
        export CUDA_VISIBLE_DEVICES=\"$gpu\"
        cmd=(
            python proteinfoundation/motif_inference.py
            --config_name \"$CONFIG_NAME\"
            --motif_task_number \"$task\"
            --split_id \"$idx\"
            --output_root_base \"$OUTPUT_ROOT_BASE\"
        )
        if [[ -n \"$CONFIG_SUBDIR\" ]]; then
            cmd+=(--config_subdir \"$CONFIG_SUBDIR\")
        fi
        if [[ -n \"$CKPT_NAME\" ]]; then
            cmd+=(--ckpt_name \"$CKPT_NAME\")
        fi
        if [[ -n \"$SC_SCALE_NOISE\" ]]; then
            cmd+=(--sc_scale_noise \"$SC_SCALE_NOISE\")
        fi
        set +e
        \"\${cmd[@]}\"
        status=\$?
        set -e
        echo \"[\$(date)] finished status=\$status\"
        if [[ \$status -ne 0 ]]; then
            exec bash
        fi
    '"

    echo "Started $session -> $log_file"
done

echo ""
echo "Monitor:"
echo "  tmux ls | grep '$SESSION_PREFIX'"
echo "  tail -f '$LOG_DIR/${SESSION_PREFIX}_${safe_config}_t${TASKS[0]}_g${GPUS[0]}_${timestamp}.log'"
