#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# SLURM job array: generate (motif, scaffold) dataset across 8 GPUs.
# Each task processes one pre-split chunk via PDBLightningDataModule.
#
# Prerequisites:
#   sbatch script_utils/prepare_afdb_chunks.sh   # one-time CIF→PDB split
#
# Usage:
#   sbatch script_utils/generate_motif_scaffold_dataset.sh
#   sbatch --export=ALL,OUTPUT_DIR=/path/to/out script_utils/generate_motif_scaffold_dataset.sh

#SBATCH --job-name=proteina_motif_dataset
#SBATCH --array=0-7
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/motif_dataset_%A_%a.out
#SBATCH --error=logs/motif_dataset_%A_%a.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-./synthetic_motif_scaffold_dataset}"
BATCH_SIZE=16

mkdir -p logs

echo "Job array task ${SLURM_ARRAY_TASK_ID}/8"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run -p ~/proteins_project/proteina_env \
    python "$REPO_DIR/script_utils/generate_motif_scaffold_dataset.py" \
        --output_dir "${OUTPUT_DIR}" \
        --chunk_id "${SLURM_ARRAY_TASK_ID}" \
        --n_chunks 8 \
        --batch_size "${BATCH_SIZE}"
