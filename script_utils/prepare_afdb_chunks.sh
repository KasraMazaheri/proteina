#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# One-time preprocessing: split AFDB CIFs into 8 chunk subdirs and convert to PDB.
# Run this before launching generate_motif_scaffold_dataset.sh.
#
# Usage:
#   sbatch script_utils/prepare_afdb_chunks.sh

#SBATCH --job-name=proteina_prepare_chunks
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prepare_chunks_%j.out
#SBATCH --error=logs/prepare_chunks_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

conda run -p ~/proteins_project/proteina_env \
    python3 "$REPO_DIR/script_utils/prepare_afdb_chunks.py" \
        --n_chunks 8 \
        --num_workers "${SLURM_CPUS_PER_TASK:-32}"
