#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Downloads AFDB structures for use with generate_motif_scaffold_dataset.py.
#
# TWO MODES
# ---------
# Mode A (default, FAST): FoldSeek bulk download
#   Downloads AlphaFold/Swiss-Prot (~550k structures) as a single compressed
#   FoldSeek database (~2-5 GB), then decompresses to individual PDB files with
#   many threads.  No per-file HTTP requests, no rate-limiting.
#   Does NOT require the NGC d_FS_index.txt file.
#
# Mode B (exact d_FS replication): aria2c per-file download
#   Requires d_FS_index.txt from the NVIDIA NGC catalog.
#   Downloads each PDB individually from alphafold.ebi.ac.uk.
#   Use --mode index if you need the exact training set.
#
# Usage:
#   bash script_utils/download_afdb_dfs.sh                   # Mode A
#   bash script_utils/download_afdb_dfs.sh --mode index      # Mode B (exact d_FS)
#   bash script_utils/download_afdb_dfs.sh --threads 32      # override thread count
#   bash script_utils/download_afdb_dfs.sh --data_path /my/path

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
MODE="foldseek"
THREADS=16
DATA_PATH=""

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";      shift 2 ;;
        --threads)    THREADS="$2";   shift 2 ;;
        --data_path)  DATA_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- resolve DATA_PATH ------------------------------------------------------
if [ -z "$DATA_PATH" ]; then
    if [ -f ".env" ]; then
        DATA_PATH=$(grep -E "^DATA_PATH=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    fi
    DATA_PATH="${DATA_PATH:-./proteina_additional_files}"
fi

DFS_DIR="$DATA_PATH/d_FS"
RAW_DIR="$DFS_DIR/raw"
TMP_DIR="$DATA_PATH/tmp_foldseek"

echo "DATA_PATH : $DATA_PATH"
echo "Output    : $RAW_DIR"
echo "Mode      : $MODE"
echo ""

mkdir -p "$RAW_DIR"

# =============================================================================
# MODE A: FoldSeek bulk download (recommended)
# =============================================================================
if [ "$MODE" = "foldseek" ]; then

    if ! command -v foldseek &>/dev/null; then
        echo "ERROR: 'foldseek' not found in PATH."
        echo "Activate your conda environment first, e.g.:"
        echo "  conda activate ~/proteins_project/proteina_env"
        exit 1
    fi

    DB_DIR="$DATA_PATH/afdb_swissprot_db"
    mkdir -p "$DB_DIR" "$TMP_DIR"

    echo "Step 1/2 — Downloading AlphaFold/Swiss-Prot database via FoldSeek..."
    echo "         (single bulk compressed download, ~2-5 GB)"
    foldseek databases "Alphafold/Swiss-Prot" "$DB_DIR/db" "$TMP_DIR" \
        --threads "$THREADS"

    echo ""
    echo "Step 2/2 — Extracting individual PDB files with $THREADS threads..."
    foldseek convert2pdb "$DB_DIR/db" "$RAW_DIR" \
        --pdb-output-mode 1 \
        --threads "$THREADS"

    N=$(ls "$RAW_DIR"/*.pdb 2>/dev/null | wc -l)
    echo ""
    echo "Done. $N PDB files written to $RAW_DIR"
    echo ""
    echo "Note: This downloads AlphaFold/Swiss-Prot (~550k structures)."
    echo "The d_FS training set is a subset; the datamodule will use whatever"
    echo "is present in $RAW_DIR."

# =============================================================================
# MODE B: exact d_FS replication via per-file download from EBI API
# =============================================================================
elif [ "$MODE" = "index" ]; then

    INDEX_FILE="$DFS_DIR/d_FS_index.txt"

    if [ ! -f "$INDEX_FILE" ]; then
        echo "ERROR: Index file not found at $INDEX_FILE"
        echo ""
        echo "Obtain it from the NVIDIA NGC catalog:"
        echo "  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_training_data_indices/files"
        echo ""
        echo "With the NGC CLI:"
        echo "  ngc registry resource download-version \"nvidia/clara/proteina_training_data_indices:1\""
        echo "  unzip proteina_training_data_indices_v1/proteina_training_data_indices.zip"
        echo "  mkdir -p \"$DFS_DIR\""
        echo "  cp d_FS_index.txt \"$INDEX_FILE\""
        exit 1
    fi

    N_PROTEINS=$(wc -l < "$INDEX_FILE")
    echo "Downloading $N_PROTEINS structures from alphafold.ebi.ac.uk..."
    echo "(This is file-by-file and may be rate-limited; expect several hours.)"
    echo ""
    bash "$(dirname "$0")/download_afdb_data.sh" "$INDEX_FILE" "$DFS_DIR"

else
    echo "Unknown mode: $MODE  (use 'foldseek' or 'index')"
    exit 1
fi

echo ""
echo "Next: run the generation pipeline:"
echo "  python script_utils/generate_motif_scaffold_dataset.py \\"
echo "    --data_dir $DFS_DIR \\"
echo "    --output_dir ./synthetic_motif_scaffold_dataset \\"
echo "    --n_pairs 10000"
