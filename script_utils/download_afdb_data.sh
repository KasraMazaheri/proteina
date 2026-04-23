# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <index_file> <output_dir>"
    exit 1
fi

# Convert paths to absolute paths
index_file=$(realpath "$1")
base_output_dir=$(realpath "$2")
output_dir="${base_output_dir}/raw"
mkdir -p "$output_dir"

# Change to the output directory before starting downloads
cd "$base_output_dir"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if command_exists aria2c; then
    BATCH_SIZE=5000

    # Build aria2c input for all missing files (two lines per entry: URL + out=)
    echo "Scanning for missing files..."
    all_input=$(mktemp)
    comm -23 \
        <(sort "$index_file") \
        <(ls "$output_dir" 2>/dev/null | sed 's/\.pdb$//' | sort) \
    | awk -v base="https://alphafold.ebi.ac.uk/files" \
          '{printf "%s/%s.pdb\n  out=raw/%s.pdb\n", base, $1, $1}' \
    > "$all_input"

    total_missing=$(( $(wc -l < "$all_input") / 2 ))
    echo "Files to download: $total_missing  (already on disk: $(ls "$output_dir" 2>/dev/null | wc -l))"

    if [ "$total_missing" -gt 0 ]; then
        # Split into chunks of BATCH_SIZE*2 lines (2 lines per file)
        split -l $(( BATCH_SIZE * 2 )) "$all_input" "${all_input}_chunk_"
        batch_num=0
        for chunk in "${all_input}_chunk_"*; do
            batch_num=$(( batch_num + 1 ))
            n_chunk=$(( $(wc -l < "$chunk") / 2 ))
            echo "Batch $batch_num: $n_chunk files"
            aria2c --input-file="$chunk" \
                --max-concurrent-downloads=16 \
                --max-connection-per-server=1 \
                --split=1 \
                --min-split-size=1M \
                --continue=true \
                --retry-wait=2 \
                --max-tries=3 \
                --console-log-level=warn \
                --optimize-concurrent-downloads=true \
                --file-allocation=none || true
            rm -f "$chunk"
            echo "  -> $(ls "$output_dir" 2>/dev/null | wc -l) files on disk so far"
        done
    fi
    rm -f "$all_input"
else
    # Fallback code for curl/wget
    download_cmd=""
    if command_exists curl; then
        download_cmd="curl -L --retry 3 --retry-delay 1 --progress-bar --continue-at - --output"
    elif command_exists wget; then
        download_cmd="wget --quiet --show-progress --progress=bar:force --continue --retry-connrefused --waitretry=1 --tries=3 -O"
    else
        echo "Error: Neither aria2c, curl, nor wget found. Please install one of them."
        exit 1
    fi

    while read -r filename; do
        output_file="raw/${filename}.pdb"
        if [ ! -f "$output_file" ]; then
            echo "Downloading: $filename"
            $download_cmd "$output_file" "https://alphafold.ebi.ac.uk/files/${filename}.pdb"
        fi
    done < "$index_file"
fi

echo "Download complete!"
