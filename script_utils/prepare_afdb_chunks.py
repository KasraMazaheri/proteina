#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
One-time preparation step: split AFDB .cif files into N chunk subdirectories
and convert each to .pdb format using biotite.

Output layout:
    <data_path>/d_FS/
        chunk_00/raw/*.pdb
        chunk_01/raw/*.pdb
        ...
        chunk_07/raw/*.pdb

Usage:
    python script_utils/prepare_afdb_chunks.py
    python script_utils/prepare_afdb_chunks.py --n_chunks 8 --num_workers 32
"""

import argparse
import os
import sys
import glob
from multiprocessing import Pool
from functools import partial

root = os.path.abspath(".")
sys.path.append(root)

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=None,
                   help="Base data path (defaults to $DATA_PATH or ./proteina_additional_files)")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=32)
    return p.parse_args()


def convert_cif_to_pdb(args_tuple):
    """Convert a single CIF file to PDB. Returns (src, dst, success, error)."""
    cif_path, pdb_path = args_tuple
    if os.path.exists(pdb_path):
        return (cif_path, pdb_path, True, None)
    try:
        import biotite.structure.io as strucio
        from biotite.structure.io.pdb import PDBFile

        structure = strucio.load_structure(cif_path, model=1)
        pdb_file = PDBFile()
        pdb_file.set_structure(structure)
        pdb_file.write(pdb_path)
        return (cif_path, pdb_path, True, None)
    except Exception as e:
        return (cif_path, pdb_path, False, str(e))


def main():
    load_dotenv()
    args = parse_args()

    if args.data_path is None:
        args.data_path = os.environ.get("DATA_PATH", "./proteina_additional_files")

    raw_dir = os.path.join(args.data_path, "d_FS", "raw")
    cif_files = sorted(glob.glob(os.path.join(raw_dir, "*.cif")))
    if not cif_files:
        logger.error(f"No .cif files found in {raw_dir}")
        sys.exit(1)

    logger.info(f"Found {len(cif_files)} .cif files in {raw_dir}")

    # Create chunk raw dirs
    chunk_dirs = []
    for i in range(args.n_chunks):
        chunk_raw = os.path.join(args.data_path, "d_FS", f"chunk_{i:02d}", "raw")
        os.makedirs(chunk_raw, exist_ok=True)
        chunk_dirs.append(chunk_raw)

    # Build interleaved (cif_path, pdb_path) work list
    work = []
    for idx, cif_path in enumerate(cif_files):
        chunk_id = idx % args.n_chunks
        stem = os.path.splitext(os.path.basename(cif_path))[0]
        pdb_path = os.path.join(chunk_dirs[chunk_id], f"{stem}.pdb")
        work.append((cif_path, pdb_path))

    already_done = sum(1 for _, dst in work if os.path.exists(dst))
    remaining = len(work) - already_done
    logger.info(f"{already_done} already converted, {remaining} remaining")

    if remaining == 0:
        logger.info("All files already converted.")
        return

    success_count = 0
    fail_count = 0

    with Pool(processes=args.num_workers) as pool:
        for _, _, ok, err in tqdm(
            pool.imap_unordered(convert_cif_to_pdb, work, chunksize=64),
            total=len(work),
            desc="CIF→PDB",
            unit="file",
        ):
            if ok:
                success_count += 1
            else:
                fail_count += 1
                if fail_count <= 20:
                    logger.warning(f"Conversion failed: {err}")

    logger.info(
        f"Done. {success_count} converted successfully, {fail_count} failed."
    )
    for i, chunk_raw in enumerate(chunk_dirs):
        n = len(glob.glob(os.path.join(chunk_raw, "*.pdb")))
        logger.info(f"  chunk_{i:02d}: {n} PDB files")


if __name__ == "__main__":
    main()
