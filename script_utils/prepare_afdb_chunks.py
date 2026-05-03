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
from pathlib import Path
from multiprocessing import Pool

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from script_utils.process_designable_training_dataset import ProcessTask, run_tasks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=None,
                   help="Base data path (defaults to $DATA_PATH or ./proteina_additional_files)")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--chunksize", type=int, default=16)
    p.add_argument("--skip_processing", action="store_true",
                   help="Only split/convert raw files and write chunk CSVs; do not create processed .pt files.")
    p.add_argument("--store_het", action="store_true")
    p.add_argument("--no-store-bfactor", action="store_true")
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


def write_chunk_csv_and_tasks(
    chunk_raw: Path,
    chunk_processed: Path,
    chunk_csv: Path,
    store_het: bool,
    store_bfactor: bool,
):
    rows = []
    tasks = []
    for pdb_file in sorted(chunk_raw.glob("*.pdb")):
        file_id = pdb_file.stem
        rows.append({"input_path": pdb_file.name, "pdb": file_id, "id": file_id})
        output_path = chunk_processed / f"{file_id}.pt"
        if output_path.exists():
            continue
        tasks.append(
            ProcessTask(
                raw_dataset_dir=str(chunk_raw),
                relative_input_path=pdb_file.name,
                output_path=str(output_path),
                file_id=file_id,
                store_het=store_het,
                store_bfactor=store_bfactor,
            )
        )

    import pandas as pd

    chunk_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["input_path", "pdb", "id"]).to_csv(chunk_csv, index=False)
    return len(rows), tasks


def main():
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if args.data_path is None:
        args.data_path = os.environ.get("DATA_PATH", str(REPO_ROOT / "proteina_additional_files"))

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    args.data_path = str(data_path)

    raw_dir = Path(args.data_path) / "d_FS" / "raw"
    cif_files = sorted(raw_dir.glob("*.cif"))

    # Create chunk dirs
    chunk_dirs = []
    for i in range(args.n_chunks):
        chunk_raw = Path(args.data_path) / "d_FS" / f"chunk_{i:02d}" / "raw"
        chunk_raw.mkdir(parents=True, exist_ok=True)
        chunk_dirs.append(chunk_raw)

    if cif_files:
        logger.info(f"Found {len(cif_files)} .cif files in {raw_dir}")

        # Build interleaved (cif_path, pdb_path) work list
        work = []
        for idx, cif_path in enumerate(cif_files):
            chunk_id = idx % args.n_chunks
            pdb_path = chunk_dirs[chunk_id] / f"{cif_path.stem}.pdb"
            work.append((str(cif_path), str(pdb_path)))

        already_done = sum(1 for _, dst in work if os.path.exists(dst))
        remaining = len(work) - already_done
        logger.info(f"{already_done} already converted, {remaining} remaining")

        if remaining > 0:
            success_count = 0
            fail_count = 0

            with Pool(processes=args.num_workers) as pool:
                for _, _, ok, err in tqdm(
                    pool.imap_unordered(convert_cif_to_pdb, work, chunksize=64),
                    total=len(work),
                    desc="CIF->PDB",
                    unit="file",
                ):
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        if fail_count <= 20:
                            logger.warning(f"Conversion failed: {err}")

            logger.info(
                f"Conversion done. {success_count} converted successfully, {fail_count} failed."
            )
        else:
            logger.info("All CIF files already converted.")
    elif not any(next(chunk_raw.glob("*.pdb"), None) is not None for chunk_raw in chunk_dirs):
        logger.error(f"No .cif files found in {raw_dir} and no chunk PDBs found under d_FS/chunk_*/raw")
        sys.exit(1)
    else:
        logger.info(f"No .cif files found in {raw_dir}; using existing chunk PDB files.")

    all_tasks = []
    store_bfactor = not args.no_store_bfactor
    for i, chunk_raw in enumerate(chunk_dirs):
        chunk_dir = chunk_raw.parent
        chunk_processed = chunk_dir / "processed"
        chunk_processed.mkdir(parents=True, exist_ok=True)
        chunk_csv = chunk_dir / f"chunk_{i:02d}.csv"
        n_rows, tasks = write_chunk_csv_and_tasks(
            chunk_raw=chunk_raw,
            chunk_processed=chunk_processed,
            chunk_csv=chunk_csv,
            store_het=bool(args.store_het),
            store_bfactor=store_bfactor,
        )
        all_tasks.extend(tasks)
        logger.info(f"chunk_{i:02d}: rows={n_rows} missing_pt={len(tasks)} csv={chunk_csv}")

    if args.skip_processing:
        logger.info("Skipping PDB -> PT processing by request.")
        return

    processed = run_tasks(
        tasks=all_tasks,
        num_workers=max(1, int(args.num_workers)),
        chunksize=max(1, int(args.chunksize)),
    )
    logger.info(f"Processing done. processed_new={processed} missing_initial={len(all_tasks)}")


if __name__ == "__main__":
    main()
