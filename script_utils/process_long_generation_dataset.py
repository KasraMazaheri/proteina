#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from script_utils.process_designable_training_dataset import (  # noqa: E402
    ProcessTask,
    discover_dataset_ids,
    length_prefix_from_name,
    link_or_copy,
    process_single_pdb,
    run_tasks,
    write_csv,
)


DEFAULT_TARGET_ROOT = Path("/homes/kasram/broteina/dataset/long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process merged long-generation proteins in-place. Flat PDB files under "
            "dataset/long/raw/dataset_* are mirrored into the nested training layout, "
            "missing .pt files are generated, and aggregate CSV manifests are rebuilt."
        )
    )
    parser.add_argument(
        "--target-root",
        default=str(DEFAULT_TARGET_ROOT),
        help="Dataset root containing raw/dataset_* flat PDB files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of dataset indices to process, e.g. 0 1 2.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="CPU worker processes used for PDB -> .pt conversion. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Chunksize passed to the process pool.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to mirror flat raw PDBs into nested raw/ paths.",
    )
    parser.add_argument(
        "--store-het",
        action="store_true",
        help="Pass store_het=True into protein_to_pyg.",
    )
    parser.add_argument(
        "--no-store-bfactor",
        action="store_true",
        help="Disable storing B-factors in processed graph objects.",
    )
    args = parser.parse_args()
    if args.num_workers is None:
        import os

        args.num_workers = os.cpu_count() or 1
    return args


def nested_pdb_root(raw_dataset_dir: Path) -> Path:
    return raw_dataset_dir / "protein_out" / "0" / "pdbs"


def relative_input_path_from_flat_name(name: str) -> Path:
    prefix = length_prefix_from_name(name)
    if prefix is None:
        raise ValueError(f"Could not parse length prefix from {name}")
    return Path("protein_out") / "0" / "pdbs" / prefix / name


def mirror_flat_sources_in_place(raw_dataset_dir: Path, link_mode: str) -> int:
    mirrored = 0
    for source_file in sorted(raw_dataset_dir.glob("*.pdb")):
        try:
            relative_path = relative_input_path_from_flat_name(source_file.name)
        except ValueError:
            logger.warning(f"Skipping file with unexpected name format: {source_file}")
            continue
        target_path = raw_dataset_dir / relative_path
        if target_path.exists():
            continue
        link_or_copy(source_file, target_path, link_mode)
        mirrored += 1
    return mirrored


def build_dataset_rows_from_nested(raw_dataset_dir: Path) -> pd.DataFrame:
    root = nested_pdb_root(raw_dataset_dir)
    pdb_files = sorted(root.rglob("*.pdb")) if root.exists() else []
    rows = []
    for pdb_file in pdb_files:
        relative_path = pdb_file.relative_to(raw_dataset_dir)
        flat_id = "_".join(relative_path.with_suffix("").parts)
        rows.append(
            {
                "input_path": str(relative_path),
                "pdb": flat_id,
                "id": flat_id,
            }
        )
    return pd.DataFrame(rows, columns=["input_path", "pdb", "id"])


def collect_tasks_for_dataset(
    raw_dataset_dir: Path,
    processed_dataset_dir: Path,
    store_het: bool,
    store_bfactor: bool,
) -> tuple[pd.DataFrame, List[ProcessTask]]:
    df = build_dataset_rows_from_nested(raw_dataset_dir)
    tasks: List[ProcessTask] = []
    for row in df.itertuples(index=False):
        output_path = processed_dataset_dir / f"{row.pdb}.pt"
        if output_path.exists():
            continue
        tasks.append(
            ProcessTask(
                raw_dataset_dir=str(raw_dataset_dir),
                relative_input_path=str(row.input_path),
                output_path=str(output_path),
                file_id=str(row.pdb),
                store_het=store_het,
                store_bfactor=store_bfactor,
            )
        )
    return df, tasks


def extract_length_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["input_path"]
        .str.extract(r"protein_out/0/pdbs/(\d+)/", expand=False)
        .astype(int)
    )


def rebuild_aggregate_csvs(target_root: Path, dataset_ids: Sequence[int]) -> tuple[int, int]:
    dfs = []
    for dataset_id in dataset_ids:
        csv_path = target_root / f"custom_pdb_{dataset_id}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))

    if dfs:
        aggregate_df = pd.concat(dfs, ignore_index=True)
    else:
        aggregate_df = pd.DataFrame(columns=["input_path", "pdb", "id"])

    lengths = extract_length_series(aggregate_df) if not aggregate_df.empty else pd.Series(dtype=int)
    long_df = aggregate_df.loc[lengths <= 512].reset_index(drop=True) if not aggregate_df.empty else aggregate_df

    write_csv(target_root / "full_pdb.csv", aggregate_df)
    write_csv(target_root / "long_pdb.csv", long_df)
    write_csv(target_root / "custom_pdb.csv", aggregate_df)
    return len(long_df), len(aggregate_df)


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    target_root = Path(args.target_root).expanduser().resolve()
    raw_root = target_root / "raw"
    processed_root = target_root / "processed"
    processed_dataset_dir = processed_root / "dataset"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    processed_dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = discover_dataset_ids(raw_root, args.datasets)
    if not dataset_ids:
        raise ValueError(f"No dataset_* directories found under {raw_root}")

    logger.info(
        f"target_root={target_root} num_workers={args.num_workers} "
        f"link_mode={args.link_mode} datasets={dataset_ids}"
    )
    logger.info("This preprocessing path is CPU-only. No GPU memory is required.")

    store_bfactor = not args.no_store_bfactor
    total_mirrored = 0
    total_processed = 0

    for dataset_id in dataset_ids:
        raw_dataset_dir = raw_root / f"dataset_{dataset_id}"
        mirrored = mirror_flat_sources_in_place(raw_dataset_dir, args.link_mode)
        total_mirrored += mirrored

        df_dataset, dataset_tasks = collect_tasks_for_dataset(
            raw_dataset_dir=raw_dataset_dir,
            processed_dataset_dir=processed_dataset_dir,
            store_het=bool(args.store_het),
            store_bfactor=store_bfactor,
        )
        write_csv(target_root / f"custom_pdb_{dataset_id}.csv", df_dataset)
        logger.info(
            f"dataset_{dataset_id}: mirrored_new={mirrored} rows={len(df_dataset)} "
            f"missing_pt={len(dataset_tasks)}"
        )
        total_processed += run_tasks(
            tasks=dataset_tasks,
            num_workers=max(1, int(args.num_workers)),
            chunksize=max(1, int(args.chunksize)),
        )

    long_rows, full_rows = rebuild_aggregate_csvs(target_root, dataset_ids)
    logger.info(
        f"done mirrored_new={total_mirrored} processed_new={total_processed} "
        f"long_rows={long_rows} full_rows={full_rows}"
    )


if __name__ == "__main__":
    main()
