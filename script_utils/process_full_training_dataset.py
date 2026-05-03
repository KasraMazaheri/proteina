#!/usr/bin/env python

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from script_utils.process_designable_training_dataset import process_once


DEFAULT_SOURCE_ROOT = Path("/homes/kasram/broteina/dataset/des/raw")
DEFAULT_TARGET_ROOT = Path("/homes/kasram/broteina/dataset/des/full_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror raw unconditional PDB files into a training dataset layout, "
            "process missing .pt files, rebuild CSV manifests, and maintain a "
            "length-balanced manifest."
        )
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Root containing des/raw/dataset_* folders.",
    )
    parser.add_argument(
        "--target-root",
        default=str(DEFAULT_TARGET_ROOT),
        help="Target dataset root that contains raw/, processed/, and CSV files.",
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
        help="How to mirror raw PDBs into target raw/.",
    )
    parser.add_argument(
        "--min-file-age-sec",
        type=int,
        default=15,
        help="Ignore source files newer than this to avoid racing active writers.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep scanning for new raw PDBs and process them incrementally.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=60,
        help="Sleep interval between scans in watch mode.",
    )
    parser.add_argument(
        "--idle-exit-after-sec",
        type=int,
        default=1800,
        help="Exit after this many idle seconds in watch mode. Use 0 to never auto-exit.",
    )
    parser.add_argument(
        "--aggregate-name",
        default="full_pdb",
        help="Base filename (without .csv) for the aggregate CSV.",
    )
    parser.add_argument(
        "--balanced-name",
        default="uniform_pdb",
        help="Base filename (without .csv) for the equal-length resampled CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced resampling.",
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


def extract_length_series(df: pd.DataFrame) -> pd.Series:
    lengths = (
        df["input_path"]
        .str.extract(r"protein_out/0/pdbs/(\d+)/", expand=False)
        .astype(int)
    )
    return lengths


def rebuild_balanced_csv(target_root: Path, aggregate_name: str, balanced_name: str, seed: int) -> int:
    aggregate_csv = target_root / f"{aggregate_name}.csv"
    if not aggregate_csv.exists():
        balanced_df = pd.DataFrame(columns=["input_path", "pdb", "id"])
        balanced_df.to_csv(target_root / f"{balanced_name}.csv", index=False)
        return 0

    df = pd.read_csv(aggregate_csv)
    if df.empty:
        df.to_csv(target_root / f"{balanced_name}.csv", index=False)
        return 0

    df = df.copy()
    df["length"] = extract_length_series(df)
    length_counts = df["length"].value_counts().sort_index()
    samples_per_length = int(length_counts.min())

    balanced_df = (
        df.groupby("length", group_keys=False)
        .apply(lambda g: g.sample(n=samples_per_length, random_state=seed))
        .drop(columns="length")
        .reset_index(drop=True)
    )
    balanced_df.to_csv(target_root / f"{balanced_name}.csv", index=False)
    return len(balanced_df)


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    logger.info(
        f"source_root={source_root} target_root={target_root} "
        f"num_workers={args.num_workers} link_mode={args.link_mode} watch={args.watch}"
    )
    logger.info("This preprocessing path is CPU-only. No GPU memory is required.")

    idle_started = None
    while True:
        mirrored, processed_now, aggregate_rows = process_once(args)
        balanced_rows = rebuild_balanced_csv(
            target_root=target_root,
            aggregate_name=args.aggregate_name,
            balanced_name=args.balanced_name,
            seed=int(args.seed),
        )
        logger.info(
            f"cycle_done mirrored_new={mirrored} processed_new={processed_now} "
            f"aggregate_rows={aggregate_rows} balanced_rows={balanced_rows}"
        )

        if mirrored > 0 or processed_now > 0:
            idle_started = None
        else:
            if not args.watch:
                break
            now = time.time()
            if idle_started is None:
                idle_started = now
            idle_exit_after = int(args.idle_exit_after_sec)
            if idle_exit_after > 0 and now - idle_started >= idle_exit_after:
                logger.info(f"idle for {idle_exit_after} seconds, exiting watch mode")
                break

        if not args.watch:
            break

        logger.info(f"idle sleep {int(args.poll_interval_sec)}s")
        time.sleep(int(args.poll_interval_sec))


if __name__ == "__main__":
    main()
