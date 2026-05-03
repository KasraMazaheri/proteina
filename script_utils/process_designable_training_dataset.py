#!/usr/bin/env python

import argparse
import errno
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import torch
from loguru import logger
from openfold.np.residue_constants import resname_to_idx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from graphein_utils.graphein_utils import protein_to_pyg
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR  # noqa: F401


DEFAULT_SOURCE_ROOT = Path("/homes/kasram/broteina/dataset/des/designable")
DEFAULT_TARGET_ROOT = Path("/homes/kasram/broteina/dataset/des/training_dataset")


@dataclass(frozen=True)
class ProcessTask:
    raw_dataset_dir: str
    relative_input_path: str
    output_path: str
    file_id: str
    store_het: bool
    store_bfactor: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror designable PDB files into the training dataset layout, "
            "process missing .pt files, and rebuild CSV manifests."
        )
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Root containing des/designable/dataset_* folders.",
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
        default=os.cpu_count() or 1,
        help="CPU worker processes used for PDB -> .pt conversion.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Chunksize passed to the process pool map.",
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
        help="Keep scanning for new designable PDBs and process them incrementally.",
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
        default="designable_pdb",
        help="Base filename (without .csv) for the aggregate CSV used by training.",
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
    return parser.parse_args()


def discover_dataset_ids(source_root: Path, requested: Optional[Sequence[int]]) -> List[int]:
    if requested is not None:
        return sorted(set(int(v) for v in requested))
    dataset_ids = []
    for entry in source_root.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.startswith("dataset_"):
            continue
        try:
            dataset_ids.append(int(entry.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(dataset_ids)


def length_prefix_from_name(name: str) -> Optional[str]:
    stem = Path(name).stem
    prefix = stem.split("_", 1)[0]
    return prefix if prefix.isdigit() else None


def raw_relative_path_for_source(name: str) -> Path:
    length_prefix = length_prefix_from_name(name)
    if length_prefix is None:
        raise ValueError(f"Could not parse length prefix from {name}")
    return Path("protein_out") / "0" / "pdbs" / length_prefix / name


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_parent_dir(dst)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                raise
            mode = "symlink"
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except FileExistsError:
            return
        except OSError:
            mode = "copy"
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def build_dataset_rows(raw_dataset_dir: Path) -> pd.DataFrame:
    pdb_files = sorted(raw_dataset_dir.rglob("*.pdb"))
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


def process_single_pdb(task: ProcessTask) -> Optional[str]:
    raw_dataset_dir = Path(task.raw_dataset_dir)
    path = raw_dataset_dir / task.relative_input_path
    if not path.exists():
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            path = gz_path
        else:
            return None

    fill_value_coords = 1e-5
    graph = protein_to_pyg(
        path=str(path),
        chain_selection="all",
        keep_insertions=True,
        store_het=task.store_het,
        store_bfactor=task.store_bfactor,
        fill_value_coords=fill_value_coords,
    )

    graph.id = task.file_id
    coord_mask = graph.coords != fill_value_coords
    graph.coord_mask = coord_mask[..., 0]
    graph.residue_type = torch.tensor(
        [resname_to_idx[residue] for residue in graph.residues]
    ).long()
    graph.database = "pdb"
    if task.store_bfactor:
        graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
    graph.residue_pdb_idx = torch.tensor(
        [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
    )
    graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

    output_path = Path(task.output_path)
    ensure_parent_dir(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(graph, tmp_path)
    os.replace(tmp_path, output_path)
    return task.file_id


def mirror_dataset_sources(
    source_dataset_dir: Path,
    target_raw_dataset_dir: Path,
    min_file_age_sec: int,
    link_mode: str,
) -> int:
    now = time.time()
    mirrored = 0
    for source_file in sorted(source_dataset_dir.glob("*.pdb")):
        if now - source_file.stat().st_mtime < min_file_age_sec:
            continue
        try:
            relative_path = raw_relative_path_for_source(source_file.name)
        except ValueError:
            logger.warning(f"Skipping file with unexpected name format: {source_file}")
            continue
        target_path = target_raw_dataset_dir / relative_path
        if target_path.exists():
            continue
        link_or_copy(source_file, target_path, link_mode)
        mirrored += 1
    return mirrored


def collect_tasks_for_dataset(
    raw_dataset_dir: Path,
    processed_dataset_dir: Path,
    legacy_processed_dataset_dir: Optional[Path],
    store_het: bool,
    store_bfactor: bool,
) -> tuple[pd.DataFrame, List[ProcessTask]]:
    df = build_dataset_rows(raw_dataset_dir)
    tasks: List[ProcessTask] = []
    for row in df.itertuples(index=False):
        output_path = processed_dataset_dir / f"{row.pdb}.pt"
        if output_path.exists():
            continue
        if legacy_processed_dataset_dir is not None:
            legacy_output_path = legacy_processed_dataset_dir / f"{row.pdb}.pt"
            if legacy_output_path.exists():
                link_or_copy(legacy_output_path, output_path, "hardlink")
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


def run_tasks(tasks: Sequence[ProcessTask], num_workers: int, chunksize: int) -> int:
    if not tasks:
        return 0
    processed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_pdb, task): task
            for task in tasks
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing PDB -> PT",
            unit="file",
        ):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(f"Failed processing {task.file_id}: {exc!r}")
                continue
            if result is not None:
                processed += 1
    return processed


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def rebuild_aggregate_csvs(target_root: Path, dataset_ids: Sequence[int], aggregate_name: str) -> int:
    dfs = []
    for dataset_id in dataset_ids:
        csv_path = target_root / f"custom_pdb_{dataset_id}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    if dfs:
        aggregate_df = pd.concat(dfs, ignore_index=True)
    else:
        aggregate_df = pd.DataFrame(columns=["input_path", "pdb", "id"])
    write_csv(target_root / f"{aggregate_name}.csv", aggregate_df)
    write_csv(target_root / "custom_pdb.csv", aggregate_df)
    return len(aggregate_df)


def process_once(args: argparse.Namespace) -> tuple[int, int, int]:
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    raw_root = target_root / "raw"
    processed_root = target_root / "processed"
    processed_dataset_dir = processed_root / "dataset"

    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    processed_dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = discover_dataset_ids(source_root, args.datasets)
    if not dataset_ids:
        raise ValueError(f"No dataset_* directories found under {source_root}")

    store_bfactor = not args.no_store_bfactor
    total_mirrored = 0
    total_processed = 0

    for dataset_id in dataset_ids:
        source_dataset_dir = source_root / f"dataset_{dataset_id}"
        if not source_dataset_dir.exists():
            logger.warning(f"Missing source dataset dir {source_dataset_dir}, skipping")
            continue

        raw_dataset_dir = raw_root / f"dataset_{dataset_id}"
        raw_dataset_dir.mkdir(parents=True, exist_ok=True)
        legacy_processed_dataset_dir = processed_root / f"dataset_{dataset_id}"

        mirrored = mirror_dataset_sources(
            source_dataset_dir=source_dataset_dir,
            target_raw_dataset_dir=raw_dataset_dir,
            min_file_age_sec=int(args.min_file_age_sec),
            link_mode=args.link_mode,
        )
        total_mirrored += mirrored

        df_dataset, dataset_tasks = collect_tasks_for_dataset(
            raw_dataset_dir=raw_dataset_dir,
            processed_dataset_dir=processed_dataset_dir,
            legacy_processed_dataset_dir=legacy_processed_dataset_dir,
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

    aggregate_rows = rebuild_aggregate_csvs(
        target_root=target_root,
        dataset_ids=dataset_ids,
        aggregate_name=args.aggregate_name,
    )
    return total_mirrored, total_processed, aggregate_rows


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.info(
        f"source_root={Path(args.source_root).expanduser().resolve()} "
        f"target_root={Path(args.target_root).expanduser().resolve()} "
        f"num_workers={args.num_workers} link_mode={args.link_mode} "
        f"watch={args.watch}"
    )
    logger.info("This preprocessing path is CPU-only. No GPU memory is required.")

    idle_started = None
    while True:
        mirrored, processed_now, aggregate_rows = process_once(args)
        logger.info(
            f"cycle_done mirrored_new={mirrored} processed_new={processed_now} "
            f"aggregate_rows={aggregate_rows}"
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
                logger.info(
                    f"idle for {idle_exit_after} seconds, exiting watch mode"
                )
                break

        if not args.watch:
            break

        logger.info(f"idle sleep {int(args.poll_interval_sec)}s")
        time.sleep(int(args.poll_interval_sec))


if __name__ == "__main__":
    main()
