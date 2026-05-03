#!/usr/bin/env python

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from loguru import logger
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

import biotite.structure.io.pdb as pdb
import biotite.structure.sse as annotate
import numpy as np


DEFAULT_INPUT_CSV = Path("/homes/kasram/broteina/dataset/des/training_dataset/designable_pdb.csv")
DEFAULT_OUTPUT_CSV = Path("/homes/kasram/broteina/dataset/des/training_dataset/designable_ss_pdb.csv")
DEFAULT_RAW_ROOT = Path("/homes/kasram/broteina/dataset/des/training_dataset/raw")
GPU_NAME_RE = re.compile(r"_g(?P<gpu>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate secondary structure for every entry in a designable_pdb.csv-like manifest "
            "and write a merged CSV with alpha/beta/coil fractions appended."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="CSV manifest to read, e.g. designable_pdb.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Merged output CSV with SS columns appended.",
    )
    parser.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help="Root containing raw/dataset_* directories referenced by the manifest.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU worker processes for SS evaluation.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Executor chunksize.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Append results to output CSV every N processed proteins.",
    )
    return parser.parse_args()


def analyze_secondary_structure(pdb_path: Path) -> np.ndarray:
    pdb_file = pdb.PDBFile.read(str(pdb_path))
    array = pdb_file.get_structure(model=1)
    ca_atoms = array[array.atom_name == "CA"]
    sse = annotate.annotate_sse(ca_atoms)

    total = len(sse)
    if total == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    alpha_count = np.count_nonzero(sse == "a")
    beta_count = np.count_nonzero(sse == "b")
    coil_count = np.count_nonzero(sse == "c")
    return np.array([alpha_count, beta_count, coil_count], dtype=np.float64) / total


def read_input_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"No CSV columns found in {path}")
    return rows, fieldnames


def read_done_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = row.get("id")
            if row_id:
                done.add(row_id)
    return done


def append_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return len(rows)


def dataset_ids_from_raw_root(raw_root: Path) -> List[int]:
    dataset_ids = []
    for entry in raw_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("dataset_"):
            continue
        try:
            dataset_ids.append(int(entry.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(dataset_ids)


def parse_gpu_dataset_id(*values: str) -> Optional[int]:
    for value in values:
        if not value:
            continue
        match = GPU_NAME_RE.search(value)
        if match is not None:
            return int(match.group("gpu"))
    return None


def resolve_pdb_path(row: Dict[str, str], raw_root: Path, dataset_ids: Sequence[int]) -> Path:
    input_path = row["input_path"]
    preferred_dataset_id = parse_gpu_dataset_id(row.get("input_path", ""), row.get("pdb", ""), row.get("id", ""))
    if preferred_dataset_id is not None:
        candidate = raw_root / f"dataset_{preferred_dataset_id}" / input_path
        if candidate.exists():
            return candidate

    found = []
    for dataset_id in dataset_ids:
        candidate = raw_root / f"dataset_{dataset_id}" / input_path
        if candidate.exists():
            found.append(candidate)
            if len(found) > 1:
                break

    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"Could not resolve raw path for row id={row.get('id')} input_path={input_path}")
    raise RuntimeError(f"Ambiguous raw path for row id={row.get('id')} input_path={input_path}")


def worker_eval(task: Tuple[Dict[str, str], str]) -> Tuple[Optional[Dict[str, str]], Optional[str], str]:
    row, pdb_path_str = task
    row_id = row.get("id", "")
    try:
        ss = analyze_secondary_structure(Path(pdb_path_str))
    except Exception as exc:
        return None, repr(exc), row_id

    merged = dict(row)
    merged["alpha"] = f"{float(ss[0]):.8f}"
    merged["beta"] = f"{float(ss[1]):.8f}"
    merged["coil"] = f"{float(ss[2]):.8f}"
    return merged, None, row_id


def task_iterator(
    rows: Sequence[Dict[str, str]], raw_root: Path, dataset_ids: Sequence[int]
) -> Iterator[Tuple[Dict[str, str], str]]:
    for row in rows:
        try:
            yield row, str(resolve_pdb_path(row, raw_root, dataset_ids))
        except Exception as exc:
            logger.warning(f"Skipping id={row.get('id', '')}: {exc!r}")


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()

    rows, input_fieldnames = read_input_rows(input_csv)
    dataset_ids = dataset_ids_from_raw_root(raw_root)
    if not dataset_ids:
        raise ValueError(f"No dataset_* directories found under {raw_root}")

    done_ids = read_done_ids(output_csv)
    pending_rows = [row for row in rows if row.get("id") not in done_ids]
    output_fieldnames = list(input_fieldnames) + ["alpha", "beta", "coil"]

    logger.info(
        f"input_csv={input_csv} output_csv={output_csv} raw_root={raw_root} "
        f"rows_total={len(rows)} done_at_start={len(done_ids)} pending={len(pending_rows)} "
        f"num_workers={args.num_workers}"
    )
    logger.info("This SS evaluation path is CPU-only. No GPU memory is required.")

    if not pending_rows:
        logger.info("No pending rows. Output CSV is already up to date.")
        return

    buffer: List[Dict[str, str]] = []
    processed = 0

    tasks = task_iterator(pending_rows, raw_root, dataset_ids)
    with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
        results = executor.map(worker_eval, tasks, chunksize=max(1, int(args.chunksize)))
        for merged_row, error, row_id in tqdm(results, total=len(pending_rows), desc="SS eval", unit="protein"):
            if error is not None:
                logger.warning(f"Failed SS evaluation for id={row_id}: {error}")
                continue
            if merged_row is None:
                continue
            buffer.append(merged_row)
            processed += 1
            if len(buffer) >= int(args.flush_every):
                flushed = append_rows(output_csv, output_fieldnames, buffer)
                buffer.clear()
                logger.info(f"flushed={flushed} processed={processed}")

    if buffer:
        flushed = append_rows(output_csv, output_fieldnames, buffer)
        logger.info(f"final_flush={flushed} processed={processed}")

    logger.info("done")


if __name__ == "__main__":
    main()
