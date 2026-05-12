#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


LENGTH_RE = re.compile(r"/pdbs/(\d+)/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic length-uniform dataset budget subsets."
    )
    parser.add_argument(
        "--input_csv",
        default="/homes/kasram/broteina/dataset/des/full_dataset/uniform_pdb.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/homes/kasram/broteina/dataset/des/full_dataset",
    )
    parser.add_argument(
        "--budgets",
        default="12500,50000,200000",
        help="Comma-separated target subset sizes. The full dataset alias is handled separately.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def extract_lengths(series: pd.Series) -> pd.Series:
    lengths = series.str.extract(LENGTH_RE)[0]
    if lengths.isna().any():
        bad = series[lengths.isna()].head(5).tolist()
        raise ValueError(f"Failed to infer lengths for some rows, examples: {bad}")
    return lengths.astype(int)


def budget_label(size: int) -> str:
    if size % 1000 == 0:
        return f"{size // 1000}k"
    if size == 12500:
        return "12k5"
    return str(size)


def sample_uniform_by_length(df: pd.DataFrame, lengths: pd.Series, target_size: int, seed: int):
    unique_lengths = np.array(sorted(lengths.unique().tolist()))
    n_lengths = len(unique_lengths)
    base = target_size // n_lengths
    remainder = target_size % n_lengths

    if base <= 0:
        raise ValueError(
            f"Target size {target_size} is too small for {n_lengths} lengths."
        )

    counts_by_length = lengths.value_counts().to_dict()
    if any(counts_by_length[length] < base + 1 for length in unique_lengths[:remainder]):
        raise ValueError("Some lengths do not have enough rows to satisfy the requested target.")

    rng = np.random.default_rng(seed)
    extra_lengths = set(rng.choice(unique_lengths, size=remainder, replace=False).tolist())

    sampled_parts = []
    allocation = []
    for length in unique_lengths:
        n_take = base + (1 if length in extra_lengths else 0)
        cur_df = df[lengths == length]
        sampled = cur_df.sample(n=n_take, replace=False, random_state=seed + int(length))
        sampled_parts.append(sampled)
        allocation.append({"length": int(length), "count": int(n_take)})

    out_df = pd.concat(sampled_parts, axis=0, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    assert len(out_df) == target_size, (len(out_df), target_size)
    return out_df, allocation


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    lengths = extract_lengths(df["input_path"])

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    manifest_rows = []
    allocation_dir = out_dir / "data_budget_allocations"
    allocation_dir.mkdir(parents=True, exist_ok=True)

    for size in budgets:
        label = budget_label(size)
        out_csv = out_dir / f"uniform_pdb_{label}.csv"
        out_df, allocation = sample_uniform_by_length(df, lengths, size, args.seed + size)
        out_df.to_csv(out_csv, index=False)

        alloc_json = allocation_dir / f"uniform_pdb_{label}_allocation.json"
        alloc_json.write_text(json.dumps(allocation, indent=2))
        manifest_rows.append(
            {
                "budget_label": label,
                "target_rows": size,
                "actual_rows": len(out_df),
                "source_csv": str(input_csv),
                "output_csv": str(out_csv),
                "allocation_json": str(alloc_json),
                "seed": args.seed + size,
                "n_lengths": lengths.nunique(),
                "min_per_length": min(item["count"] for item in allocation),
                "max_per_length": max(item["count"] for item in allocation),
            }
        )

    # 800k budget aliases the full uniform dataset used in this repo.
    full_label = "800k"
    full_csv = out_dir / f"uniform_pdb_{full_label}.csv"
    df.to_csv(full_csv, index=False)
    full_allocation = (
        pd.DataFrame({"length": sorted(lengths.unique())})
        .assign(count=lambda x: x["length"].map(lengths.value_counts().to_dict()))
        .to_dict(orient="records")
    )
    full_alloc_json = allocation_dir / f"uniform_pdb_{full_label}_allocation.json"
    full_alloc_json.write_text(json.dumps(full_allocation, indent=2))
    manifest_rows.append(
        {
            "budget_label": full_label,
            "target_rows": 800000,
            "actual_rows": len(df),
            "source_csv": str(input_csv),
            "output_csv": str(full_csv),
            "allocation_json": str(full_alloc_json),
            "seed": "",
            "n_lengths": lengths.nunique(),
            "min_per_length": int(lengths.value_counts().min()),
            "max_per_length": int(lengths.value_counts().max()),
        }
    )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("target_rows").reset_index(drop=True)
    manifest_path = out_dir / "uniform_pdb_data_budget_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(manifest_df.to_string(index=False))
    print(f"\nWrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
