#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize designability per protein length from "
            "samples/neurips/<checkpoint>/pdbs/{designable,undesignable}."
        )
    )
    parser.add_argument(
        "-c",
        "--ckpt_name",
        required=True,
        help="Checkpoint directory name under the samples root.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("samples/neurips"),
        help="Root directory containing checkpoint sample folders.",
    )
    parser.add_argument(
        "--pdb-subdir",
        default="pdbs",
        help="PDB subdirectory inside the checkpoint folder.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="50,100,150,200,250",
        help=(
            "Optional comma-separated list of lengths to report, "
            "e.g. 50,100,150,200,250. If omitted, lengths are inferred."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint_dir(samples_root: Path, ckpt_name: str) -> Path:
    exact = samples_root / ckpt_name
    if exact.is_dir():
        return exact

    matches = sorted(p for p in samples_root.glob(f"{ckpt_name}*") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find checkpoint directory '{ckpt_name}' under {samples_root}"
        )
    raise RuntimeError(
        f"Checkpoint name '{ckpt_name}' is ambiguous under {samples_root}: "
        + ", ".join(p.name for p in matches)
    )


def count_ca_residues(pdb_path: Path) -> int:
    nres = 0
    with pdb_path.open("r") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                nres += 1
    return nres


def collect_counts(pdb_dir: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not pdb_dir.is_dir():
        return counts

    for pdb_path in sorted(pdb_dir.glob("*.pdb")):
        nres = count_ca_residues(pdb_path)
        counts[nres] += 1
    return counts


def parse_lengths(lengths_arg: str, designable: Counter[int], undesignable: Counter[int]) -> list[int]:
    if lengths_arg.strip():
        return [int(tok.strip()) for tok in lengths_arg.split(",") if tok.strip()]
    return sorted(set(designable) | set(undesignable))


def print_block(label: str, sampled: int, designable: int, indent: str = "") -> None:
    designability = (designable / sampled) if sampled > 0 else 0.0
    print(f"{indent}[{label}] Sampled: {sampled} proteins.")
    print(f"{indent}Designability: {designability:.3f}")
    print(f"{indent}Avg scRMSD:    N/A")


def main() -> int:
    args = parse_args()
    ckpt_dir = resolve_checkpoint_dir(args.samples_root, args.ckpt_name)
    base = ckpt_dir / args.pdb_subdir
    designable_dir = base / "designable"
    undesignable_dir = base / "undesignable"

    if not designable_dir.is_dir() or not undesignable_dir.is_dir():
        raise FileNotFoundError(
            f"Expected designable/undesignable folders under {base}"
        )

    designable_counts = collect_counts(designable_dir)
    undesignable_counts = collect_counts(undesignable_dir)
    lengths = parse_lengths(args.lengths, designable_counts, undesignable_counts)

    total_designable = sum(designable_counts.values())
    total_undesignable = sum(undesignable_counts.values())
    total_sampled = total_designable + total_undesignable

    print_block("Total", total_sampled, total_designable)
    for length in lengths:
        sampled = designable_counts[length] + undesignable_counts[length]
        print()
        print_block(
            f"Length {length}",
            sampled,
            designable_counts[length],
            indent="\t",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
