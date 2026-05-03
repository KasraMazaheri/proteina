#!/usr/bin/env python

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


DEFAULT_DATASET_ROOT = Path("/homes/kasram/broteina/dataset/motif_scaffold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate motif_scaffold chunk outputs into a consolidated CSV manifest."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Root motif_scaffold directory that contains raw/chunk_*/",
    )
    parser.add_argument(
        "--output-name",
        default="motif_scaffold_pdb.csv",
        help="Output manifest filename written into dataset-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    raw_root = dataset_root / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    rows = []
    for pt_path in sorted(raw_root.glob("chunk_*/*.pt")):
        relative_path = pt_path.relative_to(raw_root)
        file_id = pt_path.stem
        rows.append(
            {
                "input_path": str(relative_path),
                "pdb": file_id,
                "id": file_id,
                "source_id": file_id,
                "chunk": relative_path.parts[0],
            }
        )

    if not rows:
        raise ValueError(f"No motif scaffold .pt files found under {raw_root}")

    df = pd.DataFrame(rows, columns=["input_path", "pdb", "id", "source_id", "chunk"])
    output_path = dataset_root / args.output_name
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
