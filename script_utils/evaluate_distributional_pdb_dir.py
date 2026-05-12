#!/usr/bin/env python3
import argparse
import csv
import json
import os
import resource
import sys
from collections import OrderedDict

root = os.path.abspath(".")
sys.path.append(root)

import torch

# DataLoader workers pass tensors over UNIX sockets by default ("file_descriptor"
# sharing). With many workers + many small tensors this exhausts the per-process
# fd budget, surfacing as `RuntimeError: received 0 items of ancdata` deep inside
# multiprocessing reduction. Switch to memory-mapped sharing and raise the soft
# fd limit before any worker is spawned.
torch.multiprocessing.set_sharing_strategy("file_system")
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except (ValueError, OSError):
    pass

from dotenv import load_dotenv

from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--ca_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def to_float(v):
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def summarize_distributional_metrics(raw_metrics):
    paper = OrderedDict()

    def first_key(*candidates):
        for key in candidates:
            if key in raw_metrics:
                return key
        return None

    pdb_fid = first_key("PDB_FID", "PDB_FPSD")
    afdb_fid = first_key("AFDB_FID", "AFDB_FPSD", "D_FS_FID", "D_FS_FPSD")
    if pdb_fid:
        paper["FPSD_vs_PDB"] = to_float(raw_metrics[pdb_fid])
    if afdb_fid:
        paper["FPSD_vs_AFDB"] = to_float(raw_metrics[afdb_fid])

    for level in ("C", "A", "T"):
        key = first_key(f"fS_{level}", f"IS_{level}")
        if key:
            paper[f"fS_{level}"] = to_float(raw_metrics[key])

    for ref, prefixes in {
        "PDB": ("PDB",),
        "AFDB": ("AFDB", "D_FS"),
    }.items():
        vals = []
        for level in ("C", "A", "T"):
            key = first_key(*[f"{prefix}_fJSD_{level}" for prefix in prefixes])
            if key:
                value = to_float(raw_metrics[key])
                paper[f"fJSD_{level}_vs_{ref}"] = value
                vals.append(value)
        if len(vals) == 3:
            paper[f"fJSD_vs_{ref}"] = sum(vals) / 3.0

    return paper


def main():
    args = parse_args()
    load_dotenv()

    pdb_list = [
        os.path.join(args.data_dir, f)
        for f in sorted(os.listdir(args.data_dir))
        if f.endswith(".pdb")
    ]

    data_path = os.environ["DATA_PATH"]
    model_name = "gearnet_ca.pth" if args.ca_only else "gearnet.pth"
    ckpt_path = os.path.join(data_path, "metric_factory", "model_weights", model_name)
    feat_name = "%seval_ca_features.pth" if args.ca_only else "%seval_features.pth"
    feat_path = os.path.join(data_path, "metric_factory", "features", feat_name)

    raw = OrderedDict()
    for db in ["pdb_", "D_FS_"]:
        metric_factory = GenerationMetricFactory(
            ckpt_path=ckpt_path,
            ca_only=args.ca_only,
            metrics=["FID", "fJSD_C", "fJSD_A", "fJSD_T"],
            real_features_path=feat_path % db,
            reset_real_features=False,
            prefix=db.upper(),
        ).cuda()
        metric = generation_metric_from_list(
            pdb_list,
            metric_factory,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            verbose=True,
        )
        for k, v in metric.items():
            raw[k] = to_float(v)

    metric_factory = GenerationMetricFactory(
        ckpt_path=ckpt_path,
        ca_only=args.ca_only,
        metrics=["fS_C", "fS_A", "fS_T"],
        real_features_path=None,
        reset_real_features=False,
    ).cuda()
    metric = generation_metric_from_list(
        pdb_list,
        metric_factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=True,
    )
    for k, v in metric.items():
        raw[k] = to_float(v)

    paper = summarize_distributional_metrics(raw)
    print("Raw metrics:")
    print(raw)
    print("\nPaper-style distributional metrics:")
    print(paper)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"raw_metrics": raw, "paper_metrics": paper}, f, indent=2)

    if args.output_csv:
        fields = list(paper.keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(paper)


if __name__ == "__main__":
    main()
