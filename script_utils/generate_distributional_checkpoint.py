#!/usr/bin/env python3
import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

root = os.path.abspath(".")
sys.path.append(root)

import hydra
import lightning as L
import loralib as lora
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.model_designability import (
    GenDataset,
    parse_len_cath_code,
    parse_nlens_cfg,
)
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.utils.lora_utils import replace_lora_layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-protocol distributional samples for a checkpoint."
    )
    parser.add_argument("--ckpt_file", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--config_name", default="inference_fid_ca", type=str)
    parser.add_argument("--config_subdir", default=None, type=str)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_stride", type=int, default=1000003)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--clear_output_root", action="store_true")
    parser.add_argument("--lengths", type=str, default=None)
    parser.add_argument("--nsamples_per_len", type=int, default=None)
    parser.add_argument("--max_nsamples", type=int, default=None)
    return parser.parse_args()


def split_nlens_exact(nlens_dict, max_nsamples=16):
    lengths_range = nlens_dict["length_ranges"].tolist()
    length_distribution = nlens_dict["length_distribution"].tolist()
    lens_sample, nsamples = [], []
    for length, cnt in zip(lengths_range, length_distribution):
        remaining = int(cnt)
        while remaining > 0:
            take = min(int(max_nsamples), remaining)
            lens_sample.append(int(length))
            nsamples.append(int(take))
            remaining -= take
    return lens_sample, nsamples


def configure_cfg(args):
    config_path = (
        "../configs/experiment_config"
        if args.config_subdir is None
        else f"../configs/experiment_config/{args.config_subdir}"
    )
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.config_name)
    if args.noise_scale is not None:
        cfg.sampling_caflow.sc_scale_noise = float(args.noise_scale)
    if args.lengths:
        cfg.nres_lens = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
        cfg.min_len = None
        cfg.max_len = None
        cfg.step_len = None
    if args.nsamples_per_len is not None:
        cfg.nsamples_per_len = int(args.nsamples_per_len)
    if args.max_nsamples is not None:
        cfg.max_nsamples = int(args.max_nsamples)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    return cfg


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed)


def main():
    args = parse_args()
    load_dotenv()
    assert torch.cuda.is_available(), "CUDA not available"

    ckpt_file = str(Path(args.ckpt_file).resolve())
    assert os.path.exists(ckpt_file), f"Checkpoint not found: {ckpt_file}"

    cfg = configure_cfg(args)
    effective_seed = int(cfg.seed) + args.split_id * int(args.seed_stride)
    seed_all(effective_seed)

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )
    logger.info(f"Using checkpoint {ckpt_file}")
    logger.info(
        f"Distributional shard split_id={args.split_id} num_splits={args.num_splits} effective_seed={effective_seed}"
    )

    if not cfg.lora.use:
        model = Proteina.load_from_checkpoint(ckpt_file)
    else:
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
        replace_lora_layers(
            model,
            cfg["lora"]["r"],
            cfg["lora"]["lora_alpha"],
            cfg["lora"]["lora_dropout"],
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg["lora"]["train_bias"])
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    model.configure_inference(cfg, nn_ag=None)

    nlens_dict = parse_nlens_cfg(cfg)
    lens_sample, nsamples = split_nlens_exact(
        nlens_dict, max_nsamples=cfg.max_nsamples
    )
    if cfg.fold_cond:
        len_cath_codes = parse_len_cath_code(cfg)
    else:
        len_cath_codes = None

    shard_lens = []
    shard_nsamples = []
    for idx, (length, nsamp) in enumerate(zip(lens_sample, nsamples)):
        if idx % args.num_splits == args.split_id:
            shard_lens.append(int(length))
            shard_nsamples.append(int(nsamp))

    dataset = GenDataset(
        nres=shard_lens, nsamples=shard_nsamples, dt=cfg.dt, len_cath_codes=len_cath_codes
    )
    dataloader = DataLoader(dataset, batch_size=1)

    output_root = Path(args.output_root)
    if args.clear_output_root and args.split_id == 0 and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples_dir = output_root / ("samples_fid_sharded" if args.num_splits > 1 else "samples_fid")
    samples_dir.mkdir(parents=True, exist_ok=True)

    split_prefix = f"split{args.split_id:02d}_"
    for p in samples_dir.glob(f"{split_prefix}*.pdb"):
        p.unlink()

    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    counters = defaultdict(int)
    metadata_rows = []
    total_written = 0
    for pred, n in zip(predictions, shard_lens):
        coors_atom37 = pred
        for i in range(coors_atom37.shape[0]):
            idx = counters[int(n)]
            counters[int(n)] += 1
            fname = f"split{args.split_id:02d}_L{int(n):03d}_{idx:04d}_fid.pdb"
            pdb_path = samples_dir / fname
            write_prot_to_pdb(
                coors_atom37[i].numpy(),
                str(pdb_path),
                overwrite=True,
                no_indexing=True,
            )
            metadata_rows.append(
                {
                    "split_id": args.split_id,
                    "num_splits": args.num_splits,
                    "effective_seed": effective_seed,
                    "length": int(n),
                    "index_within_length_and_split": idx,
                    "pdb_path": str(pdb_path),
                }
            )
            total_written += 1

    metadata_path = output_root / f"metadata_split{args.split_id:02d}.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_id",
                "num_splits",
                "effective_seed",
                "length",
                "index_within_length_and_split",
                "pdb_path",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info(f"Wrote {total_written} samples to {samples_dir}")
    logger.info(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
