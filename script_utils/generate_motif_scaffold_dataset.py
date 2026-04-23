#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Generate a synthetic dataset of (motif, scaffold) pairs from AFDB chunks.

Prerequisites:
    Run script_utils/prepare_afdb_chunks.py once to split and convert AFDB CIFs.

Data loading  : configs/datasets_config/afdb/d_FS_chunk.yaml  (--dataset_config)
Inference     : configs/experiment_config/inference_motif.yaml (--inference_config)

Designed to run as a SLURM job array (array=0-7). Each job processes one
pre-split chunk via PDBLightningDataModule, which handles raw→processed .pt
caching automatically.

Output per pair (one .pt file):
    ca_coords        [nres, 3]  float32  Å  — generated CA backbone
    motif_seq_mask   [nres]     bool        — True at motif residue positions
    motif_ca_coords  [nres, 3]  float32  Å  — source CA (zero at non-motif)
    source_id        str
    nres             int

Output layout:
    <output_dir>/chunk_<id>/pair_<xxxxxxxx>.pt
    <output_dir>/chunk_<id>/metadata.csv

Usage (single GPU test):
    python script_utils/generate_motif_scaffold_dataset.py \\
        --chunk_id 0 --n_chunks 8 \\
        --output_dir ./synthetic_motif_scaffold_dataset

SLURM array (8 GPUs):
    sbatch script_utils/generate_motif_scaffold_dataset.sh
"""

import argparse
import atexit
import os
import signal
import sys

root = os.path.abspath(".")
sys.path.append(root)

import numpy as np
import pandas as pd
import torch
import hydra
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from proteinfoundation.nn.motif_factory import SingleMotifFactory
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.coors_utils import nm_to_ang


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate (motif, scaffold) pairs from chunked AFDB PDB files"
    )
    p.add_argument(
        "--dataset_config",
        default="d_FS_chunk",
        help="Dataset config name under configs/datasets_config/afdb/",
    )
    p.add_argument(
        "--inference_config",
        default="inference_motif",
        help="Inference config name under configs/experiment_config/",
    )
    p.add_argument("--output_dir", default="./synthetic_motif_scaffold_dataset")
    p.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="Which chunk this job processes (0-indexed). "
             "Auto-set from $SLURM_ARRAY_TASK_ID if present.",
    )
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Proteins per GPU mini-batch during generation (overrides config).",
    )
    p.add_argument("--min_protein_length", type=int, default=50)
    p.add_argument("--max_protein_length", type=int, default=256)
    # Motif factory parameters
    p.add_argument("--motif_min_pct", type=float, default=0.1)
    p.add_argument("--motif_max_pct", type=float, default=0.4)
    p.add_argument("--motif_min_seg", type=int, default=1)
    p.add_argument("--motif_max_seg", type=int, default=4)
    # Pass-through Hydra overrides
    p.add_argument("--data_overrides", nargs="*", default=[])
    p.add_argument("--inf_overrides", nargs="*", default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _protein_length(batch, idx: int) -> int:
    """Actual (non-padded) residue count for protein i in a DensePadding batch."""
    return int(batch.mask_dict["coords"][idx, :, 0, 0].sum().item())


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_padded_batch(model, motif_factory, batch, lengths, device, cfg_inf):
    """
    Run motif extraction and scaffold generation for a padded DensePadding batch.

    Returns list of result dicts (or None for proteins with no motif residues),
    one per protein in the batch.
    """
    sc = cfg_inf.sampling_caflow
    sched = cfg_inf.schedule

    B = batch.coords.shape[0]
    max_nres = max(lengths)

    # Trim batch tensors to max_nres to save memory/compute
    coords_trim = batch.coords[:, :max_nres].float()            # [B, max_nres, 37, 3]
    coord_mask_trim = batch.mask_dict["coords"][:, :max_nres]  # [B, max_nres, 37, 3]

    fake_batch = {
        "coords": coords_trim,
        "mask_dict": {"coords": coord_mask_trim},
    }

    motif_info = motif_factory.create_batch_motif(fake_batch)
    seq_mask = motif_info["fixed_sequence_mask"].cpu()  # [B, max_nres] bool
    x_motif  = motif_info["x_motif"].cpu()              # [B, max_nres, 3] nm (centered)

    # Overall padding mask for the padded batch
    overall_mask = torch.zeros(B, max_nres, dtype=torch.bool)
    for i, nres_i in enumerate(lengths):
        overall_mask[i, :nres_i] = True

    with torch.no_grad():
        x_gen = model.generate(
            nsamples=B,
            n=max_nres,
            dt=cfg_inf.dt,
            self_cond=cfg_inf.self_cond,
            cath_code=None,
            mask=overall_mask.to(device),
            x_motif=x_motif.to(device),
            fixed_sequence_mask=seq_mask.to(device),
            fixed_structure_mask=(seq_mask[:, :, None] * seq_mask[:, None, :]).to(device),
            dtype=torch.float32,
            schedule_mode=sched.schedule_mode,
            schedule_p=sched.schedule_p,
            sampling_mode=sc.sampling_mode,
            sc_scale_noise=sc.sc_scale_noise,
            sc_scale_score=sc.sc_scale_score,
            gt_mode=sc.gt_mode,
            gt_p=sc.gt_p,
            gt_clamp_val=sc.gt_clamp_val,
        )  # [B, max_nres, 3] nm

    results = []
    for i, nres_i in enumerate(lengths):
        motif_mask_i = seq_mask[i, :nres_i]       # [nres_i] bool
        if motif_mask_i.sum() == 0:
            results.append(None)
            continue

        ca_gen_ang     = nm_to_ang(x_gen[i, :nres_i]).cpu().float()   # [nres_i, 3] Å
        source_ca_ang  = coords_trim[i, :nres_i, 1, :].cpu().float()  # [nres_i, 3] Å (CA = OF index 1)
        motif_ca_ang   = source_ca_ang * motif_mask_i.float().unsqueeze(-1)

        results.append({
            "ca_coords":       ca_gen_ang,
            "motif_seq_mask":  motif_mask_i,
            "motif_ca_coords": motif_ca_ang,
            "nres":            nres_i,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    args = parse_args()

    # SLURM_ARRAY_TASK_ID overrides --chunk_id
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task is not None:
        args.chunk_id = int(slurm_task)

    assert torch.cuda.is_available(), "CUDA is required for inference"
    device = torch.device("cuda")

    chunk_out_dir = os.path.join(args.output_dir, f"chunk_{args.chunk_id:02d}")
    os.makedirs(chunk_out_dir, exist_ok=True)
    logger.add(os.path.join(chunk_out_dir, "generation.log"), level="INFO")
    logger.info(f"Chunk {args.chunk_id}/{args.n_chunks} → {chunk_out_dir}")

    # ------------------------------------------------------------------
    # Load dataset config, override data_dir for this chunk
    # ------------------------------------------------------------------
    data_path = os.environ.get("DATA_PATH", "./proteina_additional_files")
    chunk_data_dir = os.path.join(data_path, "d_FS", f"chunk_{args.chunk_id:02d}")

    default_data_overrides = [
        f"datamodule.data_dir={chunk_data_dir}",
        f"datamodule.datasplitter.data_dir={chunk_data_dir}",
        "datamodule.overwrite=False",
        f"datamodule.batch_size={args.batch_size}",
        "datamodule.num_workers=4",
    ]
    data_overrides = default_data_overrides + (args.data_overrides or [])

    with hydra.initialize(
        config_path="../configs/datasets_config/afdb",
        version_base=hydra.__version__,
    ):
        cfg_data = hydra.compose(
            config_name=args.dataset_config,
            overrides=data_overrides,
        )

    logger.info(f"Dataset config:\n{cfg_data}")
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")

    loader = datamodule.train_dataloader()
    logger.info(f"Train split: {len(loader.dataset)} proteins")

    # ------------------------------------------------------------------
    # Load inference config and model
    # ------------------------------------------------------------------
    inf_overrides = args.inf_overrides or []
    with hydra.initialize(
        config_path="../configs/experiment_config",
        version_base=hydra.__version__,
    ):
        cfg_inf = hydra.compose(
            config_name=args.inference_config,
            overrides=inf_overrides,
        )

    logger.info(f"Inference config:\n{cfg_inf}")

    ckpt_file = os.path.join(cfg_inf.ckpt_path, cfg_inf.ckpt_name)
    assert os.path.isfile(ckpt_file), f"Checkpoint not found: {ckpt_file}"

    logger.info(f"Loading model from {ckpt_file}")
    model: Proteina = Proteina.load_from_checkpoint(ckpt_file, map_location="cpu")
    model = model.to(device)
    model.eval()

    assert model.motif_conditioning, (
        "Checkpoint does not support motif conditioning. "
        "Use proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt"
    )

    import lightning as L
    L.seed_everything(cfg_inf.get("seed", 42) + args.chunk_id)

    # ------------------------------------------------------------------
    # Motif factory
    # ------------------------------------------------------------------
    motif_factory = SingleMotifFactory(
        motif_prob=1.0,
        motif_min_pct_res=args.motif_min_pct,
        motif_max_pct_res=args.motif_max_pct,
        motif_min_n_seg=args.motif_min_seg,
        motif_max_n_seg=args.motif_max_seg,
    )

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    metadata_rows = []
    pair_id = 0

    def _flush_metadata():
        if metadata_rows:
            pd.DataFrame(metadata_rows).to_csv(
                os.path.join(chunk_out_dir, "metadata.csv"), index=False
            )
            logger.info(f"Flushed metadata.csv ({len(metadata_rows)} rows)")

    atexit.register(_flush_metadata)
    signal.signal(signal.SIGTERM, lambda s, f: (_flush_metadata(), sys.exit(0)))

    for batch in tqdm(loader, desc=f"chunk {args.chunk_id}", unit="batch"):
        # Compute per-protein lengths and filter by length bounds
        B = batch.coords.shape[0]
        lengths = [_protein_length(batch, i) for i in range(B)]
        valid_idx = [
            i for i, n in enumerate(lengths)
            if args.min_protein_length <= n <= args.max_protein_length
        ]

        if not valid_idx:
            continue

        # Subset batch to valid proteins using a fake_batch dict
        # (avoids rebuilding a full PyG batch object)
        max_nres = max(lengths[i] for i in valid_idx)
        sub_coords = batch.coords[valid_idx, :max_nres].float()
        sub_cmask  = batch.mask_dict["coords"][valid_idx, :max_nres]
        sub_batch = {
            "coords":    sub_coords,
            "mask_dict": {"coords": sub_cmask},
        }
        sub_lengths = [lengths[i] for i in valid_idx]

        # Recover protein IDs (graphein stores them in batch.id as a list of strings)
        if hasattr(batch, "id"):
            sub_ids = [batch.id[i] for i in valid_idx]
        else:
            sub_ids = [f"unk_{pair_id + k}" for k in range(len(valid_idx))]

        # Run generation on this sub-batch
        try:
            motif_info = motif_factory.create_batch_motif(sub_batch)
            seq_mask = motif_info["fixed_sequence_mask"].cpu()
            x_motif  = motif_info["x_motif"].cpu()

            overall_mask = torch.zeros(len(valid_idx), max_nres, dtype=torch.bool)
            for k, nres_k in enumerate(sub_lengths):
                overall_mask[k, :nres_k] = True

            sc = cfg_inf.sampling_caflow
            sched = cfg_inf.schedule

            with torch.no_grad():
                x_gen = model.generate(
                    nsamples=len(valid_idx),
                    n=max_nres,
                    dt=cfg_inf.dt,
                    self_cond=cfg_inf.self_cond,
                    cath_code=None,
                    mask=overall_mask.to(device),
                    x_motif=x_motif.to(device),
                    fixed_sequence_mask=seq_mask.to(device),
                    fixed_structure_mask=(
                        seq_mask[:, :, None] * seq_mask[:, None, :]
                    ).to(device),
                    dtype=torch.float32,
                    schedule_mode=sched.schedule_mode,
                    schedule_p=sched.schedule_p,
                    sampling_mode=sc.sampling_mode,
                    sc_scale_noise=sc.sc_scale_noise,
                    sc_scale_score=sc.sc_scale_score,
                    gt_mode=sc.gt_mode,
                    gt_p=sc.gt_p,
                    gt_clamp_val=sc.gt_clamp_val,
                )  # [B_sub, max_nres, 3] nm

        except Exception as e:
            logger.warning(f"Batch generation failed: {e}")
            continue

        for k, (nres_k, source_id) in enumerate(zip(sub_lengths, sub_ids)):
            motif_mask_k = seq_mask[k, :nres_k]
            if motif_mask_k.sum() == 0:
                continue

            ca_gen_ang    = nm_to_ang(x_gen[k, :nres_k]).cpu().float()        # [nres_k, 3] Å
            source_ca_ang = sub_coords[k, :nres_k, 1, :].cpu().float()        # [nres_k, 3] Å
            motif_ca_ang  = source_ca_ang * motif_mask_k.float().unsqueeze(-1)

            out_path = os.path.join(chunk_out_dir, f"{source_id}.pt")
            torch.save(
                {
                    "ca_coords":       ca_gen_ang,
                    "motif_seq_mask":  motif_mask_k,
                    "motif_ca_coords": motif_ca_ang,
                    "source_id":       source_id,
                    "nres":            nres_k,
                },
                out_path,
            )

            n_motif = int(motif_mask_k.sum().item())
            metadata_rows.append({
                "pair_id":          pair_id,
                "source_id":        source_id,
                "nres":             nres_k,
                "n_motif_residues": n_motif,
                "motif_pct":        round(n_motif / nres_k, 4),
                "pt_file":          f"{source_id}.pt",
            })
            pair_id += 1

        if pair_id > 0 and pair_id % 500 == 0:
            _flush_metadata()

    logger.info(f"Done. {pair_id} pairs saved to {chunk_out_dir}")
    _flush_metadata()


if __name__ == "__main__":
    main()
