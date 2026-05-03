#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Generate a synthetic dataset of (motif, scaffold) pairs from AFDB chunks.

Prerequisites:
    Run script_utils/prepare_afdb_chunks.py once to split and convert AFDB CIFs.

Data loading  : configs/datasets_config/afdb/d_FS_chunk.yaml  (--dataset_config)
Inference     : configs/experiment_config/inference_motif.yaml (--inference_config)

Designed to run one local process per GPU. Each process handles one pre-split
chunk via PDBLightningDataModule, which reads cached processed .pt files.

Output per pair (one .pt file):
    ca_coords        [nres, 3]  float32  Å  — generated CA backbone
    motif_seq_mask   [nres]     bool        — True at motif residue positions
    motif_ca_coords  [nres, 3]  float32  Å  — source CA (zero at non-motif)
    source_id        str
    nres             int

Output layout:
    <output_dir>/chunk_<id>/<source_id>.pt
    <output_dir>/chunk_<id>/metadata.csv
    <output_dir>/chunk_<id>/metadata_nodeXX.csv

Usage (single GPU test):
    python script_utils/generate_motif_scaffold_dataset.py \\
        --chunk_id 0 --n_chunks 8 \\
        --output_dir ./synthetic_motif_scaffold_dataset

Local 8-GPU launch:
    bash script_utils/generate_motif_scaffold_dataset.sh
"""

import argparse
import atexit
from contextlib import nullcontext
import hashlib
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

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
        help="Which chunk this job processes (0-indexed).",
    )
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Total number of cooperating nodes writing into the same output_dir.",
    )
    p.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="This node's rank in [0, num_nodes - 1].",
    )
    p.add_argument(
        "--seed_base",
        type=int,
        default=None,
        help="Optional override for the inference config seed base.",
    )
    p.add_argument(
        "--seed_stride",
        type=int,
        default=1000,
        help="Per-node seed stride added to the base seed.",
    )
    p.add_argument(
        "--amp_dtype",
        choices=["config", "fp32", "bf16"],
        default="config",
        help="Inference autocast mode. 'config' reads cfg_inf.precision, defaulting to fp32.",
    )
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


def _assigned_to_node(source_id: str, num_nodes: int, node_rank: int) -> bool:
    if num_nodes == 1:
        return True
    digest = hashlib.md5(source_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_nodes == node_rank


def _resolve_autocast_mode(args, cfg_inf):
    requested = args.amp_dtype
    if requested == "config":
        requested = str(cfg_inf.get("precision", "fp32")).lower()
    if requested == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            logger.warning("bf16 was requested but is not supported on this CUDA device; falling back to fp32")
            return "fp32"
        torch.set_float32_matmul_precision("medium")
        return "bf16"
    return "fp32"


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
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if not (0 <= args.chunk_id < args.n_chunks):
        raise ValueError(f"--chunk_id must be in [0, {args.n_chunks - 1}], got {args.chunk_id}")
    if args.num_nodes < 1:
        raise ValueError(f"--num_nodes must be >= 1, got {args.num_nodes}")
    if not (0 <= args.node_rank < args.num_nodes):
        raise ValueError(
            f"--node_rank must be in [0, {args.num_nodes - 1}], got {args.node_rank}"
        )

    data_path = os.environ.get("DATA_PATH", str(REPO_ROOT / "proteina_additional_files"))
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    os.environ["DATA_PATH"] = str(data_path)

    assert torch.cuda.is_available(), "CUDA is required for inference"
    device = torch.device("cuda")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    chunk_out_dir = output_dir / f"chunk_{args.chunk_id:02d}"
    chunk_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_name = (
        "metadata.csv"
        if args.num_nodes == 1 and args.node_rank == 0
        else f"metadata_nodes{args.num_nodes:02d}_rank{args.node_rank:02d}.csv"
    )
    logger.add(chunk_out_dir / f"generation_node{args.node_rank:02d}.log", level="INFO")
    logger.info(
        f"Chunk {args.chunk_id}/{args.n_chunks} → {chunk_out_dir} "
        f"(node_rank={args.node_rank}/{args.num_nodes}, metadata={metadata_name})"
    )

    # ------------------------------------------------------------------
    # Load dataset config, override data_dir for this chunk
    # ------------------------------------------------------------------
    chunk_name = f"chunk_{args.chunk_id:02d}"
    chunk_data_dir = data_path / "d_FS" / chunk_name
    if not chunk_data_dir.exists():
        raise FileNotFoundError(f"AFDB chunk directory not found: {chunk_data_dir}")

    default_data_overrides = [
        f"datamodule.data_dir={chunk_data_dir}",
        f"datamodule.datasplitter.data_dir={chunk_data_dir}",
        f"datamodule.file_identifier={chunk_name}",
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
    autocast_mode = _resolve_autocast_mode(args, cfg_inf)
    logger.info(f"Inference precision mode: {autocast_mode}")

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

    base_seed = cfg_inf.get("seed", 42) if args.seed_base is None else args.seed_base
    run_seed = int(base_seed) + args.chunk_id + args.seed_stride * args.node_rank
    logger.info(
        f"Seeding generation with base_seed={base_seed}, chunk_id={args.chunk_id}, "
        f"node_rank={args.node_rank}, seed_stride={args.seed_stride} -> seed={run_seed}"
    )
    L.seed_everything(run_seed)

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
    metadata_path = chunk_out_dir / metadata_name
    if metadata_path.exists():
        metadata_rows = pd.read_csv(metadata_path).to_dict("records")
    else:
        metadata_rows = []

    # source_ids already recorded in this node's metadata CSV
    recorded_source_ids = {str(row["source_id"]) for row in metadata_rows}

    # All .pt files on disk (cross-run, cross-node) — primary skip guard
    completed_source_ids = {path.stem for path in chunk_out_dir.glob("*.pt")}
    # Also include any metadata entries whose .pt has been deleted (edge case)
    completed_source_ids.update(
        str(row["source_id"])
        for row in metadata_rows
        if (chunk_out_dir / str(row.get("pt_file", f'{row["source_id"]}.pt'))).exists()
    )

    # Reconstruct metadata rows for .pt files that were saved but never flushed
    # (happens when the job is killed before the periodic/exit flush fires).
    orphan_ids = completed_source_ids - recorded_source_ids
    if orphan_ids:
        logger.info(
            f"Resuming: found {len(orphan_ids)} .pt files with no metadata row — "
            "reconstructing from saved tensors."
        )

        def _reconstruct(orphan_id: str):
            pt_path = chunk_out_dir / f"{orphan_id}.pt"
            if not pt_path.exists():
                return None
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                nres = int(data.get("nres", data["ca_coords"].shape[0]))
                n_motif = int(data["motif_seq_mask"].sum().item())
                return {
                    "source_id":        orphan_id,
                    "nres":             nres,
                    "n_motif_residues": n_motif,
                    "motif_pct":        round(n_motif / nres, 4) if nres > 0 else 0.0,
                    "pt_file":          f"{orphan_id}.pt",
                }
            except Exception as e:
                logger.warning(f"Could not reconstruct metadata for {orphan_id}: {e}")
                return None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:
            recovered = list(tqdm(
                ex.map(_reconstruct, sorted(orphan_ids)),
                total=len(orphan_ids),
                desc="recover-orphans",
                unit="file",
            ))
        for row in recovered:
            if row is not None:
                row["pair_id"] = len(metadata_rows)
                metadata_rows.append(row)

    pair_id = len(metadata_rows)
    initial_pair_id = pair_id
    batch_failures = 0

    # Flush the recovered metadata immediately so reconstruction is durable.
    if orphan_ids:
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
        logger.info(
            f"Persisted recovered metadata: {len(metadata_rows)} rows "
            f"({len(orphan_ids)} reconstructed)"
        )

    def _flush_metadata():
        if metadata_rows:
            pd.DataFrame(metadata_rows).to_csv(
                metadata_path, index=False
            )
            logger.info(f"Flushed {metadata_path.name} ({len(metadata_rows)} rows)")

    atexit.register(_flush_metadata)
    signal.signal(signal.SIGTERM, lambda s, f: (_flush_metadata(), sys.exit(0)))

    # ------------------------------------------------------------------
    # Pre-filter the dataset: skip proteins this node won't generate
    # (already completed OR assigned to a different node). This cuts disk
    # I/O dramatically — without it, every epoch re-loads every cached .pt
    # only to throw it away in shard_idx.
    # ------------------------------------------------------------------
    ds = loader.dataset
    n_before = len(ds)

    keep_mask = [
        (fname not in completed_source_ids)
        and _assigned_to_node(fname, args.num_nodes, args.node_rank)
        for fname in ds.file_names
    ]
    if not all(keep_mask):
        ds.file_names = [f for f, k in zip(ds.file_names, keep_mask) if k]
        ds.pdb_codes = [p for p, k in zip(ds.pdb_codes, keep_mask) if k]
        if ds.chains is not None:
            ds.chains = [c for c, k in zip(ds.chains, keep_mask) if k]
        if ds.in_memory and getattr(ds, "data", None) is not None:
            ds.data = [d for d, k in zip(ds.data, keep_mask) if k]
    logger.info(
        f"Dataset pre-filter: {n_before} → {len(ds)} proteins "
        f"(skipped {n_before - len(ds)} completed-or-other-node)"
    )

    if len(ds) == 0:
        logger.info("Nothing to generate for this shard; exiting.")
        _flush_metadata()
        return

    # Rebuild the dataloader against the filtered dataset (the cached
    # RandomSampler holds a stale len()).
    loader = datamodule.train_dataloader()

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
            sub_ids = [str(batch.id[i]) for i in valid_idx]
        else:
            sub_ids = [f"unk_{pair_id + k}" for k in range(len(valid_idx))]

        shard_idx = [
            i for i, source_id in enumerate(sub_ids)
            if _assigned_to_node(source_id, args.num_nodes, args.node_rank)
            and source_id not in completed_source_ids
        ]

        if not shard_idx:
            continue

        sub_coords = sub_coords[shard_idx]
        sub_cmask = sub_cmask[shard_idx]
        max_nres = max(sub_lengths[i] for i in shard_idx)
        sub_coords = sub_coords[:, :max_nres]
        sub_cmask = sub_cmask[:, :max_nres]
        sub_batch = {
            "coords": sub_coords,
            "mask_dict": {"coords": sub_cmask},
        }
        sub_lengths = [sub_lengths[i] for i in shard_idx]
        sub_ids = [sub_ids[i] for i in shard_idx]
        batch_sub = len(sub_lengths)

        # Run generation on this sub-batch
        try:
            motif_info = motif_factory.create_batch_motif(sub_batch)
            seq_mask = motif_info["fixed_sequence_mask"].cpu()
            x_motif  = motif_info["x_motif"].cpu()

            overall_mask = torch.zeros(batch_sub, max_nres, dtype=torch.bool)
            for k, nres_k in enumerate(sub_lengths):
                overall_mask[k, :nres_k] = True

            sc = cfg_inf.sampling_caflow
            sched = cfg_inf.schedule

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if autocast_mode == "bf16"
                else nullcontext()
            )
            with torch.no_grad(), amp_ctx:
                x_gen = model.generate(
                    nsamples=batch_sub,
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
            batch_failures += 1
            logger.warning(f"Batch generation failed: {e}")
            continue

        for k, (nres_k, source_id) in enumerate(zip(sub_lengths, sub_ids)):
            source_id = str(source_id)
            if source_id in completed_source_ids:
                continue

            motif_mask_k = seq_mask[k, :nres_k]
            if motif_mask_k.sum() == 0:
                continue

            ca_gen_ang    = nm_to_ang(x_gen[k, :nres_k]).cpu().float()        # [nres_k, 3] Å
            source_ca_ang = sub_coords[k, :nres_k, 1, :].cpu().float()        # [nres_k, 3] Å
            motif_ca_ang  = source_ca_ang * motif_mask_k.float().unsqueeze(-1)

            out_path = chunk_out_dir / f"{source_id}.pt"
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
            completed_source_ids.add(source_id)
            pair_id += 1

        # Flush after every batch — the CSV write is cheap relative to a
        # diffusion forward pass, and this bounds the orphan window to one
        # batch worth of work if the job is killed (incl. SIGKILL).
        if pair_id > initial_pair_id:
            _flush_metadata()

    new_pairs = pair_id - initial_pair_id
    logger.info(
        f"Done. total_pairs={pair_id} new_pairs={new_pairs} "
        f"batch_failures={batch_failures} saved_to={chunk_out_dir}"
    )
    _flush_metadata()
    if new_pairs == 0 and batch_failures > 0:
        raise RuntimeError(
            f"Generation produced zero new pairs and encountered {batch_failures} batch failures"
        )


if __name__ == "__main__":
    main()
