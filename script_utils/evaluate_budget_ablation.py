#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

root = os.path.abspath(".")
sys.path.append(root)

import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_residue_data, get_structure

from compute_stats import analyze_secondary_structure


STEP_RE = re.compile(r"step=(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all target checkpoints for one data-budget ablation run."
    )
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--noise_scale", type=float, default=0.45)
    parser.add_argument("--classic_lengths", default="50,100,150,200,250")
    parser.add_argument("--classic_samples_per_len", type=int, default=100)
    parser.add_argument("--classic_max_nsamples", type=int, default=5)
    parser.add_argument("--checkpoint_steps", default="6250,12500,25000,50000")
    parser.add_argument(
        "--output_root",
        default=None,
        help="Defaults to store/<run_name>/evaluation_budget",
    )
    parser.add_argument("--classic_seed", type=int, default=5)
    parser.add_argument("--distributional_seed", type=int, default=5)
    parser.add_argument("--distributional_seed_stride", type=int, default=1000003)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--foldseek_db_root",
        default="/homes/kasram/broteina/SiD_Protein/additional_files/foldseek_databases",
        help="Directory containing foldseek-prepared 'pdb' and 'afdb' databases.",
    )
    return parser.parse_args()


def run(cmd, env=None):
    subprocess.run(cmd, check=True, env=env)


def discover_checkpoints(run_dir: Path, target_steps: set[int]):
    ckpt_dir = run_dir / "checkpoints"
    rows = []
    all_non_ema = []
    for ckpt in sorted(ckpt_dir.glob("chk*.ckpt")):
        if ckpt.name.endswith("-EMA.ckpt"):
            continue
        m = STEP_RE.search(ckpt.name)
        if not m:
            continue
        step = int(m.group(1))
        all_non_ema.append((step, ckpt))

    all_non_ema.sort(key=lambda x: x[0])
    exact = {step: ckpt for step, ckpt in all_non_ema if step in target_steps}
    max_target = max(target_steps) if target_steps else None

    for target_step in sorted(target_steps):
        if target_step in exact:
            rows.append((target_step, target_step, exact[target_step]))
        elif target_step == max_target and all_non_ema:
            last_step, last_ckpt = all_non_ema[-1]
            rows.append((target_step, last_step, last_ckpt))

    return rows


def count_designability(sample_root: Path):
    pdb_dir = sample_root / "pdbs"
    designable_dir = pdb_dir / "designable"
    undesignable_dir = pdb_dir / "undesignable"
    designable = sorted(f for f in os.listdir(designable_dir) if f.endswith(".pdb"))
    undesignable = sorted(f for f in os.listdir(undesignable_dir) if f.endswith(".pdb"))
    return pdb_dir, designable, undesignable


def per_length_summary(designable, undesignable):
    summary = []
    lengths = sorted(
        {
            int(f.split("_")[0])
            for f in list(designable) + list(undesignable)
            if "_" in f
        }
    )
    for length in lengths:
        d = sum(1 for f in designable if f.startswith(f"{length}_"))
        u = sum(1 for f in undesignable if f.startswith(f"{length}_"))
        sampled = d + u
        summary.append(
            {
                "length": length,
                "sampled": sampled,
                "designable": d,
                "undesignable": u,
                "designability": (d / sampled) if sampled > 0 else 0.0,
            }
        )
    return summary


def evaluate_secondary_structure_dir(designable_dir: Path, designable):
    if not designable:
        return dict(alpha=np.nan, beta=np.nan, coil=np.nan)
    sec = np.zeros((3,), dtype=float)
    for f in designable:
        sec += analyze_secondary_structure(str(designable_dir / f))
    sec /= float(len(designable))
    return dict(alpha=float(sec[0]), beta=float(sec[1]), coil=float(sec[2]))


def evaluate_novelty_dir(designable_dir: Path, dataset: str, tmp_root: Path, db_root: Path):
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_file = tmp_root / f"novelty_{dataset}"
    if out_file.exists():
        out_file.unlink()
    db_path = db_root / dataset
    if not db_path.exists() and not Path(str(db_path) + ".dbtype").exists():
        raise FileNotFoundError(
            f"Foldseek database not found at {db_path}. "
            f"Pass --foldseek_db_root pointing to a directory containing '{dataset}' (and its sidecar files)."
        )
    run(
        [
            "foldseek",
            "easy-search",
            str(designable_dir),
            str(db_path),
            str(out_file),
            str(tmp_root),
            "--alignment-type",
            "1",
            "--exhaustive-search",
            "--tmscore-threshold",
            "0.0",
            "--max-seqs",
            "10000000000",
            "--format-output",
            "query,target,alntmscore,lddt",
        ],
        env=os.environ.copy(),
    )
    df = pd.read_csv(
        out_file,
        header=None,
        names=["protein", "target", "TM", "lddt"],
        sep="\t",
    )
    if len(df) == 0:
        return float("nan")
    tot_tm = 0.0
    for protein in df["protein"].unique():
        tot_tm += float(df[df["protein"] == protein]["TM"].max())
    return tot_tm / len(df["protein"].unique())


def evaluate_diversity_dir(designable_dir: Path, tmp_root: Path):
    tmp_root.mkdir(parents=True, exist_ok=True)
    res_prefix = tmp_root / "res"
    run(
        [
            "foldseek",
            "easy-cluster",
            str(designable_dir),
            str(res_prefix),
            str(tmp_root),
            "--alignment-type",
            "1",
            "--cov-mode",
            "0",
            "--min-seq-id",
            "0",
            "--tmscore-threshold",
            "0.5",
        ],
        env=os.environ.copy(),
    )
    cluster_tsv = tmp_root / "res_cluster.tsv"
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["cluster", "protein"])
    if len(df) == 0:
        return dict(diversity=np.nan, n_clusters=0)
    return dict(
        diversity=float(df["cluster"].nunique()) / float(len(df)),
        n_clusters=int(df["cluster"].nunique()),
    )


def evaluate_tm_diversity(designable_dir: Path, designable):
    if not designable:
        return {"avg_tm": np.nan}
    struc = {}
    for f in designable:
        struc[f] = get_structure(str(designable_dir / f))
    tm_sum = dict(zip(range(50, 251, 50), [0.0 for _ in range(5)]))
    tm_count = dict(zip(range(50, 251, 50), [0 for _ in range(5)]))
    for idx, f1 in enumerate(designable):
        nres = int(f1.split("_")[0])
        coords1, seq1 = get_residue_data(next(struc[f1].get_chains()))
        for f2 in designable[idx + 1 :]:
            if int(f2.split("_")[0]) == nres:
                coords2, seq2 = get_residue_data(next(struc[f2].get_chains()))
                result = tm_align(coords1, coords2, seq1, seq2)
                tm_sum[nres] += float(result.tm_norm_chain1)
                tm_count[nres] += 1
    per_length = {}
    vals = []
    for nres in range(50, 251, 50):
        avg = tm_sum[nres] / max(tm_count[nres], 1)
        per_length[f"tm_avg_L{nres}"] = float(avg)
        vals.append(float(avg))
    per_length["avg_tm"] = float(sum(vals) / len(vals))
    return per_length


def evaluate_classic(sample_root: Path, checkpoint_name: str, foldseek_db_root: Path):
    pdb_dir, designable, undesignable = count_designability(sample_root)
    designable_dir = pdb_dir / "designable"
    denom = len(designable) + len(undesignable)
    overall = {
        "classic_sample_root": str(sample_root),
        "designable_count": len(designable),
        "undesignable_count": len(undesignable),
        "designability": (len(designable) / denom) if denom > 0 else 0.0,
    }
    overall.update(evaluate_secondary_structure_dir(designable_dir, designable))
    overall.update(
        evaluate_diversity_dir(designable_dir, sample_root / "foldseek_tmp" / checkpoint_name)
    )
    overall["novelty_pdb"] = evaluate_novelty_dir(
        designable_dir,
        "pdb",
        sample_root / "foldseek_tmp" / f"{checkpoint_name}_pdb",
        foldseek_db_root,
    )
    overall["novelty_afdb"] = evaluate_novelty_dir(
        designable_dir,
        "afdb",
        sample_root / "foldseek_tmp" / f"{checkpoint_name}_afdb",
        foldseek_db_root,
    )
    overall.update(evaluate_tm_diversity(designable_dir, designable))
    per_length = per_length_summary(designable, undesignable)
    return overall, per_length


def launch_distributional_generation(ckpt_file: Path, output_root: Path, gpus, noise_scale, seed, seed_stride, expected_total=5000):
    samples_dir = output_root / "samples_fid_sharded"
    if samples_dir.is_dir():
        existing = sum(1 for _ in samples_dir.glob("*.pdb"))
        if existing >= expected_total:
            print(
                f"[evaluate_budget_ablation] Reusing existing distributional samples "
                f"({existing} pdbs) under {samples_dir}",
                flush=True,
            )
            return
        if existing > 0:
            print(
                f"[evaluate_budget_ablation] Found partial distributional samples "
                f"({existing}/{expected_total}) under {samples_dir}; regenerating from scratch.",
                flush=True,
            )
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    procs = []
    for split_id, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable,
            "script_utils/generate_distributional_checkpoint.py",
            "--ckpt_file",
            str(ckpt_file),
            "--output_root",
            str(output_root),
            "--config_name",
            "inference_fid_ca",
            "--split_id",
            str(split_id),
            "--num_splits",
            str(len(gpus)),
            "--seed",
            str(seed),
            "--seed_stride",
            str(seed_stride),
            "--noise_scale",
            str(noise_scale),
        ]
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        ret = p.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, p.args)


def evaluate_distributional(distributional_root: Path, metric_gpu: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = metric_gpu
    output_csv = distributional_root / "distributional_metrics.csv"
    output_json = distributional_root / "distributional_metrics.json"
    run(
        [
            sys.executable,
            "script_utils/evaluate_distributional_pdb_dir.py",
            "--data_dir",
            str(distributional_root / "samples_fid_sharded"),
            "--ca_only",
            "--batch_size",
            "12",
            "--num_workers",
            "32",
            "--output_csv",
            str(output_csv),
            "--output_json",
            str(output_json),
        ],
        env=env,
    )
    return pd.read_csv(output_csv).iloc[0].to_dict()


def write_csv(path: Path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    run_dir = Path("/homes/kasram/broteina/proteina/store") / args.run_name
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else run_dir / "evaluation_budget"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    steps = {int(x.strip()) for x in args.checkpoint_steps.split(",") if x.strip()}
    ckpts = discover_checkpoints(run_dir, steps)
    if not ckpts:
        raise SystemExit(f"No matching checkpoints found for {args.run_name} with steps {sorted(steps)}")

    foldseek_db_root = Path(args.foldseek_db_root)
    for dataset in ("pdb", "afdb"):
        candidate = foldseek_db_root / dataset
        if not candidate.exists() and not (foldseek_db_root / f"{dataset}.dbtype").exists():
            raise SystemExit(
                f"Foldseek database '{dataset}' not found under {foldseek_db_root}. "
                f"Pass --foldseek_db_root pointing to a directory containing the foldseek-prepared 'pdb' and 'afdb' databases."
            )

    overall_rows = []
    per_length_rows = []

    for requested_step, actual_step, ckpt_file in ckpts:
        ckpt_stem = ckpt_file.stem
        ckpt_eval_root = output_root / ckpt_stem
        classic_root = ckpt_eval_root / "classic"
        distributional_root = ckpt_eval_root / "distributional"
        ckpt_summary_csv = ckpt_eval_root / "checkpoint_summary.csv"

        if args.skip_existing and ckpt_summary_csv.exists():
            row = pd.read_csv(ckpt_summary_csv).iloc[0].to_dict()
            overall_rows.append(row)
            per_length_csv = ckpt_eval_root / "classic_per_length.csv"
            if per_length_csv.exists():
                dfl = pd.read_csv(per_length_csv)
                per_length_rows.extend(dfl.to_dict(orient="records"))
            continue

        classic_samples_present = (
            (classic_root / "pdbs" / "designable").is_dir()
            and (classic_root / "pdbs" / "undesignable").is_dir()
            and (
                any((classic_root / "pdbs" / "designable").glob("*.pdb"))
                or any((classic_root / "pdbs" / "undesignable").glob("*.pdb"))
            )
        )

        if not classic_samples_present:
            if classic_root.exists():
                shutil.rmtree(classic_root)
            run(
                [
                    sys.executable,
                    "proteinfoundation/model_designability.py",
                    "-c",
                    str(ckpt_file),
                    "--eval_length",
                    "short",
                    "--config_name",
                    "inference_base",
                    "--noise_scale",
                    str(args.noise_scale),
                    "--lengths",
                    args.classic_lengths,
                    "--nsamples_per_len",
                    str(args.classic_samples_per_len),
                    "--max_nsamples",
                    str(args.classic_max_nsamples),
                    "--seed",
                    str(args.classic_seed),
                    "--output_root",
                    str(classic_root),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)},
            )
        else:
            print(
                f"[evaluate_budget_ablation] Reusing existing classic samples under {classic_root}",
                flush=True,
            )

        # Clear stale foldseek_tmp so re-runs don't trip over partial outputs.
        foldseek_tmp = classic_root / "foldseek_tmp"
        if foldseek_tmp.exists():
            shutil.rmtree(foldseek_tmp)

        classic_overall, classic_per_length = evaluate_classic(
            classic_root, ckpt_stem, foldseek_db_root
        )
        for row in classic_per_length:
            row["run_name"] = args.run_name
            row["checkpoint_name"] = ckpt_stem
            row["requested_step"] = requested_step
            row["actual_step"] = actual_step
        per_length_rows.extend(classic_per_length)

        launch_distributional_generation(
            ckpt_file,
            distributional_root,
            gpus,
            args.noise_scale,
            args.distributional_seed,
            args.distributional_seed_stride,
        )
        distributional_metrics = evaluate_distributional(distributional_root, gpus[0])

        row = {
            "run_name": args.run_name,
            "checkpoint_name": ckpt_stem,
            "requested_step": requested_step,
            "actual_step": actual_step,
            "samples_reviewed_requested": int(requested_step) * 128,
            "samples_reviewed_actual": int(actual_step) * 128,
            **classic_overall,
            **distributional_metrics,
        }
        overall_rows.append(row)

        write_csv(ckpt_summary_csv, [row])
        write_csv(ckpt_eval_root / "classic_per_length.csv", classic_per_length)

    write_csv(output_root / "summary_overall.csv", overall_rows)
    write_csv(output_root / "summary_classic_per_length.csv", per_length_rows)


if __name__ == "__main__":
    main()
