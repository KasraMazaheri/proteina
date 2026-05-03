#!/usr/bin/env python

import argparse
import copy
import math
import os
import random
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

import hydra
import lightning as L
import loralib as lora
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.utils.lora_utils import replace_lora_layers


CONFIG_DIR = root / "configs" / "experiment_config"
DEFAULT_CONFIG_NAME = "inference_uncond_dataset_base"


def ensure_repo_env() -> None:
    load_dotenv(root / ".env")
    data_path = os.environ.get("DATA_PATH")
    if data_path and not os.path.isabs(data_path):
        os.environ["DATA_PATH"] = str((root / data_path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch or run large-scale unconditional Proteina sampling jobs."
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Base config name under configs/experiment_config, without .yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional full path to the checkpoint to sample. Overrides the config value.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Minimum sequence length to sample, inclusive.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Maximum sequence length to sample, inclusive.",
    )
    parser.add_argument(
        "--step-len",
        type=int,
        default=None,
        help="Stride between sampled lengths.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=None,
        help="Total number of backbones to generate across all lengths.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum number of same-length proteins sampled in parallel per batch.",
    )
    parser.add_argument(
        "--samples-per-length-per-gpu",
        type=int,
        default=None,
        help=(
            "If set, each GPU generates exactly this many samples for every requested "
            "length. Overrides launcher.total_samples."
        ),
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="SDE noise scaling used for stochastic sampling.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling step size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed. Each worker uses seed + gpu_id.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU ids to use for tmux worker sessions.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory where dataset_<gpu_id>/ folders are created.",
    )
    parser.add_argument(
        "--session-prefix",
        default=None,
        help="Prefix for tmux session names.",
    )
    parser.add_argument(
        "--generated-config-dir",
        default=None,
        help="Directory where per-GPU config files are written.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory where per-GPU logs are written.",
    )
    parser.add_argument(
        "--self-cond",
        choices=["true", "false"],
        default=None,
        help="Override self-conditioning if needed for a different checkpoint.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing files. Worker exits on filename collisions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write worker configs and print tmux commands without launching them.",
    )
    parser.add_argument(
        "--worker-config",
        default=None,
        help="Internal worker mode: run a single GPU job from a generated config file.",
    )
    return parser.parse_args()


def load_base_config(config_name: str):
    with hydra.initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        version_base=hydra.__version__,
    ):
        return hydra.compose(config_name=config_name)


def resolve_lengths(cfg) -> List[int]:
    if cfg.get("nres_lens"):
        return [int(v) for v in cfg.nres_lens]

    min_len = int(cfg.min_len)
    max_len = int(cfg.max_len)
    step_len = int(cfg.step_len)
    if step_len <= 0:
        raise ValueError(f"step_len must be positive, got {step_len}")
    if min_len > max_len:
        raise ValueError(f"min_len must be <= max_len, got {min_len} > {max_len}")
    return list(range(min_len, max_len + 1, step_len))


def balanced_length_tasks(lengths: Sequence[int], total_samples: int) -> List[Dict[str, int]]:
    if total_samples < len(lengths):
        raise ValueError(
            f"total_samples={total_samples} is smaller than the number of lengths={len(lengths)}"
        )

    base = total_samples // len(lengths)
    if base <= 0:
        raise ValueError(
            f"Per-length count must be positive, got total_samples={total_samples} for {len(lengths)} lengths"
        )

    return [{"length": int(length), "count": int(base)} for length in lengths]


def fixed_tasks_per_gpu(
    lengths: Sequence[int], samples_per_length_per_gpu: int, gpu_ids: Sequence[int]
) -> List[Dict[str, int]]:
    if samples_per_length_per_gpu <= 0:
        raise ValueError(
            f"samples_per_length_per_gpu must be positive, got {samples_per_length_per_gpu}"
        )
    return [
        {
            "length": int(length),
            "count": int(samples_per_length_per_gpu) * len(gpu_ids),
        }
        for length in lengths
    ]


def assign_tasks_per_length(
    tasks: Sequence[Dict[str, int]], gpu_ids: Sequence[int]
) -> List[Tuple[int, List[Dict[str, int]]]]:
    per_gpu: List[List[Dict[str, int]]] = [[] for _ in gpu_ids]

    for task_idx, task in enumerate(tasks):
        count = int(task["count"])
        base = count // len(gpu_ids)
        remainder = count % len(gpu_ids)
        remainder_start = task_idx % len(gpu_ids)
        for gpu_idx in range(len(gpu_ids)):
            gets_extra = ((gpu_idx - remainder_start) % len(gpu_ids)) < remainder
            gpu_count = base + int(gets_extra)
            if gpu_count <= 0:
                continue
            per_gpu[gpu_idx].append({"length": int(task["length"]), "count": gpu_count})

    return list(zip(gpu_ids, per_gpu))


def build_worker_command(gpu_id: int, worker_cfg_path: Path, log_path: Path) -> str:
    quoted_root = shlex.quote(str(root))
    quoted_cfg = shlex.quote(str(worker_cfg_path))
    quoted_log = shlex.quote(str(log_path))
    return (
        'source "$(conda info --base)/etc/profile.d/conda.sh"'
        f" && conda activate proteina_env"
        f" && cd {quoted_root}"
        f" && export CUDA_VISIBLE_DEVICES={gpu_id}"
        f" && python script_utils/sample_uncond_dataset.py --worker-config {quoted_cfg}"
        f" > {quoted_log} 2>&1"
        " ; status=$?"
        ' ; if [ "$status" -ne 0 ]; then'
        ' echo "Worker failed with exit code $status. Showing log tail:"'
        f" ; tail -n 80 {quoted_log}"
        " ; exec bash"
        " ; fi"
    )


def launch_tmux_session(session_name: str, command: str) -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, f"bash -lc {shlex.quote(command)}"],
        check=True,
    )


def apply_launcher_overrides(cfg, args: argparse.Namespace):
    cfg = copy.deepcopy(cfg)

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        cfg.ckpt_path = str(ckpt_path.parent)
        cfg.ckpt_name = ckpt_path.name

    if args.min_len is not None:
        cfg.min_len = args.min_len
    if args.max_len is not None:
        cfg.max_len = args.max_len
    if args.step_len is not None:
        cfg.step_len = args.step_len
    if args.batch_size is not None:
        cfg.max_nsamples = args.batch_size
    if args.noise_scale is not None:
        cfg.sampling_caflow.sc_scale_noise = args.noise_scale
    if args.dt is not None:
        cfg.dt = args.dt
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gpus is not None:
        cfg.launcher.gpu_ids = list(args.gpus)
    if args.output_root is not None:
        cfg.launcher.output_root = args.output_root
    if args.session_prefix is not None:
        cfg.launcher.tmux_session_prefix = args.session_prefix
    if args.generated_config_dir is not None:
        cfg.launcher.generated_config_dir = args.generated_config_dir
    if args.log_dir is not None:
        cfg.launcher.log_dir = args.log_dir
    if args.total_samples is not None:
        cfg.launcher.total_samples = args.total_samples
    if args.samples_per_length_per_gpu is not None:
        cfg.launcher.samples_per_length_per_gpu = args.samples_per_length_per_gpu
    if args.self_cond is not None:
        cfg.self_cond = args.self_cond == "true"
    if args.no_resume:
        cfg.launcher.resume = False

    return cfg


def load_model(cfg):
    ckpt_file = Path(cfg.ckpt_path) / cfg.ckpt_name
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    if not cfg.lora.use:
        model = Proteina.load_from_checkpoint(str(ckpt_file))
    else:
        model = Proteina.load_from_checkpoint(str(ckpt_file), strict=False)
        replace_lora_layers(
            model,
            cfg.lora.r,
            cfg.lora.lora_alpha,
            cfg.lora.lora_dropout,
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg.lora.train_bias)
        ckpt = torch.load(str(ckpt_file), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    nn_ag = None
    if (
        cfg.get("autoguidance_ratio", 0.0) > 0
        and cfg.get("guidance_weight", 1.0) != 1.0
    ):
        if cfg.autoguidance_ckpt_path is None:
            raise ValueError("autoguidance_ckpt_path must be set when autoguidance is enabled")
        model_ag = Proteina.load_from_checkpoint(cfg.autoguidance_ckpt_path)
        nn_ag = model_ag.nn

    model.configure_inference(cfg, nn_ag=nn_ag)
    model = model.cuda()
    model.eval()
    return model


def existing_index_state(output_dir: Path, length: int, gpu_id: int) -> Tuple[int, int]:
    pattern = re.compile(rf"^{length}_g{gpu_id}_i(\d+)\.pdb$")
    indices = []
    if output_dir.exists():
        for entry in output_dir.iterdir():
            match = pattern.match(entry.name)
            if match:
                indices.append(int(match.group(1)))
    if not indices:
        return 0, 0
    return len(indices), max(indices) + 1


def next_output_path(output_dir: Path, length: int, gpu_id: int, next_index: int, resume: bool) -> Tuple[Path, int]:
    while True:
        path = output_dir / f"{length}_g{gpu_id}_i{next_index:06d}.pdb"
        if not path.exists():
            return path, next_index
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        next_index += 1


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def batch_seed(base_seed: int, gpu_id: int, length: int, batch_ordinal: int) -> int:
    return int(base_seed + gpu_id * 1_000_000 + length * 1_000 + batch_ordinal)


def run_worker(worker_cfg_path: Path) -> None:
    cfg = OmegaConf.load(worker_cfg_path)
    gpu_id = int(cfg.worker.gpu_id)
    worker_id = int(cfg.worker.worker_id)
    output_dir = Path(cfg.worker.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "_worker_config.yaml"
    OmegaConf.save(cfg, metadata_path)

    base_seed = int(cfg.seed)
    L.seed_everything(base_seed)
    model = load_model(cfg)
    dt_tensor = torch.tensor(float(cfg.dt), device=model.device)
    max_batch = int(cfg.max_nsamples)
    resume = bool(cfg.launcher.resume)

    print(
        f"worker_id={worker_id} gpu_id={gpu_id} output_dir={output_dir} "
        f"num_lengths={len(cfg.worker.length_tasks)} max_batch={max_batch}",
        flush=True,
    )

    per_length_state = []
    for task in cfg.worker.length_tasks:
        length = int(task.length)
        target_count = int(task.count)
        existing_count, next_index = existing_index_state(output_dir, length, gpu_id)
        if existing_count >= target_count:
            print(
                f"length={length}: already complete ({existing_count}/{target_count}), skipping",
                flush=True,
            )
            continue

        completed_batches = int(math.ceil(existing_count / max_batch))
        per_length_state.append(
            {
                "length": length,
                "target_count": target_count,
                "produced": existing_count,
                "next_index": next_index,
                "batch_ordinal": completed_batches,
            }
        )
        print(
            f"length={length}: existing={existing_count} target={target_count} "
            f"remaining={target_count - existing_count} completed_batches={completed_batches}",
            flush=True,
        )

    while True:
        active = False
        for state in per_length_state:
            remaining = int(state["target_count"]) - int(state["produced"])
            if remaining <= 0:
                continue
            active = True
            length = int(state["length"])
            batch_size = min(max_batch, remaining)
            seed = batch_seed(
                base_seed=base_seed,
                gpu_id=gpu_id,
                length=length,
                batch_ordinal=int(state["batch_ordinal"]),
            )
            set_generation_seed(seed)
            batch = {
                "nsamples": batch_size,
                "nres": length,
                "dt": dt_tensor,
            }

            with torch.inference_mode():
                atom37 = model.predict_step(batch, batch_idx=0)
                if isinstance(atom37, tuple):
                    atom37 = atom37[0]
                atom37 = atom37.detach().cpu().numpy()

            for sample_idx in range(atom37.shape[0]):
                output_path, next_index = next_output_path(
                    output_dir=output_dir,
                    length=length,
                    gpu_id=gpu_id,
                    next_index=int(state["next_index"]),
                    resume=resume,
                )
                write_prot_to_pdb(
                    atom37[sample_idx],
                    str(output_path),
                    overwrite=True,
                    no_indexing=True,
                )
                state["next_index"] = next_index + 1

            state["produced"] = int(state["produced"]) + int(atom37.shape[0])
            state["batch_ordinal"] = int(state["batch_ordinal"]) + 1
            print(
                f"length={length}: produced={state['produced']}/{state['target_count']} "
                f"seed={seed}",
                flush=True,
            )

        if not active:
            break
        torch.cuda.empty_cache()

    print(f"worker_id={worker_id} gpu_id={gpu_id} finished", flush=True)


def launch_workers(args: argparse.Namespace) -> None:
    base_cfg = apply_launcher_overrides(load_base_config(args.config_name), args)
    lengths = resolve_lengths(base_cfg)
    gpu_ids = [int(v) for v in base_cfg.launcher.gpu_ids]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    samples_per_length_per_gpu = base_cfg.launcher.get("samples_per_length_per_gpu")
    if samples_per_length_per_gpu is not None:
        tasks = fixed_tasks_per_gpu(
            lengths=lengths,
            samples_per_length_per_gpu=int(samples_per_length_per_gpu),
            gpu_ids=gpu_ids,
        )
        total_samples = sum(int(task["count"]) for task in tasks)
    else:
        total_samples = int(base_cfg.launcher.total_samples)
        tasks = balanced_length_tasks(lengths, total_samples)
    per_gpu_tasks = assign_tasks_per_length(tasks, gpu_ids)

    generated_config_dir = Path(base_cfg.launcher.generated_config_dir)
    log_dir = Path(base_cfg.launcher.log_dir)
    output_root = Path(base_cfg.launcher.output_root)
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_prefix = f"{base_cfg.launcher.tmux_session_prefix}_{timestamp}"
    launch_manifest = []

    for worker_id, (gpu_id, worker_tasks) in enumerate(per_gpu_tasks):
        worker_cfg = OmegaConf.create(
            OmegaConf.to_container(base_cfg, resolve=False)
        )
        worker_cfg.worker.worker_id = worker_id
        worker_cfg.worker.gpu_id = gpu_id
        worker_cfg.worker.output_dir = str(output_root / f"dataset_{gpu_id}")
        worker_cfg.worker.length_tasks = worker_tasks
        worker_cfg.seed = int(base_cfg.seed) + gpu_id

        worker_cfg_path = generated_config_dir / f"{session_prefix}_gpu{gpu_id}.yaml"
        log_path = log_dir / f"{session_prefix}_gpu{gpu_id}.log"
        OmegaConf.save(worker_cfg, worker_cfg_path)

        session_name = f"{session_prefix}_g{gpu_id}"
        command = build_worker_command(gpu_id=gpu_id, worker_cfg_path=worker_cfg_path, log_path=log_path)

        launch_manifest.append(
            {
                "gpu_id": gpu_id,
                "worker_id": worker_id,
                "session_name": session_name,
                "worker_config": str(worker_cfg_path),
                "log_path": str(log_path),
                "num_lengths": len(worker_tasks),
                "num_samples": sum(int(task["count"]) for task in worker_tasks),
                "command": command,
            }
        )

        if not args.dry_run:
            launch_tmux_session(session_name=session_name, command=command)

    manifest_path = generated_config_dir / f"{session_prefix}_manifest.yaml"
    OmegaConf.save(OmegaConf.create({"launches": launch_manifest}), manifest_path)

    length_counts = [task["count"] for task in tasks]
    actual_total_samples = sum(length_counts)
    discarded_samples = total_samples - actual_total_samples
    print(f"Base config: {args.config_name}")
    print(f"Checkpoint: {Path(base_cfg.ckpt_path) / base_cfg.ckpt_name}")
    print(
        f"Lengths: {lengths[0]}-{lengths[-1]} step={base_cfg.step_len} "
        f"({len(lengths)} total lengths)"
    )
    print(
        f"Requested total samples: {total_samples}; actual total samples: {actual_total_samples} "
        f"(per length={min(length_counts)}, discarded={discarded_samples})"
    )
    if samples_per_length_per_gpu is not None:
        print(f"Samples per length per GPU: {int(samples_per_length_per_gpu)}")
    print(f"Batch size: {base_cfg.max_nsamples}")
    print(f"Noise scale: {base_cfg.sampling_caflow.sc_scale_noise}")
    print(f"Resume mode: {bool(base_cfg.launcher.resume)}")
    print(f"Manifest: {manifest_path}")

    for launch in launch_manifest:
        print(
            f"GPU {launch['gpu_id']}: session={launch['session_name']} "
            f"lengths={launch['num_lengths']} samples={launch['num_samples']}"
        )
        print(f"  log: {launch['log_path']}")
        print(f"  attach: tmux attach -t {launch['session_name']}")
        if args.dry_run:
            print(f"  command: {launch['command']}")

    if args.dry_run:
        print("Dry run only. No tmux sessions were launched.")


def main() -> None:
    ensure_repo_env()
    args = parse_args()
    if args.worker_config is not None:
        run_worker(Path(args.worker_config).expanduser().resolve())
    else:
        launch_workers(args)


if __name__ == "__main__":
    main()
