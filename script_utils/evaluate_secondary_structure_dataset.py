#!/usr/bin/env python

import argparse
import csv
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

import fcntl
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

import biotite.structure.io.pdb as pdb
import biotite.structure.sse as annotate


DEFAULT_CONFIG_PATH = root / "configs" / "experiment_config" / "secondary_structure_eval_base.yaml"


def ensure_repo_env() -> None:
    load_dotenv(root / ".env")
    data_path = os.environ.get("DATA_PATH")
    if data_path and not os.path.isabs(data_path):
        os.environ["DATA_PATH"] = str((root / data_path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch or run secondary-structure evaluation watchers."
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root containing dataset_<i> designable folders.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="Dataset ids to launch. Kept as 'gpus' for consistency with the other launchers.",
    )
    parser.add_argument(
        "--generated-config-dir",
        default=None,
        help="Directory where per-worker configs are written.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory where per-worker logs are written.",
    )
    parser.add_argument(
        "--session-prefix",
        default=None,
        help="Tmux session prefix.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=None,
        help="Polling interval while watching.",
    )
    parser.add_argument(
        "--idle-exit-after-sec",
        type=int,
        default=None,
        help="Exit after this many idle seconds with no new designable proteins. Use 0 to never auto-exit.",
    )
    parser.add_argument(
        "--min-file-age-sec",
        type=int,
        default=None,
        help="Ignore very new PDBs to avoid racing ongoing copies.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=None,
        help="Append results to CSV every N proteins.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan pass and exit instead of watching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without launching tmux.",
    )
    parser.add_argument(
        "--worker-config",
        default=None,
        help="Internal worker mode: run one worker from a generated config.",
    )
    return parser.parse_args()


def load_base_config(config_path: str):
    return OmegaConf.load(config_path)


def apply_overrides(cfg, args: argparse.Namespace):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if args.dataset_root is not None:
        cfg.launcher.dataset_root = args.dataset_root
    if args.gpus is not None:
        cfg.launcher.gpu_ids = list(args.gpus)
    if args.generated_config_dir is not None:
        cfg.launcher.generated_config_dir = args.generated_config_dir
    if args.log_dir is not None:
        cfg.launcher.log_dir = args.log_dir
    if args.session_prefix is not None:
        cfg.launcher.tmux_session_prefix = args.session_prefix
    if args.poll_interval_sec is not None:
        cfg.launcher.poll_interval_sec = args.poll_interval_sec
    if args.idle_exit_after_sec is not None:
        cfg.launcher.idle_exit_after_sec = args.idle_exit_after_sec
    if args.min_file_age_sec is not None:
        cfg.launcher.min_file_age_sec = args.min_file_age_sec
    if args.flush_every is not None:
        cfg.secondary_structure.flush_every = args.flush_every
    if args.once:
        cfg.launcher.watch = False
    return cfg


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


def is_complete_pdb(path: Path) -> bool:
    try:
        with open(path, "r") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        return bool(lines) and lines[-1] == "END"
    except Exception:
        return False


def read_existing_names(csv_path: Path) -> Set[str]:
    names: Set[str] = set()
    if not csv_path.exists():
        return names
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("name")
            if name:
                names.add(name)
    return names


def append_rows(csv_path: Path, rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(
                    [
                        "name",
                        "source_path",
                        "length",
                        "alpha_fraction",
                        "beta_fraction",
                        "coil_fraction",
                        "processed_at_utc",
                    ]
                )
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_length_from_name(name: str) -> int:
    try:
        return int(name.split("_", 1)[0])
    except Exception as exc:
        raise ValueError(f"Could not parse length from {name}") from exc


def discover_pending(dataset_dir: Path, done_names: Set[str], min_file_age_sec: int) -> List[Path]:
    now = time.time()
    pending: List[Path] = []
    if not dataset_dir.exists():
        return pending

    with os.scandir(dataset_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".pdb"):
                continue
            if entry.name in done_names:
                continue
            if now - entry.stat().st_mtime < min_file_age_sec:
                continue
            path = Path(entry.path)
            if not is_complete_pdb(path):
                continue
            pending.append(path)

    pending.sort(key=lambda p: p.name)
    return pending


def build_worker_command(worker_cfg_path: Path, log_path: Path) -> str:
    quoted_root = shlex.quote(str(root))
    quoted_cfg = shlex.quote(str(worker_cfg_path))
    quoted_log = shlex.quote(str(log_path))
    return (
        'source "$(conda info --base)/etc/profile.d/conda.sh"'
        f" && conda activate proteina_env"
        f" && cd {quoted_root}"
        f" && python script_utils/evaluate_secondary_structure_dataset.py --worker-config {quoted_cfg}"
        f" > {quoted_log} 2>&1"
        " ; status=$?"
        ' ; if [ "$status" -ne 0 ]; then'
        ' echo "Worker failed with exit code $status. Showing log tail:"'
        f" ; tail -n 80 {quoted_log}"
        " ; exec bash"
        " ; fi"
    )


def launch_tmux_session(session_name: str, command: str) -> None:
    import subprocess

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, f"bash -lc {shlex.quote(command)}"],
        check=True,
    )


def flush_buffer(csv_path: Path, buffer_rows: List[Sequence[object]]) -> int:
    if not buffer_rows:
        return 0
    append_rows(csv_path, buffer_rows)
    flushed = len(buffer_rows)
    buffer_rows.clear()
    return flushed


def run_worker(worker_cfg_path: Path) -> None:
    ensure_repo_env()
    cfg = OmegaConf.load(worker_cfg_path)
    dataset_id = int(cfg.worker.dataset_id)
    dataset_dir = Path(cfg.worker.dataset_dir)
    csv_path = Path(cfg.worker.csv_path)

    done_names = read_existing_names(csv_path)
    buffer_rows: List[Sequence[object]] = []
    flush_every = int(cfg.secondary_structure.flush_every)

    print(
        f"dataset_id={dataset_id} dataset_dir={dataset_dir} csv_path={csv_path} "
        f"done_at_start={len(done_names)}",
        flush=True,
    )

    idle_started = None

    while True:
        pending = discover_pending(
            dataset_dir=dataset_dir,
            done_names=done_names,
            min_file_age_sec=int(cfg.launcher.min_file_age_sec),
        )

        if pending:
            idle_started = None
            processed_now = 0
            for path in pending:
                ss = analyze_secondary_structure(path)
                buffer_rows.append(
                    [
                        path.name,
                        str(path),
                        parse_length_from_name(path.name),
                        f"{float(ss[0]):.8f}",
                        f"{float(ss[1]):.8f}",
                        f"{float(ss[2]):.8f}",
                        datetime.utcnow().isoformat(timespec="seconds"),
                    ]
                )
                done_names.add(path.name)
                processed_now += 1

                if len(buffer_rows) >= flush_every:
                    flushed = flush_buffer(csv_path, buffer_rows)
                    print(
                        f"{datetime.utcnow().isoformat(timespec='seconds')} dataset_id={dataset_id} "
                        f"processed_now={processed_now} flushed={flushed} total_done={len(done_names)}",
                        flush=True,
                    )

            flushed = flush_buffer(csv_path, buffer_rows)
            print(
                f"{datetime.utcnow().isoformat(timespec='seconds')} dataset_id={dataset_id} "
                f"cycle_complete processed_now={processed_now} flushed={flushed} total_done={len(done_names)}",
                flush=True,
            )
            continue

        flushed = flush_buffer(csv_path, buffer_rows)
        if flushed:
            print(
                f"{datetime.utcnow().isoformat(timespec='seconds')} dataset_id={dataset_id} "
                f"flushed_tail={flushed} total_done={len(done_names)}",
                flush=True,
            )

        if not bool(cfg.launcher.watch):
            print(f"dataset_id={dataset_id} no pending proteins, exiting single-pass mode", flush=True)
            break

        now = time.time()
        if idle_started is None:
            idle_started = now
        idle_exit_after = int(cfg.launcher.idle_exit_after_sec)
        if idle_exit_after > 0 and now - idle_started >= idle_exit_after:
            print(
                f"dataset_id={dataset_id} idle for {idle_exit_after} seconds with no new proteins, exiting",
                flush=True,
            )
            break

        print(
            f"dataset_id={dataset_id} idle, sleeping {int(cfg.launcher.poll_interval_sec)}s "
            f"(total_done={len(done_names)})",
            flush=True,
        )
        time.sleep(int(cfg.launcher.poll_interval_sec))


def launch_workers(args: argparse.Namespace) -> None:
    ensure_repo_env()
    cfg = apply_overrides(load_base_config(args.config_path), args)

    dataset_root = Path(cfg.launcher.dataset_root)
    generated_config_dir = Path(cfg.launcher.generated_config_dir)
    log_dir = Path(cfg.launcher.log_dir)
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_prefix = f"{cfg.launcher.tmux_session_prefix}_{timestamp}"
    manifest = {"launches": []}

    for worker_id, dataset_id in enumerate([int(v) for v in cfg.launcher.gpu_ids]):
        worker_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        worker_cfg.worker.worker_id = worker_id
        worker_cfg.worker.dataset_id = dataset_id
        worker_cfg.worker.dataset_dir = str(dataset_root / f"dataset_{dataset_id}")
        worker_cfg.worker.csv_path = str(dataset_root / f"dataset_{dataset_id}" / "secondary_structure.csv")

        worker_cfg_path = generated_config_dir / f"{session_prefix}_dataset_{dataset_id}.yaml"
        log_path = log_dir / f"{session_prefix}_dataset_{dataset_id}.log"
        OmegaConf.save(worker_cfg, worker_cfg_path)

        session_name = f"{session_prefix}_d{dataset_id}"
        command = build_worker_command(worker_cfg_path, log_path)
        manifest["launches"].append(
            {
                "dataset_id": dataset_id,
                "worker_id": worker_id,
                "session_name": session_name,
                "worker_config": str(worker_cfg_path),
                "log_path": str(log_path),
                "command": command,
            }
        )

        if not args.dry_run:
            launch_tmux_session(session_name, command)

    manifest_path = generated_config_dir / f"{session_prefix}_manifest.yaml"
    OmegaConf.save(OmegaConf.create(manifest), manifest_path)

    print(f"Base config: {args.config_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Flush every: {cfg.secondary_structure.flush_every}, "
        f"poll interval: {cfg.launcher.poll_interval_sec}s, "
        f"idle exit: {cfg.launcher.idle_exit_after_sec}s"
    )
    for launch in manifest["launches"]:
        print(f"dataset {launch['dataset_id']}: session={launch['session_name']}")
        print(f"  log: {launch['log_path']}")
        print(f"  attach: tmux attach -t {launch['session_name']}")
        if args.dry_run:
            print(f"  command: {launch['command']}")
    if args.dry_run:
        print("Dry run only. No tmux sessions were launched.")


def main() -> None:
    args = parse_args()
    if args.worker_config is not None:
        run_worker(Path(args.worker_config).expanduser().resolve())
    else:
        launch_workers(args)


if __name__ == "__main__":
    main()
