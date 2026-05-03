#!/usr/bin/env python

import argparse
import csv
import os
import re
import shlex
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Set

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

import fcntl
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN, _S_to_seq
from proteinfoundation.utils.align_utils.align_utils import kabsch_align_ind
from proteinfoundation.utils.ff_utils.pdb_utils import from_pdb_string


DEFAULT_CONFIG_PATH = root / "configs" / "experiment_config" / "designability_eval_base.yaml"
PDB_NAME_RE = re.compile(r"^(?P<length>\d+)_g(?P<gpu>\d+)_i(?P<index>\d+)\.pdb$")


def ensure_repo_env() -> None:
    load_dotenv(root / ".env")
    data_path = os.environ.get("DATA_PATH")
    if data_path and not os.path.isabs(data_path):
        os.environ["DATA_PATH"] = str((root / data_path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch or run batched designability evaluation workers."
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU ids to use for tmux workers.",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Root directory containing raw dataset_<gpu> folders.",
    )
    parser.add_argument(
        "--designable-root",
        default=None,
        help="Root directory for designable dataset_<gpu> outputs.",
    )
    parser.add_argument(
        "--undesignable-root",
        default=None,
        help="Root directory for undesignable dataset_<gpu> outputs.",
    )
    parser.add_argument(
        "--state-root",
        default=None,
        help="Root directory for worker state files.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Number of raw backbones evaluated together per same-length batch.",
    )
    parser.add_argument(
        "--max-batches-per-length-per-cycle",
        type=int,
        default=None,
        help="Max same-length batches to process before moving to the next length.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Designability threshold on scRMSD.",
    )
    parser.add_argument(
        "--num-seq-per-target",
        type=int,
        default=None,
        help="Number of ProteinMPNN sequences per target.",
    )
    parser.add_argument(
        "--sampling-temp",
        default=None,
        help="ProteinMPNN sampling temperature string.",
    )
    parser.add_argument(
        "--pmpnn-batch-size",
        type=int,
        default=None,
        help="Internal ProteinMPNN sampling batch size.",
    )
    parser.add_argument(
        "--esmfold-model-name",
        default=None,
        help="HF model id or local path for ESMFold.",
    )
    parser.add_argument(
        "--esmfold-cache-dir",
        default=None,
        help="Optional cache dir passed to transformers.",
    )
    parser.add_argument(
        "--pmpnn-weights-dir",
        default=None,
        help="Optional override for ProteinMPNN weight directory.",
    )
    parser.add_argument(
        "--min-file-age-sec",
        type=int,
        default=None,
        help="Ignore files newer than this to avoid racing generation writes.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=None,
        help="Polling interval when watch mode is enabled.",
    )
    parser.add_argument(
        "--idle-exit-after-sec",
        type=int,
        default=None,
        help="Exit after this many idle seconds with no new eligible files. Use 0 to never auto-exit.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan pass and exit instead of watching.",
    )
    parser.add_argument(
        "--session-prefix",
        default=None,
        help="Tmux session prefix.",
    )
    parser.add_argument(
        "--partition-mod",
        type=int,
        default=None,
        help="Only process files where parsed sample_index %% partition_mod == partition_rem.",
    )
    parser.add_argument(
        "--partition-rem",
        type=int,
        default=None,
        help="Partition remainder used together with --partition-mod.",
    )
    parser.add_argument(
        "--state-suffix",
        default=None,
        help="Suffix added to state/error CSV names, e.g. part0 or part1.",
    )
    parser.add_argument(
        "--max-length-prefix",
        type=int,
        default=None,
        help="Only process PDB files whose leading length prefix is <= this value.",
    )
    parser.add_argument(
        "--min-length-prefix",
        type=int,
        default=None,
        help="Only process PDB files whose leading length prefix is >= this value.",
    )
    parser.add_argument(
        "--target-designable-per-length",
        type=int,
        default=None,
        help="Stop evaluating a dataset once every length in [min_length_prefix, max_length_prefix] has at least this many designable proteins.",
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
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without launching tmux sessions.",
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

    if args.gpus is not None:
        cfg.launcher.gpu_ids = list(args.gpus)
    if args.raw_root is not None:
        cfg.launcher.raw_root = args.raw_root
    if args.designable_root is not None:
        cfg.launcher.designable_root = args.designable_root
    if args.undesignable_root is not None:
        cfg.launcher.undesignable_root = args.undesignable_root
    if args.state_root is not None:
        cfg.launcher.state_root = args.state_root
    if args.eval_batch_size is not None:
        cfg.designability.eval_batch_size = args.eval_batch_size
    if args.max_batches_per_length_per_cycle is not None:
        cfg.designability.max_batches_per_length_per_cycle = args.max_batches_per_length_per_cycle
    if args.score_threshold is not None:
        cfg.designability.score_threshold = args.score_threshold
    if args.num_seq_per_target is not None:
        cfg.designability.num_seq_per_target = args.num_seq_per_target
    if args.sampling_temp is not None:
        cfg.designability.sampling_temp = args.sampling_temp
    if args.pmpnn_batch_size is not None:
        cfg.designability.pmpnn_batch_size = args.pmpnn_batch_size
    if args.esmfold_model_name is not None:
        cfg.designability.esmfold_model_name = args.esmfold_model_name
    if args.esmfold_cache_dir is not None:
        cfg.designability.esmfold_cache_dir = args.esmfold_cache_dir
    if args.pmpnn_weights_dir is not None:
        cfg.designability.path_to_model_weights = args.pmpnn_weights_dir
    if args.min_file_age_sec is not None:
        cfg.launcher.min_file_age_sec = args.min_file_age_sec
    if args.poll_interval_sec is not None:
        cfg.launcher.poll_interval_sec = args.poll_interval_sec
    if args.idle_exit_after_sec is not None:
        cfg.launcher.idle_exit_after_sec = args.idle_exit_after_sec
    if args.once:
        cfg.launcher.watch = False
    if args.session_prefix is not None:
        cfg.launcher.tmux_session_prefix = args.session_prefix
    if args.partition_mod is not None:
        cfg.launcher.partition_mod = args.partition_mod
    if args.partition_rem is not None:
        cfg.launcher.partition_rem = args.partition_rem
    if args.state_suffix is not None:
        cfg.launcher.state_suffix = args.state_suffix
    if args.max_length_prefix is not None:
        cfg.launcher.max_length_prefix = args.max_length_prefix
    if args.min_length_prefix is not None:
        cfg.launcher.min_length_prefix = args.min_length_prefix
    if args.target_designable_per_length is not None:
        cfg.launcher.target_designable_per_length = args.target_designable_per_length
    if args.generated_config_dir is not None:
        cfg.launcher.generated_config_dir = args.generated_config_dir
    if args.log_dir is not None:
        cfg.launcher.log_dir = args.log_dir

    return cfg


class FastDesignabilityEvaluator:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.args = SimpleNamespace(
            suppress_print=int(cfg.suppress_print),
            ca_only=bool(cfg.ca_only),
            path_to_model_weights=str(cfg.path_to_model_weights or ""),
            model_name=str(cfg.model_name),
            use_soluble_model=bool(cfg.use_soluble_model),
            seed=int(cfg.seed),
            backbone_noise=float(cfg.backbone_noise),
            num_seq_per_target=int(cfg.num_seq_per_target),
            batch_size=int(cfg.pmpnn_batch_size),
            sampling_temp=str(cfg.sampling_temp),
            pssm_multi=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            omit_AAs="X",
        )

        model_folder_path = self.args.path_to_model_weights
        if model_folder_path:
            model_folder_path = os.path.join(model_folder_path, "")
        else:
            if self.args.ca_only:
                model_folder_path = str(root / "ProteinMPNN" / "ca_model_weights") + os.sep
            elif self.args.use_soluble_model:
                model_folder_path = str(root / "soluble_model_weights") + os.sep
            else:
                model_folder_path = str(root / "vanilla_model_weights") + os.sep

        checkpoint_path = os.path.join(model_folder_path, f"{self.args.model_name}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"ProteinMPNN weights not found: {checkpoint_path}. "
                "If needed, run script_utils/download_pmpnn_weghts.sh or pass --pmpnn-weights-dir."
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ProteinMPNN(
            ca_only=self.args.ca_only,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=self.args.backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.mpnn_model = model

        esmfold_kwargs = {}
        if cfg.esmfold_cache_dir:
            esmfold_kwargs["cache_dir"] = cfg.esmfold_cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.esmfold_model_name, **esmfold_kwargs)
        self.esm_model = EsmForProteinFolding.from_pretrained(
            cfg.esmfold_model_name, **esmfold_kwargs
        ).to(device)
        self.esm_model.eval()

    def protein_mpnn_sequences(self, proteins: torch.Tensor) -> List[str]:
        args = self.args
        if args.seed:
            seed = args.seed
        else:
            seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        np.random.seed(seed)

        batch_size = min(args.batch_size, proteins.shape[0])
        if batch_size <= 0:
            return []

        num_batches = proteins.shape[0] // batch_size
        if proteins.shape[0] % batch_size != 0:
            num_batches += 1

        temperatures = [float(item) for item in args.sampling_temp.split()]
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        omit_aas_np = np.array([aa in args.omit_AAs for aa in alphabet]).astype(np.float32)
        bias_aas_np = np.zeros(len(alphabet))

        all_seqs: List[str] = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, proteins.shape[0])
                x = proteins[start:end]
                bsz, nres = x.shape[:2]

                s = torch.zeros((bsz, nres), device=x.device)
                mask = torch.ones((bsz, nres), device=x.device)
                chain_m = torch.ones((bsz, nres), device=x.device)
                chain_encoding_all = torch.zeros((bsz, nres), device=x.device)
                residue_idx = torch.arange(nres, device=x.device).unsqueeze(0).expand(bsz, nres)
                chain_m_pos = torch.ones((bsz, nres), device=x.device)
                omit_aa_mask = torch.zeros((bsz, nres, 21), device=x.device)
                pssm_coef = torch.zeros((bsz, nres), device=x.device)
                pssm_bias = torch.zeros((bsz, nres, 21), device=x.device)
                pssm_log_odds_mask = torch.ones((bsz, nres, 21), device=x.device)
                bias_by_res_all = torch.zeros((bsz, nres, 21), device=x.device)

                for temp in temperatures:
                    randn_2 = torch.randn(chain_m.shape, device=x.device)
                    sample_dict = self.mpnn_model.sample(
                        x,
                        randn_2,
                        s,
                        chain_m,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_aas_np,
                        bias_AAs_np=bias_aas_np,
                        chain_M_pos=chain_m_pos,
                        omit_AA_mask=omit_aa_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=args.pssm_multi,
                        pssm_log_odds_flag=bool(args.pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(args.pssm_bias_flag),
                        bias_by_res=bias_by_res_all,
                    )
                    for sample_idx in range(bsz):
                        all_seqs.append(_S_to_seq(sample_dict["S"][sample_idx], chain_m[sample_idx]))

        return all_seqs

    def sc_rmsd(self, proteins: torch.Tensor) -> torch.Tensor:
        proteins = proteins.to(self.device, dtype=torch.float32)
        ns = self.args.num_seq_per_target
        proteins_copied = proteins.repeat_interleave(ns, dim=0)
        seqs = self.protein_mpnn_sequences(proteins_copied)
        esm_batch_size = min(20, len(seqs))
        rmsd_values = []

        for start in range(0, len(seqs), esm_batch_size):
            batch_seqs = seqs[start : start + esm_batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.esm_model(**inputs)
                atom37_outputs = atom14_to_atom37(outputs["positions"][-1], outputs)
                pred_positions = atom37_outputs[:, :, 1, :]

            for local_idx in range(pred_positions.shape[0]):
                source_idx = (start + local_idx) // ns
                aligned_1, aligned_2 = kabsch_align_ind(
                    pred_positions[local_idx], proteins[source_idx], ret_both=True
                )
                sq_err = (aligned_1 - aligned_2) ** 2
                rmsd_values.append(sq_err.sum(dim=-1).mean().sqrt())

        scores = torch.stack(rmsd_values).view(-1, ns).min(dim=-1).values
        return scores.detach().cpu()


def parse_length_from_name(name: str) -> int:
    match = PDB_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Unexpected pdb filename format: {name}")
    return int(match.group("length"))


def parse_sample_index_from_name(name: str) -> int:
    match = PDB_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Unexpected pdb filename format: {name}")
    return int(match.group("index"))


def read_processed_names(*dirs: Path) -> Set[str]:
    processed: Set[str] = set()
    for directory in dirs:
        if not directory.exists():
            continue
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.endswith(".pdb"):
                processed.add(entry.name)
    return processed


def read_designable_length_counts(
    designable_dir: Path, min_length: int, max_length: int
) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    if not designable_dir.exists():
        return counts
    for entry in designable_dir.iterdir():
        if not entry.is_file() or not entry.name.endswith(".pdb"):
            continue
        try:
            length = parse_length_from_name(entry.name)
        except Exception:
            continue
        if min_length <= length <= max_length:
            counts[length] += 1
    return counts


def lengths_still_needed(
    designable_dir: Path, min_length: int, max_length: int, target_count: int
) -> Set[int]:
    counts = read_designable_length_counts(designable_dir, min_length, max_length)
    return {
        length
        for length in range(min_length, max_length + 1)
        if counts.get(length, 0) < target_count
    }


def read_state_names(state_file: Path) -> Set[str]:
    processed: Set[str] = set()
    if not state_file.exists():
        return processed
    with open(state_file, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("name"):
                processed.add(row["name"])
    return processed


def append_csv_row(path: Path, header: Sequence[str], row: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_ca_coords(pdb_path: Path) -> np.ndarray:
    with open(pdb_path, "r") as handle:
        prot = from_pdb_string(handle.read())
    atom_mask = prot.atom_mask
    if atom_mask.shape[1] <= 1 or np.any(atom_mask[:, 1] < 0.5):
        raise ValueError(f"Missing CA atoms in {pdb_path}")
    return prot.atom_positions[:, 1, :].astype(np.float32)


def chunked(items: Sequence[Path], chunk_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def build_worker_command(worker_cfg_path: Path, log_path: Path, gpu_id: int) -> str:
    quoted_root = shlex.quote(str(root))
    quoted_cfg = shlex.quote(str(worker_cfg_path))
    quoted_log = shlex.quote(str(log_path))
    return (
        'source "$(conda info --base)/etc/profile.d/conda.sh"'
        f" && conda activate proteina_env"
        f" && cd {quoted_root}"
        f" && export CUDA_VISIBLE_DEVICES={gpu_id}"
        f" && python script_utils/evaluate_designability_dataset.py --worker-config {quoted_cfg}"
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


def discover_partitioned_candidates(
    raw_dir: Path,
    processed: Set[str],
    min_age_sec: int,
    partition_mod: int,
    partition_rem: int,
    min_length_prefix: int,
    max_length_prefix: int,
    needed_lengths: Set[int],
) -> Dict[int, List[Path]]:
    now = time.time()
    grouped: Dict[int, List[Path]] = defaultdict(list)
    if not raw_dir.exists():
        return grouped

    with os.scandir(raw_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".pdb"):
                continue
            if entry.name in processed:
                continue
            match = PDB_NAME_RE.match(entry.name)
            if match is None:
                continue
            length = int(match.group("length"))
            if length < min_length_prefix:
                continue
            if length > max_length_prefix:
                continue
            if needed_lengths and length not in needed_lengths:
                continue
            if int(match.group("index")) % partition_mod != partition_rem:
                continue
            if now - entry.stat().st_mtime < min_age_sec:
                continue
            grouped[length].append(Path(entry.path))

    for length in grouped:
        grouped[length].sort(key=lambda path: path.name)
    return grouped


def process_batch(
    batch_paths: Sequence[Path],
    evaluator: FastDesignabilityEvaluator,
    designable_dir: Path,
    undesignable_dir: Path,
    state_file: Path,
    score_threshold: float,
    processed: Set[str],
    failure_counts: Dict[str, int],
    error_file: Path,
) -> int:
    coords = [load_ca_coords(path) for path in batch_paths]
    proteins = torch.as_tensor(np.stack(coords, axis=0), device=evaluator.device, dtype=torch.float32)
    scores = evaluator.sc_rmsd(proteins).tolist()

    processed_now = 0
    for path, score in zip(batch_paths, scores):
        is_designable = float(score) < score_threshold
        target_dir = designable_dir if is_designable else undesignable_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / path.name
        if not dest_path.exists():
            shutil.copy2(path, dest_path)

        append_csv_row(
            state_file,
            header=[
                "name",
                "source_path",
                "length",
                "score",
                "is_designable",
                "dest_path",
                "processed_at_utc",
            ],
            row=[
                path.name,
                str(path),
                parse_length_from_name(path.name),
                f"{float(score):.6f}",
                int(is_designable),
                str(dest_path),
                datetime.utcnow().isoformat(timespec="seconds"),
            ],
        )
        processed.add(path.name)
        failure_counts.pop(path.name, None)
        processed_now += 1

    return processed_now


def process_grouped_candidates(
    cfg,
    evaluator: FastDesignabilityEvaluator,
    processed: Set[str],
    failure_counts: Dict[str, int],
    needed_lengths: Set[int],
) -> int:
    raw_dir = Path(cfg.worker.raw_dir)
    designable_dir = Path(cfg.worker.designable_dir)
    undesignable_dir = Path(cfg.worker.undesignable_dir)
    state_file = Path(cfg.worker.state_file)
    error_file = Path(cfg.worker.error_file)
    design_cfg = cfg.designability
    grouped = discover_partitioned_candidates(
        raw_dir=raw_dir,
        processed=processed,
        min_age_sec=int(cfg.launcher.min_file_age_sec),
        partition_mod=int(cfg.worker.partition_mod),
        partition_rem=int(cfg.worker.partition_rem),
        min_length_prefix=int(cfg.launcher.min_length_prefix),
        max_length_prefix=int(cfg.launcher.max_length_prefix),
        needed_lengths=needed_lengths,
    )
    if not grouped:
        return 0

    processed_count = 0
    for length in sorted(grouped):
        batches_done = 0
        for batch_paths in chunked(grouped[length], int(design_cfg.eval_batch_size)):
            if batches_done >= int(design_cfg.max_batches_per_length_per_cycle):
                break
            try:
                processed_count += process_batch(
                    batch_paths=batch_paths,
                    evaluator=evaluator,
                    designable_dir=designable_dir,
                    undesignable_dir=undesignable_dir,
                    state_file=state_file,
                    score_threshold=float(design_cfg.score_threshold),
                    processed=processed,
                    failure_counts=failure_counts,
                    error_file=error_file,
                )
            except Exception as exc:
                if len(batch_paths) > 1:
                    for single_path in batch_paths:
                        try:
                            processed_count += process_batch(
                                batch_paths=[single_path],
                                evaluator=evaluator,
                                designable_dir=designable_dir,
                                undesignable_dir=undesignable_dir,
                                state_file=state_file,
                                score_threshold=float(design_cfg.score_threshold),
                                processed=processed,
                                failure_counts=failure_counts,
                                error_file=error_file,
                            )
                        except Exception as single_exc:
                            failure_counts[single_path.name] = failure_counts.get(single_path.name, 0) + 1
                            append_csv_row(
                                error_file,
                                header=["name", "source_path", "error", "attempt", "timestamp_utc"],
                                row=[
                                    single_path.name,
                                    str(single_path),
                                    repr(single_exc),
                                    failure_counts[single_path.name],
                                    datetime.utcnow().isoformat(timespec="seconds"),
                                ],
                            )
                else:
                    single_path = batch_paths[0]
                    failure_counts[single_path.name] = failure_counts.get(single_path.name, 0) + 1
                    append_csv_row(
                        error_file,
                        header=["name", "source_path", "error", "attempt", "timestamp_utc"],
                        row=[
                            single_path.name,
                            str(single_path),
                            repr(exc),
                            failure_counts[single_path.name],
                            datetime.utcnow().isoformat(timespec="seconds"),
                        ],
                    )
            batches_done += 1

    return processed_count


def run_worker(worker_cfg_path: Path) -> None:
    ensure_repo_env()
    cfg = OmegaConf.load(worker_cfg_path)
    gpu_id = int(cfg.worker.gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dir = Path(cfg.worker.raw_dir)
    designable_dir = Path(cfg.worker.designable_dir)
    undesignable_dir = Path(cfg.worker.undesignable_dir)
    state_file = Path(cfg.worker.state_file)
    error_file = Path(cfg.worker.error_file)

    designable_dir.mkdir(parents=True, exist_ok=True)
    undesignable_dir.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    processed = set()
    if bool(cfg.launcher.resume):
        processed |= read_processed_names(designable_dir, undesignable_dir)
        processed |= read_state_names(state_file)

    print(
        f"gpu_id={gpu_id} raw_dir={raw_dir} designable_dir={designable_dir} "
        f"undesignable_dir={undesignable_dir} resume={bool(cfg.launcher.resume)} "
        f"partition={cfg.worker.partition_rem}/{cfg.worker.partition_mod} "
        f"state_suffix={cfg.worker.state_suffix} max_length_prefix={cfg.launcher.max_length_prefix} "
        f"processed_at_start={len(processed)}",
        flush=True,
    )

    evaluator = FastDesignabilityEvaluator(cfg.designability, device=device)
    idle_started = None
    failure_counts: Dict[str, int] = {}

    while True:
        needed_lengths = lengths_still_needed(
            designable_dir=designable_dir,
            min_length=int(cfg.launcher.min_length_prefix),
            max_length=int(cfg.launcher.max_length_prefix),
            target_count=int(cfg.launcher.target_designable_per_length),
        )
        if not needed_lengths:
            print(
                f"gpu_id={gpu_id} target met: every length in "
                f"[{cfg.launcher.min_length_prefix}, {cfg.launcher.max_length_prefix}] "
                f"has at least {cfg.launcher.target_designable_per_length} designables. Exiting.",
                flush=True,
            )
            break

        processed_now = process_grouped_candidates(
            cfg, evaluator, processed, failure_counts, needed_lengths
        )
        if processed_now > 0:
            idle_started = None
            print(
                f"{datetime.utcnow().isoformat(timespec='seconds')} gpu_id={gpu_id} "
                f"processed_now={processed_now} processed_total={len(processed)} "
                f"needed_lengths={sorted(needed_lengths)}",
                flush=True,
            )
            continue

        if not bool(cfg.launcher.watch):
            print(f"gpu_id={gpu_id} no more eligible files, exiting single-pass mode", flush=True)
            break

        now = time.time()
        if idle_started is None:
            idle_started = now
        idle_exit_after = int(cfg.launcher.idle_exit_after_sec)
        if idle_exit_after > 0 and now - idle_started >= idle_exit_after:
            print(
                f"gpu_id={gpu_id} idle for {idle_exit_after} seconds with no new eligible files, exiting",
                flush=True,
            )
            break

        print(
            f"gpu_id={gpu_id} idle, sleeping {int(cfg.launcher.poll_interval_sec)}s "
            f"(processed_total={len(processed)} needed_lengths={sorted(needed_lengths)})",
            flush=True,
        )
        time.sleep(int(cfg.launcher.poll_interval_sec))


def launch_workers(args: argparse.Namespace) -> None:
    ensure_repo_env()
    cfg = apply_overrides(load_base_config(args.config_path), args)

    gpu_ids = [int(v) for v in cfg.launcher.gpu_ids]
    generated_config_dir = Path(cfg.launcher.generated_config_dir)
    log_dir = Path(cfg.launcher.log_dir)
    state_root = Path(cfg.launcher.state_root)
    raw_root = Path(cfg.launcher.raw_root)
    designable_root = Path(cfg.launcher.designable_root)
    undesignable_root = Path(cfg.launcher.undesignable_root)

    generated_config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    designable_root.mkdir(parents=True, exist_ok=True)
    undesignable_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_prefix = f"{cfg.launcher.tmux_session_prefix}_{timestamp}"
    manifest = {"launches": []}
    state_suffix = str(cfg.launcher.state_suffix)

    for worker_id, gpu_id in enumerate(gpu_ids):
        worker_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        worker_cfg.worker.worker_id = worker_id
        worker_cfg.worker.gpu_id = gpu_id
        worker_cfg.worker.raw_dir = str(raw_root / f"dataset_{gpu_id}")
        worker_cfg.worker.designable_dir = str(designable_root / f"dataset_{gpu_id}")
        worker_cfg.worker.undesignable_dir = str(undesignable_root / f"dataset_{gpu_id}")
        worker_cfg.worker.partition_mod = int(cfg.launcher.partition_mod)
        worker_cfg.worker.partition_rem = int(cfg.launcher.partition_rem)
        worker_cfg.worker.state_suffix = state_suffix
        worker_cfg.worker.state_file = str(state_root / f"dataset_{gpu_id}.{state_suffix}.csv")
        worker_cfg.worker.error_file = str(state_root / f"dataset_{gpu_id}.{state_suffix}_errors.csv")

        worker_cfg_path = generated_config_dir / f"{session_prefix}_gpu{gpu_id}.yaml"
        log_path = log_dir / f"{session_prefix}_gpu{gpu_id}.log"
        OmegaConf.save(worker_cfg, worker_cfg_path)

        session_name = f"{session_prefix}_g{gpu_id}"
        command = build_worker_command(worker_cfg_path, log_path, gpu_id)

        manifest["launches"].append(
            {
                "gpu_id": gpu_id,
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
    print(f"Raw root: {raw_root}")
    print(f"Designable root: {designable_root}")
    print(f"Undesignable root: {undesignable_root}")
    print(f"State root: {state_root}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Eval batch size: {cfg.designability.eval_batch_size}, "
        f"max batches/length/cycle: {cfg.designability.max_batches_per_length_per_cycle}, "
        f"threshold: {cfg.designability.score_threshold}"
    )
    print(
        f"Watch mode: {bool(cfg.launcher.watch)}, poll interval: {cfg.launcher.poll_interval_sec}s, "
        f"idle exit: {cfg.launcher.idle_exit_after_sec}s"
    )
    print(
        f"Partition: rem={cfg.launcher.partition_rem}, mod={cfg.launcher.partition_mod}, "
        f"state_suffix={cfg.launcher.state_suffix}, min_length_prefix={cfg.launcher.min_length_prefix}, "
        f"max_length_prefix={cfg.launcher.max_length_prefix}, "
        f"target_designable_per_length={cfg.launcher.target_designable_per_length}"
    )
    for launch in manifest["launches"]:
        print(f"GPU {launch['gpu_id']}: session={launch['session_name']}")
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
