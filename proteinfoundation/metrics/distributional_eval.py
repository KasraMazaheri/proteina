from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def summarize_distributional_metrics(
    raw_metrics: Mapping[str, Any],
) -> "OrderedDict[str, float]":
    """
    Convert raw repo metric names into the paper-style distributional metrics.

    Expected raw keys:
    - PDB_FID, AFDB_FID
    - fS_C, fS_A, fS_T
    - PDB_fJSD_C/A/T, AFDB_fJSD_C/A/T
    """

    paper_metrics: "OrderedDict[str, float]" = OrderedDict()

    def find_first_key(*candidates: str) -> str | None:
        for key in candidates:
            if key in raw_metrics:
                return key
        return None

    pdb_fpsd_key = find_first_key("PDB_FID", "PDB_FPSD")
    afdb_fpsd_key = find_first_key("AFDB_FID", "AFDB_FPSD", "D_FS_FID", "D_FS_FPSD")

    if pdb_fpsd_key is not None:
        paper_metrics["FPSD_vs_PDB"] = _to_float(raw_metrics[pdb_fpsd_key])
    if afdb_fpsd_key is not None:
        paper_metrics["FPSD_vs_AFDB"] = _to_float(raw_metrics[afdb_fpsd_key])

    for level in ("C", "A", "T"):
        key = f"fS_{level}"
        if key in raw_metrics:
            paper_metrics[key] = _to_float(raw_metrics[key])

    ref_prefixes = {
        "PDB": ("PDB",),
        "AFDB": ("AFDB", "D_FS"),
    }

    for ref, prefixes in ref_prefixes.items():
        per_level = []
        for level in ("C", "A", "T"):
            key = find_first_key(*[f"{prefix}_fJSD_{level}" for prefix in prefixes])
            if key is not None:
                value = _to_float(raw_metrics[key])
                paper_metrics[f"fJSD_{level}_vs_{ref}"] = value
                per_level.append(value)
        if len(per_level) == 3:
            paper_metrics[f"fJSD_vs_{ref}"] = sum(per_level) / 3.0

    return paper_metrics


def format_distributional_metrics(
    paper_metrics: Mapping[str, Any],
) -> "OrderedDict[str, float]":
    """
    Re-order paper metrics for table-style reporting.
    """

    ordered = OrderedDict()
    for key in (
        "FPSD_vs_PDB",
        "FPSD_vs_AFDB",
        "fS_C",
        "fS_A",
        "fS_T",
        "fJSD_vs_PDB",
        "fJSD_vs_AFDB",
        "fJSD_C_vs_PDB",
        "fJSD_A_vs_PDB",
        "fJSD_T_vs_PDB",
        "fJSD_C_vs_AFDB",
        "fJSD_A_vs_AFDB",
        "fJSD_T_vs_AFDB",
    ):
        if key in paper_metrics:
            ordered[key] = _to_float(paper_metrics[key])
    return ordered
