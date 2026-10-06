#!/usr/bin/env python3
"""Compare frozen genomic-MAPQ and MAPQ-off SPATCH CODEX score tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


POLICIES = ("strict", "soft_expected", "hard", "gated_hard")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite(value: str) -> float | None:
    if value in {"", "NA"}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def formatted(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: formatted(value) for key, value in row.items()})


def index(
    rows: list[dict[str, str]], fields: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate table key: {key}")
        result[key] = row
    return result


def require_equal(left: str, right: str, description: str) -> None:
    left_value, right_value = finite(left), finite(right)
    if left_value is None or right_value is None:
        if left != right:
            raise ValueError(f"{description} differs: {left!r} versus {right!r}")
    elif left_value != right_value:
        raise ValueError(f"{description} differs: {left_value} versus {right_value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genomic-method-summary", type=Path, required=True)
    parser.add_argument("--genomic-marker-metrics", type=Path, required=True)
    parser.add_argument("--mapq-off-method-summary", type=Path, required=True)
    parser.add_argument("--mapq-off-marker-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        args.genomic_method_summary,
        args.genomic_marker_metrics,
        args.mapq_off_method_summary,
        args.mapq_off_marker_metrics,
    ]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")

    genomic_summary = index(
        read_tsv(args.genomic_method_summary), ("method", "bin_size_um")
    )
    off_summary = index(
        read_tsv(args.mapq_off_method_summary), ("method", "bin_size_um")
    )
    genomic_markers = index(
        read_tsv(args.genomic_marker_metrics),
        ("method", "bin_size_um", "protein"),
    )
    off_markers = index(
        read_tsv(args.mapq_off_marker_metrics),
        ("method", "bin_size_um", "protein"),
    )
    bin_sizes = sorted(
        int(key[1])
        for key in genomic_summary
        if key[0] == "space_ranger_2020a"
    )
    if not bin_sizes:
        raise ValueError("no Space Ranger summary rows")

    macro_rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    for size in bin_sizes:
        sr_key = ("space_ranger_2020a", str(size))
        sr_genomic = genomic_summary[sr_key]
        sr_off = off_summary[sr_key]
        require_equal(
            sr_genomic["macro_roc_auc"],
            sr_off["macro_roc_auc"],
            f"Space Ranger macro AUC at {size} um",
        )
        sr_auc = float(sr_genomic["macro_roc_auc"])
        for policy in POLICIES:
            genomic = genomic_summary[(f"star_2020a_{policy}", str(size))]
            off = off_summary[(f"star_2020a_mapq_off_{policy}", str(size))]
            genomic_auc = float(genomic["macro_roc_auc"])
            off_auc = float(off["macro_roc_auc"])
            macro_rows.append({
                "policy": policy,
                "bin_size_um": size,
                "space_ranger_macro_roc_auc": sr_auc,
                "genomic_macro_roc_auc": genomic_auc,
                "mapq_off_macro_roc_auc": off_auc,
                "mapq_off_minus_genomic_macro_roc_auc": off_auc - genomic_auc,
                "genomic_minus_space_ranger_macro_roc_auc": genomic_auc - sr_auc,
                "mapq_off_minus_space_ranger_macro_roc_auc": off_auc - sr_auc,
                "genomic_evaluation_mass": float(genomic["method_evaluation_mass"]),
                "mapq_off_evaluation_mass": float(off["method_evaluation_mass"]),
            })

            proteins = sorted(
                key[2]
                for key in genomic_markers
                if key[:2] == (f"star_2020a_{policy}", str(size))
            )
            for protein in proteins:
                genomic_marker = genomic_markers[
                    (f"star_2020a_{policy}", str(size), protein)
                ]
                off_marker = off_markers[
                    (f"star_2020a_mapq_off_{policy}", str(size), protein)
                ]
                sr_genomic_marker = genomic_markers[
                    ("space_ranger_2020a", str(size), protein)
                ]
                sr_off_marker = off_markers[
                    ("space_ranger_2020a", str(size), protein)
                ]
                field = "roc_auc_top_protein_quantile"
                require_equal(
                    sr_genomic_marker[field],
                    sr_off_marker[field],
                    f"Space Ranger {protein} AUC at {size} um",
                )
                genomic_auc_value = finite(genomic_marker[field])
                off_auc_value = finite(off_marker[field])
                sr_auc_value = finite(sr_genomic_marker[field])
                marker_rows.append({
                    "policy": policy,
                    "bin_size_um": size,
                    "protein": protein,
                    "reference_marker_available": (
                        genomic_marker["reference_marker_available"]
                    ),
                    "space_ranger_roc_auc": (
                        sr_auc_value if sr_auc_value is not None else "NA"
                    ),
                    "genomic_roc_auc": (
                        genomic_auc_value if genomic_auc_value is not None else "NA"
                    ),
                    "mapq_off_roc_auc": (
                        off_auc_value if off_auc_value is not None else "NA"
                    ),
                    "mapq_off_minus_genomic_roc_auc": (
                        off_auc_value - genomic_auc_value
                        if off_auc_value is not None and genomic_auc_value is not None
                        else "NA"
                    ),
                })

    args.out_dir.mkdir(parents=True)
    macro_path = args.out_dir / "macro_auc_comparison.tsv"
    marker_path = args.out_dir / "marker_auc_comparison.tsv"
    write_tsv(macro_path, macro_rows)
    write_tsv(marker_path, marker_rows)
    payload = {
        "schema": "visium_hd_processing.spatch_mapq_codex_comparison.v1",
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
        "space_ranger_rows_identical_between_frozen_results": True,
        "macro_auc": {
            policy: {
                "maximum_absolute_mapq_off_minus_genomic": max(
                    abs(float(row["mapq_off_minus_genomic_macro_roc_auc"]))
                    for row in macro_rows
                    if row["policy"] == policy
                ),
                "mapq_off_better_scales_um": [
                    int(row["bin_size_um"])
                    for row in macro_rows
                    if row["policy"] == policy
                    and float(row["mapq_off_minus_genomic_macro_roc_auc"]) > 0
                ],
            }
            for policy in POLICIES
        },
        "outputs": {
            path.name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (macro_path, marker_path)
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
