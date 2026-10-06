#!/usr/bin/env python3
"""Gate full-slide Cellpose work on the frozen CRC ROI comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PRIMARY_METRICS = (
    "in_cell_roc_auc",
    "in_nucleus_roc_auc",
    "cell_fraction_pearson_16um",
    "nucleus_fraction_pearson_16um",
)
POLICIES = ("postcollapse_soft", "postcollapse_hard")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(rows: dict[str, dict[str, str]], segmentation: dict, scoring: dict) -> tuple[list[dict], bool]:
    vendor = rows["space_ranger"]
    checks: list[dict] = []
    for policy in POLICIES:
        current = rows[policy]
        mass = float(current["raw_molecule_mass"])
        vendor_mass = float(vendor["raw_molecule_mass"])
        checks.append({
            "check": f"{policy}.raw_molecule_mass_at_least_space_ranger",
            "observed": mass - vendor_mass,
            "threshold": 0.0,
            "pass": mass >= vendor_mass,
        })
        for metric in PRIMARY_METRICS:
            difference = float(current[metric]) - float(vendor[metric])
            checks.append({
                "check": f"{policy}.{metric}_minus_space_ranger",
                "observed": difference,
                "threshold": 0.0,
                "pass": difference >= 0.0,
            })
        star_only = rows[f"{policy}_star_only"]
        vendor_only = rows[f"{policy}_space_ranger_only"]
        for metric in ("in_cell_roc_auc", "in_nucleus_roc_auc"):
            difference = float(star_only[metric]) - float(vendor_only[metric])
            checks.append({
                "check": f"{policy}.star_only_{metric}_minus_space_ranger_only",
                "observed": difference,
                "threshold": 0.0,
                "pass": difference >= 0.0,
            })
    counts = segmentation["counts"]
    checks.extend((
        {
            "check": "segmentation.retained_nucleus_labels_positive",
            "observed": int(counts["retained_nucleus_labels"]),
            "threshold": 1,
            "pass": int(counts["retained_nucleus_labels"]) > 0,
        },
        {
            "check": "segmentation.nucleus_outside_cell_pixels_zero",
            "observed": int(counts["nucleus_outside_cell_pixels"]),
            "threshold": 0,
            "pass": int(counts["nucleus_outside_cell_pixels"]) == 0,
        },
        {
            "check": "projection.supported_bin_fraction_at_least_0p999",
            "observed": (
                scoring["roi_geometry_audit"]["cellpose_supported_roi_2um_bins"]
                / scoring["roi_geometry_audit"]["native_roi_2um_bins"]
            ),
            "threshold": 0.999,
            "pass": (
                scoring["roi_geometry_audit"]["cellpose_supported_roi_2um_bins"]
                / scoring["roi_geometry_audit"]["native_roi_2um_bins"]
            ) >= 0.999,
        },
        {
            "check": "segmentation.expression_or_vendor_segmentation_not_loaded",
            "observed": bool(segmentation["input_contract"]["expression_or_vendor_segmentation_loaded"]),
            "threshold": False,
            "pass": not bool(segmentation["input_contract"]["expression_or_vendor_segmentation_loaded"]),
        },
    ))
    return checks, all(bool(row["pass"]) for row in checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-table", type=Path, required=True)
    parser.add_argument("--segmentation-summary", type=Path, required=True)
    parser.add_argument("--scoring-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.field_table, args.segmentation_summary, args.scoring_summary)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    with args.field_table.open(newline="") as handle:
        rows = {row["field"]: row for row in csv.DictReader(handle, delimiter="\t")}
    segmentation = json.loads(args.segmentation_summary.read_text())
    scoring = json.loads(args.scoring_summary.read_text())
    checks, passed = evaluate(rows, segmentation, scoring)
    table = args.out_dir / "roi_gate_checks.tsv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("check", "observed", "threshold", "pass"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(checks)
    summary = {
        "schema": "visium_hd_processing.he_cellpose_roi_gate.v1",
        "status": "pass" if passed else "fail",
        "checks_passed": sum(bool(row["pass"]) for row in checks),
        "checks_total": len(checks),
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {table.name: sha256(table)},
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "checksums.sha256").write_text(
        f"{sha256(table)}  {table.name}\n{sha256(summary_path)}  {summary_path.name}\n"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
