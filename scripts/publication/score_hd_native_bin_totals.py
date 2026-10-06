#!/usr/bin/env python3
"""Score compact 2-um molecule fields against native Cellpose compartments."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from score_hd_flex_full_compartments import comparison_rows, write_rows
from score_hd_flex_registration_roi import build_roi_geometry, score_field, sha256


PRODUCTS = ("space_ranger", "strict", "soft_expected", "hard", "gated_hard")
PAPER_NAMES = {
    "space_ranger": "space_ranger",
    "strict": "strict",
    "soft_expected": "postcollapse_soft",
    "hard": "postcollapse_hard",
    "gated_hard": "gated_hard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-total-root", type=Path, required=True)
    parser.add_argument("--cellpose-segmentation", type=Path, required=True)
    parser.add_argument("--capture-grid-json", type=Path, required=True)
    parser.add_argument("--native-registration-json", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path)
    parser.add_argument("--registration-moving-source-downsample", type=int, default=8)
    parser.add_argument(
        "--registration-moving-sampling", choices=("area", "decimate"), default="decimate",
    )
    parser.add_argument("--slide", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--umi-mode", choices=("1mm_cr", "exact"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--parent-size", type=int, default=8)
    # Attributes consumed by the shared native-geometry builder.
    parser.set_defaults(
        tissue_positions=None,
        space_ranger_alignment_json=None,
        barcode_mappings=None,
    )
    return parser.parse_args()


def parse_unit(value: str, *, width: int, height: int) -> int:
    parts = value.split("_")
    if len(parts) != 4 or parts[:2] != ["s", "002um"] or not parts[3].endswith("-1"):
        raise ValueError(f"invalid 2 um unit: {value}")
    row, column = int(parts[2]), int(parts[3][:-2])
    if row < 0 or row >= height or column < 0 or column >= width:
        raise ValueError(f"2 um unit outside declared grid: {value}")
    return row * width + column


def load_field(
    path: Path, roi_indices: np.ndarray, *, width: int, height: int,
) -> tuple[np.ndarray, float, int]:
    field = np.zeros(len(roi_indices), dtype=np.float64)
    seen = np.zeros(width * height, dtype=bool)
    total_mass = 0.0
    occupied = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["unit_2um", "molecule_mass"]:
            raise ValueError(f"invalid bin-total schema: {path}")
        for row in reader:
            index = parse_unit(row["unit_2um"], width=width, height=height)
            if seen[index]:
                raise ValueError(f"duplicate bin-total unit: {path}")
            seen[index] = True
            value = float(row["molecule_mass"])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid bin-total mass: {path}")
            total_mass += value
            occupied += 1
            offset = int(np.searchsorted(roi_indices, index))
            if offset < len(roi_indices) and int(roi_indices[offset]) == index:
                field[offset] = value
    return field, total_mass, occupied


def split_delta(policy: np.ndarray, strict: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = policy - strict
    positive = np.maximum(delta, 0.0)
    removed = np.maximum(-delta, 0.0)
    if not math.isclose(
        float(policy.sum() - strict.sum()),
        float(positive.sum() - removed.sum()),
        rel_tol=1e-12, abs_tol=1e-6,
    ):
        raise ValueError("policy/strict signed spatial delta does not conserve mass")
    return positive, removed


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    if args.parent_size != 8:
        raise SystemExit("this scorer requires 8x8 children per 16 um parent")
    summary_path = args.bin_total_root / "summary.json"
    inputs = [summary_path, args.capture_grid_json, args.native_registration_json,
              args.cellpose_segmentation / "summary.json"]
    field_paths = {name: args.bin_total_root / f"{name}.bin_totals.tsv.gz" for name in PRODUCTS}
    inputs.extend(field_paths.values())
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    source_summary = json.loads(summary_path.read_text())
    if source_summary.get("schema") != "visium_hd_processing.policy_mex_bin_totals.v1":
        raise SystemExit("unexpected bin-total summary schema")

    args.out_dir.mkdir(parents=True)
    geometry = build_roi_geometry(args)
    fields: dict[str, np.ndarray] = {}
    field_audit: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    roles = {"space_ranger": "vendor_compatibility_context"}
    for name in PRODUCTS:
        field, full_mass, occupied = load_field(
            field_paths[name], geometry.indices, width=args.width, height=args.height,
        )
        declared = float(source_summary["products"][name]["mass"])
        if not math.isclose(full_mass, declared, rel_tol=1e-12, abs_tol=1e-6):
            raise SystemExit(f"{name} bin-total mass differs from its source summary")
        fields[name] = field
        field_audit[name] = {
            "full_slide_raw_mass": full_mass,
            "full_slide_occupied_bins": occupied,
            "native_roi_raw_mass": float(field.sum()),
        }
        rows.append(score_field(
            PAPER_NAMES[name], roles.get(name, "whole_open_policy"),
            PAPER_NAMES[name], field, geometry,
        ))

    delta_audit: dict[str, object] = {}
    strict = fields["strict"]
    for name in ("soft_expected", "hard", "gated_hard"):
        paper_name = PAPER_NAMES[name]
        positive, removed = split_delta(fields[name], strict)
        positive_mass, removed_mass = float(positive.sum()), float(removed.sum())
        delta_audit[paper_name] = {
            "positive_spatial_increment_mass": positive_mass,
            "strict_spatial_mass_removed_or_redistributed": removed_mass,
            "net_mass_difference": positive_mass - removed_mass,
        }
        if positive_mass:
            rows.append(score_field(
                f"{paper_name}_positive_increment_over_strict",
                "positive_spatial_increment_over_strict", paper_name, positive, geometry,
            ))
        if removed_mass:
            rows.append(score_field(
                f"{paper_name}_strict_mass_removed_or_redistributed",
                "negative_spatial_redistribution_from_strict", paper_name, removed, geometry,
            ))

    field_path = args.out_dir / "field_biology.tsv"
    comparison_path = args.out_dir / "comparison_deltas.tsv"
    write_rows(field_path, rows)
    write_rows(comparison_path, comparison_rows(rows))
    summary = {
        "schema": "visium_hd_processing.native_bin_total_biology.v1",
        "slide": args.slide,
        "area": args.area,
        "umi_mode": args.umi_mode,
        "roi_geometry_audit": geometry.audit,
        "field_audit": field_audit,
        "policy_delta_audit": delta_audit,
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {
            field_path.name: sha256(field_path),
            comparison_path.name: sha256(comparison_path),
        },
        "invariants": {
            "raw_mass_not_normalized": True,
            "occupancy_jaccard_not_computed": True,
            "soft_expected_reported_as_complete_field_with_signed_delta": True,
            "positive_and_negative_spatial_deltas_reconcile_to_net_mass": True,
            "feature_level_compatibility_not_recomputed": True,
            "vendor_not_used_for_registration_or_segmentation": True,
        },
    }
    output = args.out_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
