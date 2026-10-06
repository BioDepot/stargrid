#!/usr/bin/env python3
"""Score full-slide open Flex policies against image-derived compartments.

Space Ranger is used only to define compatibility shared/additional matrix
mass.  Cell and nucleus masks are the independent, noisy morphology target.
All fields retain their raw molecule mass; no policy is mass-normalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from compare_hd_flex_policy_matrices import load_mex, load_space_ranger_h5, mex_file
from star_spatial.hd_natural_image_oracle import (
    binary_density_metrics,
    compare_parent_image_support,
    distribution_distance,
)


POLICIES = (
    ("strict", "strict"),
    ("postcollapse_soft", "soft_expected"),
    ("postcollapse_hard", "hard"),
    ("gated_hard", "gated_hard"),
)
UNIT = re.compile(r"^s_002um_(\d+)_(\d+)(?:-\d+)?$")
METRICS = (
    ("raw_molecule_mass", "report_only"),
    ("in_cell_mass_fraction", "higher"),
    ("in_cell_area_normalized_enrichment", "higher"),
    ("in_nucleus_mass_fraction", "higher"),
    ("in_nucleus_area_normalized_enrichment", "higher"),
    ("extracellular_mass_fraction", "lower"),
    ("in_cell_roc_auc", "higher_secondary"),
    ("in_cell_average_precision", "higher_secondary"),
    ("in_nucleus_roc_auc", "higher_secondary"),
    ("in_nucleus_average_precision", "higher_secondary"),
    ("cell_fraction_pearson_16um", "higher_secondary"),
    ("nucleus_fraction_pearson_16um", "higher_secondary"),
    ("in_cell_normalized_tv", "lower_secondary"),
    ("in_nucleus_normalized_tv", "lower_secondary"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-mex-root", type=Path, required=True)
    parser.add_argument("--vendor-h5", type=Path, required=True)
    parser.add_argument("--barcode-mappings", type=Path, required=True)
    parser.add_argument("--umi-mode", choices=("1mm_cr", "exact"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--slide", default="H1-GMHFWPH")
    parser.add_argument("--area", default="D1")
    parser.add_argument("--skip-strict-increments", action="store_true",
                        help="Omit the undefined increment cohort when post-collapse policies are not nested; whole fields and vendor residuals remain unchanged.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_file(path: Path):
    """Import the optional vendor-compartment dependency only on use."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path)


def unit_index(unit: str, width: int, height: int) -> int:
    match = UNIT.fullmatch(unit)
    if match is None:
        raise ValueError(f"invalid 2 um unit: {unit}")
    row, column = int(match.group(1)), int(match.group(2))
    if row >= height or column >= width:
        raise ValueError(f"2 um unit outside grid: {unit}")
    return row * width + column


def load_compartments(path: Path, width: int, height: int) -> np.ndarray:
    states = np.zeros(width * height, dtype=np.uint8)
    offset = 0
    parquet = parquet_file(path)
    required = {"square_002um", "in_cell", "in_nucleus"}
    if not required.issubset(parquet.schema.names):
        raise ValueError("barcode mappings lack image compartment columns")
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=100_000):
        values = batch.to_pydict()
        for unit, in_cell, in_nucleus in zip(
            values["square_002um"], values["in_cell"], values["in_nucleus"],
            strict=True,
        ):
            index = unit_index(unit, width, height)
            if index != offset:
                raise ValueError("barcode mappings are not a complete row-major grid")
            if in_nucleus and not in_cell:
                raise ValueError(f"nucleus bin is not contained in cell mask: {unit}")
            states[index] = 2 if in_nucleus else (1 if in_cell else 0)
            offset += 1
    if offset != width * height:
        raise ValueError(f"incomplete barcode mapping grid: {offset}")
    return states


def add_value(field: np.ndarray, unit: str, value: float, width: int, height: int) -> None:
    if value < 0.0 or not math.isfinite(value):
        raise ValueError(f"invalid molecule mass: {value}")
    field[unit_index(unit, width, height)] += value


def field_from_matrix(
    values: dict[tuple[str, str], float], width: int, height: int,
) -> np.ndarray:
    field = np.zeros(width * height, dtype=np.float64)
    for (_, unit), value in values.items():
        add_value(field, unit, value, width, height)
    return field


def compatibility_fields(
    star: dict[tuple[str, str], float],
    vendor: dict[tuple[str, str], float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared, STAR-only, and vendor-only feature-by-bin mass fields."""
    shared = np.zeros(width * height, dtype=np.float64)
    star_only = np.zeros_like(shared)
    vendor_only = np.zeros_like(shared)
    for key, star_value in star.items():
        vendor_value = vendor.get(key, 0.0)
        common = min(star_value, vendor_value)
        if common:
            add_value(shared, key[1], common, width, height)
        if star_value > common:
            add_value(star_only, key[1], star_value - common, width, height)
        if vendor_value > common:
            add_value(vendor_only, key[1], vendor_value - common, width, height)
    for key, vendor_value in vendor.items():
        if key not in star:
            add_value(vendor_only, key[1], vendor_value, width, height)
    return shared, star_only, vendor_only


def increment_field(
    policy: dict[tuple[str, str], float],
    strict: dict[tuple[str, str], float],
    width: int,
    height: int,
    tolerance: float = 1e-9,
) -> np.ndarray:
    result = np.zeros(width * height, dtype=np.float64)
    for key, value in policy.items():
        difference = value - strict.get(key, 0.0)
        if difference < -tolerance:
            raise ValueError(f"strict mass is not a subset at {key}: {difference}")
        if difference > 0.0:
            add_value(result, key[1], difference, width, height)
    missing = [key for key, value in strict.items() if value > tolerance and key not in policy]
    if missing:
        raise ValueError(f"policy omits {len(missing)} strict feature-by-bin entries")
    return result


def mask_summary(field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = float(np.sum(field, dtype=np.float64))
    mass = float(np.sum(field[mask], dtype=np.float64))
    area_fraction = float(np.mean(mask))
    mass_fraction = mass / total if total else 0.0
    return {
        "mass": mass,
        "mass_fraction": mass_fraction,
        "area_fraction": area_fraction,
        "area_normalized_enrichment": mass_fraction / area_fraction if area_fraction else 0.0,
    }


def score_field(
    name: str,
    role: str,
    policy: str,
    field: np.ndarray,
    states: np.ndarray,
    width: int,
    height: int,
) -> dict[str, object]:
    total = float(np.sum(field, dtype=np.float64))
    in_cell = states > 0
    in_nucleus = states == 2
    extracellular = states == 0
    cell = mask_summary(field, in_cell)
    nucleus = mask_summary(field, in_nucleus)
    outside = mask_summary(field, extracellular)
    cell_classifier = binary_density_metrics(field, in_cell)
    nucleus_classifier = binary_density_metrics(field, in_nucleus)
    cell_tv = distribution_distance(field, in_cell.astype(np.float64))
    nucleus_tv = distribution_distance(field, in_nucleus.astype(np.float64))
    parent = compare_parent_image_support(
        field,
        np.zeros(field.size, dtype=bool),
        in_cell,
        in_nucleus,
        width=width,
        height=height,
    )["all_parents"]
    return {
        "field": name,
        "field_role": role,
        "policy": policy,
        "raw_molecule_mass": total,
        "occupied_2um_bins": int(np.count_nonzero(field)),
        "in_cell_mass": cell["mass"],
        "in_cell_mass_fraction": cell["mass_fraction"],
        "in_cell_area_fraction": cell["area_fraction"],
        "in_cell_area_normalized_enrichment": cell["area_normalized_enrichment"],
        "in_nucleus_mass": nucleus["mass"],
        "in_nucleus_mass_fraction": nucleus["mass_fraction"],
        "in_nucleus_area_fraction": nucleus["area_fraction"],
        "in_nucleus_area_normalized_enrichment": nucleus["area_normalized_enrichment"],
        "extracellular_mass": outside["mass"],
        "extracellular_mass_fraction": outside["mass_fraction"],
        "extracellular_area_fraction": outside["area_fraction"],
        "extracellular_area_normalized_enrichment": outside["area_normalized_enrichment"],
        "in_cell_roc_auc": cell_classifier["roc_auc"],
        "in_cell_average_precision": cell_classifier["average_precision"],
        "in_nucleus_roc_auc": nucleus_classifier["roc_auc"],
        "in_nucleus_average_precision": nucleus_classifier["average_precision"],
        "cell_fraction_pearson_16um": parent["cell_fraction_pearson"],
        "cell_fraction_spearman_16um": parent["cell_fraction_spearman"],
        "nucleus_fraction_pearson_16um": parent["nucleus_fraction_pearson"],
        "nucleus_fraction_spearman_16um": parent["nucleus_fraction_spearman"],
        "in_cell_normalized_tv": cell_tv["normalized_total_variation"],
        "in_nucleus_normalized_tv": nucleus_tv["normalized_total_variation"],
        "equal_mass_normalization_applied": False,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_name = {str(row["field"]): row for row in rows}
    output = []
    pairs = [
        (f"{policy}_vs_space_ranger", policy, "space_ranger")
        for policy, _ in POLICIES
    ] + [
        (f"{policy}_vs_strict", policy, "strict")
        for policy, _ in POLICIES if policy != "strict"
    ]
    for comparison, left_name, right_name in pairs:
        left, right = by_name[left_name], by_name[right_name]
        for metric, direction in METRICS:
            left_value, right_value = float(left[metric]), float(right[metric])
            difference = left_value - right_value
            preferred = (
                difference if direction.startswith("higher")
                else -difference if direction.startswith("lower") else ""
            )
            output.append({
                "comparison": comparison,
                "left_field": left_name,
                "right_field": right_name,
                "metric": metric,
                "preferred_direction": direction,
                "left_value": left_value,
                "right_value": right_value,
                "difference": difference,
                "preferred_signed_difference": preferred,
                "equal_mass_normalization_applied": False,
            })
    return output


def main() -> int:
    args = parse_args()
    for path in (args.vendor_h5, args.barcode_mappings):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    states = load_compartments(args.barcode_mappings, args.width, args.height)
    vendor, target_genes = load_space_ranger_h5(args.vendor_h5, 2)
    rows = [score_field(
        "space_ranger", "vendor_compatibility_context", "space_ranger",
        field_from_matrix(vendor, args.width, args.height), states,
        args.width, args.height,
    )]
    strict_root = args.policy_mex_root / "strict" / "square_002um"
    strict = load_mex(strict_root, 2)
    unexpected = {feature for feature, _ in strict} - target_genes
    if unexpected:
        raise SystemExit(f"strict matrix contains {len(unexpected)} off-target features")
    matrix_inputs: dict[str, dict[str, str]] = {}
    for policy, directory in POLICIES:
        root = args.policy_mex_root / directory / "square_002um"
        values = strict if policy == "strict" else load_mex(root, 2)
        unexpected = {feature for feature, _ in values} - target_genes
        if unexpected:
            raise SystemExit(f"{policy} matrix contains {len(unexpected)} off-target features")
        rows.append(score_field(
            policy, "whole_open_policy", policy,
            field_from_matrix(values, args.width, args.height), states,
            args.width, args.height,
        ))
        shared, star_only, vendor_only = compatibility_fields(
            values, vendor, args.width, args.height,
        )
        rows.extend((
            score_field(
                f"{policy}_shared_with_space_ranger", "shared_feature_bin_mass",
                policy, shared, states, args.width, args.height,
            ),
            score_field(
                f"{policy}_star_only", "open_additional_feature_bin_mass",
                policy, star_only, states, args.width, args.height,
            ),
            score_field(
                f"{policy}_space_ranger_only", "vendor_only_feature_bin_mass",
                policy, vendor_only, states, args.width, args.height,
            ),
        ))
        if not math.isclose(
            float(np.sum(shared)) + float(np.sum(star_only)),
            math.fsum(values.values()), rel_tol=1e-12, abs_tol=1e-6,
        ):
            raise SystemExit(f"STAR compatibility mass does not reconcile for {policy}")
        if not math.isclose(
            float(np.sum(shared)) + float(np.sum(vendor_only)),
            math.fsum(vendor.values()), rel_tol=1e-12, abs_tol=1e-6,
        ):
            raise SystemExit(f"vendor compatibility mass does not reconcile for {policy}")
        if policy != "strict" and not args.skip_strict_increments:
            rows.append(score_field(
                f"{policy}_increment_over_strict",
                "ambiguous_increment_over_strict",
                policy,
                increment_field(values, strict, args.width, args.height),
                states, args.width, args.height,
            ))
        matrix_inputs[policy] = {
            name: sha256(mex_file(root, name))
            for name in ("matrix.mtx", "features.tsv", "barcodes.tsv")
        }

    fields_path = args.out_dir / "field_biology.tsv"
    comparisons_path = args.out_dir / "comparison_deltas.tsv"
    write_rows(fields_path, rows)
    comparisons = comparison_rows(rows)
    write_rows(comparisons_path, comparisons)
    summary = {
        "schema": "visium_hd_processing.flex_full_compartment_biology.v1",
        "slide": args.slide,
        "area": args.area,
        "umi_mode": args.umi_mode,
        "field_definition": "raw feature-by-2um-bin molecule mass",
        "compatibility_decomposition": "per-feature-per-bin min/shared and positive residuals",
        "strict_increments": "omitted_non_nested_postcollapse_policies" if args.skip_strict_increments else "require_elementwise_nesting",
        "inputs": {
            "space_ranger_h5": {"path": str(args.vendor_h5.resolve()), "sha256": sha256(args.vendor_h5)},
            "barcode_mappings": {"path": str(args.barcode_mappings.resolve()), "sha256": sha256(args.barcode_mappings)},
            "open_policy_mex": matrix_inputs,
        },
        "outputs": {
            fields_path.name: sha256(fields_path),
            comparisons_path.name: sha256(comparisons_path),
        },
        "invariants": {
            "raw_mass_not_normalized": True,
            "strict_and_ambiguous_increment_reported_separately": not args.skip_strict_increments,
            "space_ranger_is_compatibility_not_truth": True,
            "cell_nucleus_masks_not_used_as_assignment_priors": True,
            "normalized_tv_is_secondary": True,
            "occupancy_jaccard_not_computed": True,
        },
        "limitations": [
            "The image-derived cell and nucleus masks are a noisy, non-transcriptomic oracle.",
            "Shared/additional fields are matrix-level feature-by-bin mass decompositions, not exact read-member-set matches.",
            "This engineering fixture does not establish biological superiority.",
        ],
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (fields_path, comparisons_path, summary_path)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
