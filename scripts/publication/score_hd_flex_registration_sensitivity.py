#!/usr/bin/env python3
"""Re-score frozen CRC expression fields after native image registration.

Expression matrices and capture-grid placement remain fixed. The only changed
input is the microscope-to-CytAssist transform used to project the frozen
cell/nucleus compartment raster onto the expression grid. Space Ranger image
registration is therefore a baseline comparator, not assignment ancestry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from score_hd_flex_full_compartments import (
    METRICS,
    POLICIES,
    comparison_rows,
    compatibility_fields,
    field_from_matrix,
    increment_field,
    load_compartments,
    load_mex,
    load_space_ranger_h5,
    mex_file,
    score_field,
    write_rows,
)


SENSITIVITY_METRICS = (
    "in_cell_mass",
    "in_cell_mass_fraction",
    "in_nucleus_mass",
    "in_nucleus_mass_fraction",
    "extracellular_mass",
    "extracellular_mass_fraction",
    "in_cell_roc_auc",
    "in_cell_average_precision",
    "in_nucleus_roc_auc",
    "in_nucleus_average_precision",
    "cell_fraction_pearson_16um",
    "nucleus_fraction_pearson_16um",
    "in_cell_normalized_tv",
    "in_nucleus_normalized_tv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-mex-root", type=Path, required=True)
    parser.add_argument("--vendor-h5", type=Path, required=True)
    parser.add_argument("--barcode-mappings", type=Path, required=True)
    parser.add_argument("--tissue-positions", type=Path, required=True)
    parser.add_argument("--native-registration-json", type=Path, required=True)
    parser.add_argument("--space-ranger-alignment-json", type=Path, required=True)
    parser.add_argument("--baseline-field-biology", type=Path, required=True)
    parser.add_argument("--frozen-registration-summary", type=Path,
                        help="Reuse accepted transforms after verifying their image/geometry inputs; do not fit a transform.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--native-moving-downsample", type=float, default=16.0)
    parser.add_argument("--fit-sample-stride", type=int, default=2500)
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--umi-mode", choices=("1mm_cr", "exact"), default="1mm_cr")
    parser.add_argument("--slide", default="H1-GMHFWPH")
    parser.add_argument("--area", default="D1")
    parser.add_argument("--skip-strict-increments", action="store_true",
                        help="Match a baseline that omits undefined increments for non-nested post-collapse policies.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_matrix(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    value: object = json.loads(path.read_text())
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing JSON matrix key {'.'.join(keys)} in {path}")
        value = value[key]
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid 3x3 matrix at {'.'.join(keys)} in {path}")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"singular matrix at {'.'.join(keys)} in {path}")
    return matrix / matrix[2, 2]


def frozen_transforms(summary_path: Path, geometry_inputs: tuple[Path, ...]):
    summary = json.loads(summary_path.read_text())
    for path in geometry_inputs:
        if summary["inputs"].get(str(path.resolve())) != sha256(path):
            raise ValueError(f"frozen registration input drift: {path}")
    grid = read_matrix(summary_path, ("transforms", "grid_to_microscope"))
    relative = read_matrix(summary_path, ("transforms", "old_mask_grid_from_expression_grid"))
    return grid, summary["grid_to_microscope_fit"], relative


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center
    mean_distance = float(np.mean(np.sqrt(np.sum(centered * centered, axis=1))))
    if mean_distance <= 0.0:
        raise ValueError("degenerate homography point set")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def fit_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("homography point arrays must both have shape (n, 2)")
    if len(source) < 4:
        raise ValueError("at least four homography correspondences are required")
    src, src_norm = normalize_points(source.astype(np.float64))
    dst, dst_norm = normalize_points(target.astype(np.float64))
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append((-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u))
        rows.append((0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v))
    _, _, right = np.linalg.svd(np.asarray(rows), full_matrices=False)
    normalized_h = right[-1].reshape(3, 3)
    result = np.linalg.inv(dst_norm) @ normalized_h @ src_norm
    return result / result[2, 2]


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = (matrix @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("homography projects points to infinity")
    return mapped[:, :2] / mapped[:, 2:3]


def fit_grid_to_microscope(
    path: Path, *, width: int, height: int, sample_stride: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if sample_stride <= 0:
        raise ValueError("sample stride must be positive")
    parquet = pq.ParquetFile(path)
    required = {
        "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres",
    }
    if not required.issubset(parquet.schema.names):
        raise ValueError("tissue positions lack grid-to-microscope columns")
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    offset = 0
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=100_000):
        values = batch.to_pydict()
        count = len(values["array_row"])
        first = (-offset) % sample_stride
        indices = np.arange(first, count, sample_stride, dtype=np.int64)
        if len(indices):
            source_parts.append(np.column_stack((
                np.asarray(values["array_col"], dtype=np.float64)[indices],
                np.asarray(values["array_row"], dtype=np.float64)[indices],
            )))
            target_parts.append(np.column_stack((
                np.asarray(values["pxl_col_in_fullres"], dtype=np.float64)[indices],
                np.asarray(values["pxl_row_in_fullres"], dtype=np.float64)[indices],
            )))
        offset += count
    if offset != width * height:
        raise ValueError(f"unexpected tissue-position grid size: {offset}")
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    matrix = fit_homography(source, target)
    residuals = np.sqrt(np.sum((project(matrix, source) - target) ** 2, axis=1))
    metrics: dict[str, float | int] = {
        "grid_rows": offset,
        "fit_points": len(source),
        "fit_mean_residual_microscope_pixels": float(np.mean(residuals)),
        "fit_rmse_microscope_pixels": float(np.sqrt(np.mean(residuals ** 2))),
        "fit_max_residual_microscope_pixels": float(np.max(residuals)),
    }
    if metrics["fit_max_residual_microscope_pixels"] > 0.05:
        raise ValueError(f"grid-to-microscope homography residual is too large: {metrics}")
    return matrix, metrics


def relative_mask_transform(
    grid_to_microscope: np.ndarray,
    sr_microscope_to_cytassist: np.ndarray,
    native_downsampled_microscope_to_cytassist: np.ndarray,
    native_moving_downsample: float,
) -> np.ndarray:
    if native_moving_downsample <= 0.0:
        raise ValueError("native moving downsample must be positive")
    raw_to_native_input = np.diag([
        1.0 / native_moving_downsample,
        1.0 / native_moving_downsample,
        1.0,
    ])
    native_raw_to_cytassist = (
        native_downsampled_microscope_to_cytassist @ raw_to_native_input
    )
    result = (
        np.linalg.inv(grid_to_microscope)
        @ np.linalg.inv(native_raw_to_cytassist)
        @ sr_microscope_to_cytassist
        @ grid_to_microscope
    )
    return result / result[2, 2]


def remap_compartments(
    states: np.ndarray, transform: np.ndarray, *, width: int, height: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if states.shape != (width * height,):
        raise ValueError("compartment raster has the wrong size")
    output = np.zeros_like(states)
    displacement_parts = []
    out_of_bounds = 0
    columns = np.arange(width, dtype=np.float64)
    for row in range(height):
        points = np.column_stack((columns, np.full(width, row, dtype=np.float64)))
        mapped = project(transform, points)
        mapped_columns = np.rint(mapped[:, 0]).astype(np.int64)
        mapped_rows = np.rint(mapped[:, 1]).astype(np.int64)
        valid = (
            (mapped_columns >= 0) & (mapped_columns < width)
            & (mapped_rows >= 0) & (mapped_rows < height)
        )
        target = output[row * width:(row + 1) * width]
        target[valid] = states[mapped_rows[valid] * width + mapped_columns[valid]]
        out_of_bounds += int(np.count_nonzero(~valid))
        displacement_parts.append(np.sqrt(
            (mapped[:, 0] - columns) ** 2 + (mapped[:, 1] - row) ** 2
        ))
    displacement = np.concatenate(displacement_parts)
    changed = output != states
    original_cell = states > 0
    remapped_cell = output > 0
    original_nucleus = states == 2
    remapped_nucleus = output == 2
    metrics: dict[str, float | int] = {
        "grid_bins": states.size,
        "changed_state_bins": int(np.count_nonzero(changed)),
        "changed_state_fraction": float(np.mean(changed)),
        "cell_mask_xor_bins": int(np.count_nonzero(original_cell ^ remapped_cell)),
        "cell_mask_xor_fraction": float(np.mean(original_cell ^ remapped_cell)),
        "nucleus_mask_xor_bins": int(np.count_nonzero(original_nucleus ^ remapped_nucleus)),
        "nucleus_mask_xor_fraction": float(np.mean(original_nucleus ^ remapped_nucleus)),
        "out_of_bounds_bins": out_of_bounds,
        "mean_displacement_2um_bins": float(np.mean(displacement)),
        "median_displacement_2um_bins": float(np.median(displacement)),
        "maximum_displacement_2um_bins": float(np.max(displacement)),
        "mean_displacement_um": float(2.0 * np.mean(displacement)),
        "median_displacement_um": float(2.0 * np.median(displacement)),
        "maximum_displacement_um": float(2.0 * np.max(displacement)),
    }
    if not np.isin(output, (0, 1, 2)).all():
        raise ValueError("remapped compartment raster contains an unknown state")
    return output, metrics


def build_fields(args: argparse.Namespace, states: np.ndarray) -> list[dict[str, object]]:
    vendor, target_genes = load_space_ranger_h5(args.vendor_h5, 2)
    rows = [score_field(
        "space_ranger", "vendor_compatibility_context", "space_ranger",
        field_from_matrix(vendor, args.width, args.height), states,
        args.width, args.height,
    )]
    strict = load_mex(args.policy_mex_root / "strict" / "square_002um", 2)
    if {feature for feature, _ in strict} - target_genes:
        raise ValueError("strict matrix contains off-target features")
    for policy, directory in POLICIES:
        root = args.policy_mex_root / directory / "square_002um"
        values = strict if policy == "strict" else load_mex(root, 2)
        if {feature for feature, _ in values} - target_genes:
            raise ValueError(f"{policy} matrix contains off-target features")
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
        if policy != "strict" and not args.skip_strict_increments:
            rows.append(score_field(
                f"{policy}_increment_over_strict", "ambiguous_increment_over_strict",
                policy, increment_field(values, strict, args.width, args.height),
                states, args.width, args.height,
            ))
    return rows


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sensitivity_rows(
    baseline_rows: list[dict[str, str]], native_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    baseline = {row["field"]: row for row in baseline_rows}
    native = {str(row["field"]): row for row in native_rows}
    if set(baseline) != set(native):
        raise ValueError("baseline and native-registration field sets differ")
    output = []
    for field in sorted(native):
        baseline_mass = float(baseline[field]["raw_molecule_mass"])
        native_mass = float(native[field]["raw_molecule_mass"])
        if not math.isclose(baseline_mass, native_mass, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"registration changed expression mass for {field}")
        for metric in SENSITIVITY_METRICS:
            old = float(baseline[field][metric])
            new = float(native[field][metric])
            output.append({
                "field": field,
                "field_role": native[field]["field_role"],
                "policy": native[field]["policy"],
                "metric": metric,
                "space_ranger_registration_value": old,
                "native_registration_value": new,
                "native_minus_space_ranger_registration": new - old,
                "absolute_difference": abs(new - old),
            })
    return output


def contrast_sensitivity(baseline_rows, native_rows):
    baseline = {(row["comparison"], row["metric"]): row
                for row in comparison_rows(baseline_rows)}
    output = []
    for row in comparison_rows(native_rows):
        if not row["comparison"].endswith("_vs_space_ranger") or row["metric"] == "raw_molecule_mass":
            continue
        old = baseline[(row["comparison"], row["metric"])]
        output.append({"comparison": row["comparison"], "metric": row["metric"],
                       "baseline_contrast": old["difference"], "native_contrast": row["difference"],
                       "absolute_change": abs(row["difference"] - old["difference"])})
    if not output or any(not math.isfinite(row["absolute_change"]) for row in output):
        raise ValueError("invalid registration contrast sensitivity")
    return output


def main() -> int:
    args = parse_args()
    required = (
        args.vendor_h5, args.barcode_mappings, args.tissue_positions,
        args.native_registration_json, args.space_ranger_alignment_json,
        args.baseline_field_biology,
    )
    if args.frozen_registration_summary:
        required += (args.frozen_registration_summary,)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    baseline_states = load_compartments(args.barcode_mappings, args.width, args.height)
    if args.frozen_registration_summary:
        grid_to_microscope, fit_metrics, mask_transform = frozen_transforms(
            args.frozen_registration_summary,
            (args.barcode_mappings, args.tissue_positions, args.native_registration_json,
             args.space_ranger_alignment_json),
        )
    else:
        grid_to_microscope, fit_metrics = fit_grid_to_microscope(
            args.tissue_positions, width=args.width, height=args.height,
            sample_stride=args.fit_sample_stride,
        )
        sr_transform = read_matrix(
            args.space_ranger_alignment_json, ("cytAssistInfo", "transformImages")
        )
        native_transform = read_matrix(args.native_registration_json, ("moving_to_fixed",))
        mask_transform = relative_mask_transform(
            grid_to_microscope, sr_transform, native_transform, args.native_moving_downsample,
        )
    remapped_states, mask_metrics = remap_compartments(
        baseline_states, mask_transform, width=args.width, height=args.height,
    )
    rows = build_fields(args, remapped_states)
    baseline_rows = load_rows(args.baseline_field_biology)
    sensitivity = sensitivity_rows(baseline_rows, rows)
    contrast_changes = contrast_sensitivity(baseline_rows, rows)

    fields_path = args.out_dir / "field_biology_native_registration.tsv"
    comparisons_path = args.out_dir / "comparison_deltas_native_registration.tsv"
    sensitivity_path = args.out_dir / "registration_sensitivity.tsv"
    write_rows(fields_path, rows)
    write_rows(comparisons_path, comparison_rows(rows))
    write_rows(sensitivity_path, sensitivity)

    max_by_metric = {}
    for metric in SENSITIVITY_METRICS:
        selected = [row for row in sensitivity if row["metric"] == metric]
        maximum = max(selected, key=lambda row: float(row["absolute_difference"]))
        max_by_metric[metric] = {
            "maximum_absolute_difference": maximum["absolute_difference"],
            "field": maximum["field"],
        }
    summary = {
        "schema": "visium_hd_processing.flex_registration_sensitivity.v1",
        "slide": args.slide,
        "area": args.area,
        "umi_mode": args.umi_mode,
        "scope": "morphology metrics only; expression matrices and capture-grid placement fixed",
        "strict_increments": "omitted_non_nested_postcollapse_policies" if args.skip_strict_increments else "require_elementwise_nesting",
        "registration_mode": "frozen_transform_reuse" if args.frozen_registration_summary else "fit_grid_and_compose",
        "grid_to_microscope_fit": fit_metrics,
        "mask_reprojection": mask_metrics,
        "transforms": {
            "grid_to_microscope": grid_to_microscope.tolist(),
            "old_mask_grid_from_expression_grid": mask_transform.tolist(),
        },
        "maximum_metric_changes": max_by_metric,
        "star_minus_vendor_contrast_changes": contrast_changes,
        "maximum_absolute_star_minus_vendor_contrast_change": max(row["absolute_change"] for row in contrast_changes),
        "inputs": {
            str(path.resolve()): sha256(path) for path in required
        },
        "outputs": {
            path.name: sha256(path)
            for path in (fields_path, comparisons_path, sensitivity_path)
        },
        "invariants": {
            "expression_matrices_unchanged": True,
            "raw_mass_unchanged": True,
            "capture_grid_placement_held_fixed": True,
            "space_ranger_transform_used_only_as_baseline_comparator": True,
            "native_registration_estimated_without_space_ranger_oracle": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "checksums.sha256").write_text("".join(
        f"{sha256(path)}  {path.name}\n"
        for path in (fields_path, comparisons_path, sensitivity_path, summary_path)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
