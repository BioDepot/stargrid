#!/usr/bin/env python3
"""Score prepared STAR/Space-Ranger Flex fields on frozen H&E Cellpose masks.

The primary comparison is restricted to the Space Ranger barcode axis. Counts
at the requested HD scale are distributed uniformly over their H&E-supported
2-um child centers solely for morphology scoring. Spatial-bin shared and
positive-residual fields are reported separately; they are not feature-bin or
read-member decompositions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from score_hd_flex_registration_roi import (  # noqa: E402
    RoiGeometry,
    build_roi_geometry,
    score_field,
    sha256,
)
from hd_mask_shift import correct_geometry, export_geometry  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_axis_geometry(
    base: RoiGeometry,
    coordinates: np.ndarray | list[tuple[int, int]],
    *,
    width: int,
    height: int,
    scale_um: int,
    allow_unsupported_coarse_barcodes: bool = False,
) -> tuple[RoiGeometry, np.ndarray, np.ndarray]:
    """Select 2-um children of a coarse barcode axis and retain 16-um parents."""
    if scale_um < 2 or scale_um % 2:
        raise ValueError("comparison scale must be an even multiple of 2 um")
    factor = scale_um // 2
    coarse_width = math.ceil(width / factor)
    coarse_height = math.ceil(height / factor)
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coarse coordinates must have row/column pairs")
    axis_ids = coordinates[:, 0] * coarse_width + coordinates[:, 1]
    if len(axis_ids) != len(np.unique(axis_ids)):
        raise ValueError("duplicate coarse barcode coordinates")
    if np.any(axis_ids < 0) or np.any(axis_ids >= coarse_width * coarse_height):
        raise ValueError("coarse barcode outside declared grid")
    axis_lookup = np.full(coarse_width * coarse_height, -1, dtype=np.int32)
    axis_lookup[axis_ids] = np.arange(len(axis_ids), dtype=np.int32)
    rows = base.indices // width
    columns = base.indices % width
    base_coarse = (rows // factor) * coarse_width + columns // factor
    child_axis = axis_lookup[base_coarse]
    selected = child_axis >= 0
    child_axis = child_axis[selected]
    child_counts = np.bincount(child_axis, minlength=len(axis_ids)).astype(np.int32)
    if np.any(child_counts == 0) and not allow_unsupported_coarse_barcodes:
        raise ValueError("coarse barcode has no H&E-supported 2-um child center")
    indices = base.indices[selected]
    states = base.states[selected]
    parent_size = 8
    parent_width = math.ceil(width / parent_size)
    parent_height = math.ceil(height / parent_size)
    parent_ids = (
        (indices // width) // parent_size * parent_width
        + (indices % width) // parent_size
    )
    parent_counts = np.bincount(
        parent_ids, minlength=parent_width * parent_height,
    )
    complete = parent_counts == parent_size ** 2
    geometry = RoiGeometry(
        indices=indices,
        states=states,
        parent_ids=parent_ids,
        complete_parent_mask=complete,
        audit={
            "comparison_scale_um": scale_um,
            "coarse_barcodes": len(axis_ids),
            "he_unsupported_coarse_barcodes": int(np.count_nonzero(child_counts == 0)),
            "he_supported_2um_children": len(indices),
            "complete_16um_parents": int(np.count_nonzero(complete)),
        },
    )
    return geometry, child_axis, child_counts


def expand_uniformly(
    coarse_field: np.ndarray,
    child_axis: np.ndarray,
    child_counts: np.ndarray,
) -> np.ndarray:
    coarse_field = np.asarray(coarse_field, dtype=np.float64)
    if coarse_field.shape != child_counts.shape:
        raise ValueError("coarse field and child-count axes differ")
    if np.any(~np.isfinite(coarse_field)) or np.any(coarse_field < 0):
        raise ValueError("invalid coarse expression field")
    expanded = coarse_field[child_axis] / child_counts[child_axis]
    supported = child_counts > 0
    if not math.isclose(
        float(np.sum(expanded, dtype=np.float64)),
        float(np.sum(coarse_field[supported], dtype=np.float64)),
        rel_tol=1e-12,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "uniform coarse-to-2um expansion does not preserve H&E-supported mass"
        )
    return expanded


def spatial_bin_components(
    star: np.ndarray,
    space_ranger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if star.shape != space_ranger.shape:
        raise ValueError("STAR and Space Ranger spatial axes differ")
    shared = np.minimum(star, space_ranger)
    star_only = np.clip(star - space_ranger, 0.0, None)
    space_ranger_only = np.clip(space_ranger - star, 0.0, None)
    if not np.allclose(shared + star_only, star, rtol=0, atol=1e-10):
        raise ValueError("STAR spatial-bin residual does not reconcile")
    if not np.allclose(
        shared + space_ranger_only, space_ranger, rtol=0, atol=1e-10,
    ):
        raise ValueError("Space Ranger spatial-bin residual does not reconcile")
    return shared, star_only, space_ranger_only


def add_scope(row: dict[str, object], scope: str) -> dict[str, object]:
    return {**row, "scope": scope}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-fields", type=Path, required=True)
    parser.add_argument("--cellpose-segmentation", type=Path, required=True)
    parser.add_argument("--capture-grid-json", type=Path, required=True)
    parser.add_argument("--native-registration-json", type=Path, required=True)
    parser.add_argument("--registration-moving-source-downsample", type=int, default=16)
    parser.add_argument(
        "--registration-moving-sampling", choices=("area", "decimate"),
        default="decimate",
    )
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--slide", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mask-shift-squares", nargs=2, type=int, default=(0, 0), metavar=("DY", "DX"))
    parser.add_argument("--mask-shift-reason", default="No mask translation requested")
    parser.add_argument("--mask-affine-json", type=Path)
    parser.add_argument("--export-scoring-geometry", action="store_true")
    parser.add_argument(
        "--allow-unsupported-coarse-barcodes",
        action="store_true",
        help=(
            "exclude coarse barcodes lacking independently registered H&E pixel "
            "support symmetrically from all morphology fields"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (any(args.mask_shift_squares) or args.mask_affine_json) and args.mask_shift_reason == "No mask translation requested":
        raise ValueError("a nonzero mask shift requires --mask-shift-reason")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    prepared_summary_path = args.prepared_fields / "summary.json"
    prepared_arrays_path = args.prepared_fields / "coarse_fields.npz"
    for path in (
        prepared_summary_path,
        prepared_arrays_path,
        args.cellpose_segmentation,
        args.capture_grid_json,
        args.native_registration_json,
    ):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    prepared = json.loads(prepared_summary_path.read_text(encoding="utf-8"))
    if prepared.get("schema") != "visium_hd_processing.flex_star_sr_prepared_fields.v1":
        raise ValueError("unexpected prepared-field schema")
    if prepared.get("dataset") != args.dataset:
        raise ValueError("prepared-field dataset differs from requested dataset")
    if sha256(prepared_arrays_path) != prepared["outputs"]["coarse_fields.npz"]["sha256"]:
        raise ValueError("prepared-field array hash differs from its summary")
    scale_um = int(prepared["comparison_scale_um"])
    methods = prepared["methods"]

    geometry_args = argparse.Namespace(
        native_registration_json=args.native_registration_json,
        roi_manifest=None,
        capture_grid_json=args.capture_grid_json,
        tissue_positions=None,
        space_ranger_alignment_json=None,
        cellpose_segmentation=args.cellpose_segmentation,
        barcode_mappings=None,
        registration_moving_source_downsample=args.registration_moving_source_downsample,
        registration_moving_sampling=args.registration_moving_sampling,
        width=args.width,
        height=args.height,
        parent_size=8,
    )
    base_geometry = build_roi_geometry(geometry_args)
    base_geometry = correct_geometry(
        base_geometry, shift=args.mask_shift_squares, affine=args.mask_affine_json,
        width=args.width, height=args.height,
    )
    with np.load(prepared_arrays_path, allow_pickle=False) as arrays:
        sr_coordinates = np.asarray(arrays["space_ranger_coordinates"], dtype=np.int32)
        sr_bins = np.asarray(arrays["space_ranger_bin_totals"], dtype=np.float64)
        common_geometry, common_child_axis, common_child_counts = build_axis_geometry(
            base_geometry, sr_coordinates, width=args.width, height=args.height,
            scale_um=scale_um,
            allow_unsupported_coarse_barcodes=(
                args.allow_unsupported_coarse_barcodes
            ),
        )
        sr_field = expand_uniformly(sr_bins, common_child_axis, common_child_counts)
        rows = [add_scope(score_field(
            "space_ranger_common_axis", "space_ranger_whole", "space_ranger",
            sr_field, common_geometry,
        ), "space_ranger_barcode_axis")]
        reconciliation: list[dict[str, object]] = []
        for label in sorted(methods):
            star_on_sr = np.asarray(
                arrays[f"{label}_star_common_bin_totals"], dtype=np.float64,
            )
            star_coordinates = np.asarray(
                arrays[f"{label}_star_full_coordinates"], dtype=np.int32,
            )
            star_bins = np.asarray(
                arrays[f"{label}_star_full_bin_totals"], dtype=np.float64,
            )
            shared, star_only, sr_only = spatial_bin_components(star_on_sr, sr_bins)
            common_fields = {
                f"{label}_star_common_axis": ("star_whole_common_axis", star_on_sr),
                f"{label}_shared_spatial_bin_mass": ("shared_spatial_bin_mass", shared),
                f"{label}_star_positive_spatial_bin_residual": (
                    "star_positive_spatial_bin_residual", star_only,
                ),
                f"{label}_space_ranger_positive_spatial_bin_residual": (
                    "space_ranger_positive_spatial_bin_residual", sr_only,
                ),
            }
            for name, (role, field) in common_fields.items():
                rows.append(add_scope(score_field(
                    name, role, label,
                    expand_uniformly(field, common_child_axis, common_child_counts),
                    common_geometry,
                ), "space_ranger_barcode_axis"))
            full_geometry, full_child_axis, full_child_counts = build_axis_geometry(
                base_geometry, star_coordinates, width=args.width, height=args.height,
                scale_um=scale_um,
                allow_unsupported_coarse_barcodes=(
                    args.allow_unsupported_coarse_barcodes
                ),
            )
            rows.append(add_scope(score_field(
                f"{label}_star_full_axis", "star_whole_full_capture_axis", label,
                expand_uniformly(star_bins, full_child_axis, full_child_counts),
                full_geometry,
            ), "star_capture_barcode_axis"))
            reconciliation.append({
                "dataset": args.dataset,
                "method": label,
                "comparison_scale_um": scale_um,
                **{
                    key: value for key, value in methods[label].items()
                    if key != "inputs"
                },
            })

    args.out_dir.mkdir(parents=True)
    geometry_path = args.out_dir / "scoring_geometry.npz"
    if args.export_scoring_geometry:
        export_geometry(geometry_path, base_geometry, width=args.width, height=args.height)
    fields_path = args.out_dir / "morphology_fields.tsv"
    reconciliation_path = args.out_dir / "mass_reconciliation.tsv"
    write_rows(fields_path, rows)
    write_rows(reconciliation_path, reconciliation)
    cellpose_summary = args.cellpose_segmentation / "summary.json"
    summary = {
        "schema": "visium_hd_processing.flex_star_sr_he_morphology.v1",
        "dataset": args.dataset,
        "slide": args.slide,
        "area": args.area,
        "comparison_scale_um": scale_um,
        "mask_shift": {
            "squares_dy_dx": list(args.mask_shift_squares),
            "micrometres_dy_dx": [2 * v for v in args.mask_shift_squares],
            "euclidean_micrometres": 2 * math.hypot(*args.mask_shift_squares),
            "reason": args.mask_shift_reason,
            "sign_convention": "state and support at (r,c) move to (r+dy,c+dx)",
            "boundary_policy": "discard shifted-out squares; newly exposed squares have no support; no wrapping",
            "applied_before": "coarse-axis restriction and all field scoring",
        },
        "primary_comparison_axis": "space_ranger_barcode_axis",
        "morphology_projection": (
            "Each coarse-bin count is distributed uniformly over its independent "
            "H&E-supported 2-um child centers; no expression enters registration or segmentation."
        ),
        "unsupported_coarse_barcode_policy": (
            "excluded symmetrically from STAR and Space Ranger morphology fields; "
            "retained in separate whole-slide count accounting"
            if args.allow_unsupported_coarse_barcodes
            else "not allowed"
        ),
        "residual_definition": (
            "Positive differences of total spatial-bin mass; not feature-bin or read-member residuals."
        ),
        "inputs": {
            "prepared_fields": {
                "path": str(args.prepared_fields.resolve()),
                "summary_sha256": sha256(prepared_summary_path),
                "arrays_sha256": sha256(prepared_arrays_path),
            },
            "space_ranger": prepared["space_ranger"],
            "star_methods": {
                label: value["inputs"] for label, value in methods.items()
            },
            "cellpose_summary": {
                "path": str(cellpose_summary.resolve()),
                "sha256": sha256(cellpose_summary),
            },
            "capture_grid": {
                "path": str(args.capture_grid_json.resolve()),
                "sha256": sha256(args.capture_grid_json),
            },
            "native_registration": {
                "path": str(args.native_registration_json.resolve()),
                "sha256": sha256(args.native_registration_json),
            },
        },
        "geometry_audit": {
            "base": base_geometry.audit,
            "common_axis": common_geometry.audit,
        },
        "outputs": {
            fields_path.name: {"bytes": fields_path.stat().st_size, "sha256": sha256(fields_path)},
            reconciliation_path.name: {
                "bytes": reconciliation_path.stat().st_size,
                "sha256": sha256(reconciliation_path),
            },
        },
        "invariants": {
            "space_ranger_is_comparator_not_truth": True,
            "biological_feature_axis_only": True,
            "diagnostic_non_panel_counts_excluded": True,
            "raw_mass_not_normalized": True,
            "image_or_segmentation_not_used_for_assignment": True,
            "existing_expression_and_image_artifacts_only": True,
        },
    }
    if any(args.mask_shift_squares) or args.mask_affine_json:
        summary["morphology_projection"] = (
            "Each coarse-bin count is distributed uniformly over its supported 2-um child centers. "
            "The frozen image-derived states and support are translated together by the disclosed "
            "count-guided whole-square correction before any field is scored. "
            "No new image registration or segmentation is fitted."
        )
    if args.export_scoring_geometry:
        summary["outputs"][geometry_path.name] = {
            "bytes": geometry_path.stat().st_size, "sha256": sha256(geometry_path),
        }
        summary["scoring_geometry_export"] = "shifted base geometry before coarse-axis restriction"
    if args.mask_affine_json:
        summary['morphology_projection'] = (
            'Each coarse-bin count is distributed uniformly over its supported 2-um child centers. '
            'Frozen image-derived states and support are resampled together by the disclosed '
            'count-guided affine correction before scoring. No segmentation is refitted.')
        summary['mask_affine'] = {
            'fit_summary':str(args.mask_affine_json.resolve()),
            'sha256':sha256(args.mask_affine_json),
            'coefficients':base_geometry.audit['mask_affine_coefficients'],
            'model':base_geometry.audit['affine_model'],
            'resampling':base_geometry.audit['resampling'],
            'reason':args.mask_shift_reason,
        }
    summary["provenance"] = {
        "command": sys.argv,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cpus": os.environ.get("OFF_BENCH_CPUS_USED"),
        "generator_sha256": sha256(Path(__file__)),
        "mask_shift_helper_sha256": sha256(Path(__file__).with_name("hd_mask_shift.py")),
        "geometry_helper_sha256": sha256(Path(__file__).with_name("score_hd_flex_registration_roi.py")),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (fields_path, reconciliation_path, summary_path, *([geometry_path] if args.export_scoring_geometry else []))
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
