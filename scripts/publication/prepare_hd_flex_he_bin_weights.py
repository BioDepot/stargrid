#!/usr/bin/env python3
"""Project a frozen H&E/Cellpose segmentation onto a coarse barcode axis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from score_hd_flex_registration_roi import build_roi_geometry, sha256  # noqa: E402
from score_hd_flex_star_sr_morphology import build_axis_geometry  # noqa: E402
from hd_mask_shift import correct_geometry, load_affine  # noqa: E402


def compartment_weights(
    states: np.ndarray,
    child_axis: np.ndarray,
    child_counts: np.ndarray,
    allow_unsupported: bool = False,
) -> dict[str, np.ndarray]:
    """Return per-coarse-bin H&E compartment fractions."""
    states = np.asarray(states, dtype=np.uint8)
    child_axis = np.asarray(child_axis, dtype=np.int64)
    child_counts = np.asarray(child_counts, dtype=np.int64)
    if states.shape != child_axis.shape or np.any(child_counts < 0) or (not allow_unsupported and np.any(child_counts == 0)):
        raise ValueError("invalid coarse-bin H&E child mapping")
    count = len(child_counts)
    cell_children = np.bincount(
        child_axis, weights=(states > 0), minlength=count,
    ).astype(np.float64)
    nucleus_children = np.bincount(
        child_axis, weights=(states == 2), minlength=count,
    ).astype(np.float64)
    if np.any(nucleus_children > cell_children) or np.any(cell_children > child_counts):
        raise ValueError("invalid H&E cell/nucleus hierarchy")
    denominator = np.maximum(child_counts, 1).astype(np.float64)
    cell_fraction = cell_children / denominator
    nucleus_fraction = nucleus_children / denominator
    extracellular_fraction = (child_counts > 0).astype(float) - cell_fraction
    if not np.allclose(
        cell_fraction + extracellular_fraction, (child_counts > 0).astype(float), rtol=0.0, atol=1e-15,
    ):
        raise ValueError("H&E compartment fractions do not reconcile")
    return {
        "he_supported_children": child_counts.astype(np.int32),
        "cell_fraction": cell_fraction,
        "nucleus_fraction": nucleus_fraction,
        "extracellular_fraction": extracellular_fraction,
    }


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
    parser.add_argument("--allow-unsupported-coarse-barcodes", action="store_true")
    parser.add_argument("--mask-affine-json", type=Path)
    parser.add_argument("--mask-shift-reason", default="No mask correction requested")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mask_affine_json and args.mask_shift_reason == 'No mask correction requested':
        raise ValueError('affine correction requires a reason')
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
    base_geometry = correct_geometry(base_geometry,width=args.width,height=args.height,
                                     affine=args.mask_affine_json)
    with np.load(prepared_arrays_path, allow_pickle=False) as arrays:
        coordinates = np.asarray(arrays["space_ranger_coordinates"], dtype=np.int32)
    coarse_geometry, child_axis, child_counts = build_axis_geometry(
        base_geometry,
        coordinates,
        width=args.width,
        height=args.height,
        scale_um=scale_um,
        allow_unsupported_coarse_barcodes=args.allow_unsupported_coarse_barcodes,
    )
    weights = compartment_weights(coarse_geometry.states, child_axis, child_counts, args.allow_unsupported_coarse_barcodes)
    if len(coordinates) != len(weights["cell_fraction"]):
        raise ValueError("H&E weights differ from the Space Ranger barcode axis")

    args.out_dir.mkdir(parents=True)
    weights_path = args.out_dir / "he_bin_weights.npz"
    np.savez_compressed(weights_path, coordinates=coordinates, **weights)
    cell_area = float(
        np.sum(weights["cell_fraction"] * weights["he_supported_children"])
        / np.sum(weights["he_supported_children"])
    )
    nucleus_area = float(
        np.sum(weights["nucleus_fraction"] * weights["he_supported_children"])
        / np.sum(weights["he_supported_children"])
    )
    summary = {
        "schema": "visium_hd_processing.flex_he_bin_weights.v1",
        "dataset": args.dataset,
        "slide": args.slide,
        "area": args.area,
        "comparison_scale_um": scale_um,
        "barcode_axis": "space_ranger_barcode_axis",
        "coarse_barcodes": int(len(coordinates)),
        "allow_unsupported_coarse_barcodes": args.allow_unsupported_coarse_barcodes,
        "he_unsupported_coarse_barcodes": int(np.count_nonzero(child_counts == 0)),
        "he_supported_2um_children": int(np.sum(weights["he_supported_children"])),
        "he_cell_area_fraction": cell_area,
        "he_nucleus_area_fraction": nucleus_area,
        "projection": (
            "Per-coarse-bin fractions of independently segmented H&E-supported "
            "2-um child centers; expression is not used to define the weights."
        ),
        "inputs": {
            "prepared_summary": {
                "path": str(prepared_summary_path.resolve()),
                "sha256": sha256(prepared_summary_path),
            },
            "prepared_arrays": {
                "path": str(prepared_arrays_path.resolve()),
                "sha256": sha256(prepared_arrays_path),
            },
            "cellpose_summary": {
                "path": str((args.cellpose_segmentation / "summary.json").resolve()),
                "sha256": sha256(args.cellpose_segmentation / "summary.json"),
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
            "coarse_axis": coarse_geometry.audit,
        },
        "outputs": {
            weights_path.name: {
                "bytes": weights_path.stat().st_size,
                "sha256": sha256(weights_path),
            }
        },
        "invariants": {
            "image_or_segmentation_not_used_for_assignment": True,
            "existing_segmentation_reused_without_rerun": True,
            "space_ranger_barcode_axis_is_primary": True,
        },
    }
    if args.mask_affine_json:
        summary['mask_affine'] = {
            'fit_summary':str(args.mask_affine_json.resolve()),
            'sha256':sha256(args.mask_affine_json),
            'coefficients':load_affine(args.mask_affine_json),
            'model':base_geometry.audit['affine_model'],
            'resampling':base_geometry.audit['resampling'],
            'reason':args.mask_shift_reason,
        }
        summary['projection'] = ('Per-coarse-bin fractions of frozen image-derived states; '
            'states and support resampled together by the disclosed count-guided affine correction.')
        summary['mask_shift_helper_sha256'] = sha256(Path(__file__).with_name('hd_mask_shift.py'))
    if not math.isclose(
        cell_area,
        float(np.mean(coarse_geometry.states > 0)),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("coarse H&E cell area fraction does not reconcile")
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        f"{sha256(weights_path)}  {weights_path.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
