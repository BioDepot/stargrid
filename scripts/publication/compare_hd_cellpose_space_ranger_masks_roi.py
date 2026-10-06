#!/usr/bin/env python3
"""Compare Cellpose and retained Space Ranger compartment masks on one HD ROI.

This is an aggregate geometry diagnostic.  It does not read expression matrices
and does not materialize per-bin membership.  Space Ranger compartments are a
compatibility calibration target, not biological truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from score_hd_flex_registration_roi import (
    load_cellpose_states,
    load_roi_states,
    matrix_from_json,
    select_native_roi,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_mask_metrics(reference: np.ndarray, query: np.ndarray) -> dict[str, int | float]:
    reference = np.asarray(reference, dtype=bool)
    query = np.asarray(query, dtype=bool)
    if reference.shape != query.shape or reference.ndim != 1:
        raise ValueError("mask vectors must be one-dimensional and shape matched")
    true_positive = int(np.count_nonzero(reference & query))
    false_positive = int(np.count_nonzero(~reference & query))
    false_negative = int(np.count_nonzero(reference & ~query))
    true_negative = int(np.count_nonzero(~reference & ~query))
    reference_positive = true_positive + false_negative
    query_positive = true_positive + false_positive
    if reference_positive == 0 or query_positive == 0:
        raise ValueError("mask comparison requires positive reference and query support")
    return {
        "common_bins": int(len(reference)),
        "reference_positive_bins": reference_positive,
        "query_positive_bins": query_positive,
        "true_positive_bins": true_positive,
        "false_positive_bins": false_positive,
        "false_negative_bins": false_negative,
        "true_negative_bins": true_negative,
        "reference_fraction": reference_positive / len(reference),
        "query_fraction": query_positive / len(query),
        "query_minus_reference_fraction": (query_positive - reference_positive) / len(reference),
        "precision": true_positive / query_positive,
        "recall": true_positive / reference_positive,
        "dice": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barcode-mappings", type=Path, required=True)
    parser.add_argument("--cellpose-segmentation", type=Path, required=True)
    parser.add_argument("--tissue-positions", type=Path, required=True)
    parser.add_argument("--space-ranger-alignment-json", type=Path, required=True)
    parser.add_argument("--native-registration-json", type=Path, required=True)
    parser.add_argument(
        "--roi-manifest", type=Path,
        help="Frozen ROI manifest; omit to compare the complete registered field.",
    )
    parser.add_argument("--registration-moving-source-downsample", type=int, default=4)
    parser.add_argument(
        "--registration-moving-sampling", choices=("area", "decimate"), default="area",
    )
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.cellpose_segmentation / "summary.json"
    inputs = (
        args.barcode_mappings,
        summary_path,
        args.tissue_positions,
        args.space_ranger_alignment_json,
        args.native_registration_json,
    )
    if args.roi_manifest is not None:
        inputs += (args.roi_manifest,)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    if args.registration_moving_source_downsample < 1:
        raise SystemExit("registration moving downsample must be positive")

    native_json = json.loads(args.native_registration_json.read_text())
    moving_shape = tuple(int(value) for value in native_json["moving_shape_yx"])
    fixed_shape = tuple(int(value) for value in native_json["fixed_shape_yx"])
    roi_manifest = (
        json.loads(args.roi_manifest.read_text())
        if args.roi_manifest is not None else None
    )
    sr_transform = matrix_from_json(
        args.space_ranger_alignment_json, ("cytAssistInfo", "transformImages"),
    )
    native_transform = matrix_from_json(args.native_registration_json, ("moving_to_fixed",))
    indices, roi_audit = select_native_roi(
        args.tissue_positions,
        sr_transform,
        native_transform,
        roi_manifest,
        moving_shape,
        fixed_shape,
        width=args.width,
        height=args.height,
    )
    reference = load_roi_states(
        args.barcode_mappings, indices, width=args.width, height=args.height,
    )
    query, supported, projection_audit = load_cellpose_states(
        args.cellpose_segmentation,
        args.tissue_positions,
        indices,
        sr_transform,
        native_transform,
        roi_manifest,
        width=args.width,
        height=args.height,
        moving_downsample=args.registration_moving_source_downsample,
        moving_sampling=args.registration_moving_sampling,
    )
    reference = reference[supported]
    query = query[supported]
    rows = []
    for compartment, reference_mask, query_mask in (
        ("cell", reference > 0, query > 0),
        ("nucleus", reference == 2, query == 2),
    ):
        rows.append({"compartment": compartment} | binary_mask_metrics(reference_mask, query_mask))

    args.out_dir.mkdir(parents=True)
    table_path = args.out_dir / "mask_concordance.tsv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    segmentation = json.loads(summary_path.read_text())
    summary = {
        "schema": "visium_hd_processing.cellpose_space_ranger_mask_roi_comparison.v1",
        "role": "vendor_calibrated_compatibility_control",
        "disclosure": (
            "Space Ranger masks are used only as a morphology compatibility target; "
            "this comparison is not an independent biological validation."
        ),
        "cellpose_method": segmentation["method"],
        "counts": {
            "native_roi_bins": int(len(indices)),
            "supported_common_bins": int(np.count_nonzero(supported)),
            "unsupported_cellpose_bins": int(np.count_nonzero(~supported)),
        },
        "roi_audit": roi_audit,
        "projection_audit": projection_audit,
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {table_path.name: sha256(table_path)},
    }
    output_summary = args.out_dir / "summary.json"
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checksums = args.out_dir / "checksums.sha256"
    checksums.write_text(
        f"{sha256(table_path)}  {table_path.name}\n"
        f"{sha256(output_summary)}  {output_summary.name}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
