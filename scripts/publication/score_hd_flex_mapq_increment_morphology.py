#!/usr/bin/env python3
"""Score an existing Flex MAPQ-mode increment on frozen H&E Cellpose masks."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from compare_hd_flex_policy_matrices import load_space_ranger_sparse, mex_file
from score_hd_flex_registration_roi import (
    build_mex_axis,
    build_roi_geometry,
    load_mex_roi_sparse,
    matrix_field,
    score_field,
    sha256,
)


POLICY_DIRECTORIES = {
    "strict": "strict",
    "soft_expected": "soft_expected",
    "hard": "hard",
    "gated_hard": "gated_hard",
}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def split_sparse_delta(
    mapq_off: sparse.csr_matrix,
    genomic: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    delta = (mapq_off - genomic).tocsr()
    delta.sum_duplicates()
    positive = delta.copy()
    positive.data = np.clip(positive.data, 0.0, None)
    positive.eliminate_zeros()
    negative = delta.copy()
    negative.data = np.clip(-negative.data, 0.0, None)
    negative.eliminate_zeros()
    if not math.isclose(
        float(positive.sum() - negative.sum()),
        float(mapq_off.sum() - genomic.sum()),
        rel_tol=1e-12,
        abs_tol=1e-5,
    ):
        raise ValueError("positive/negative MAPQ delta does not reconcile")
    return positive, negative


def load_target_feature_list(path: Path) -> dict[str, int]:
    """Load a one-feature-ID-per-line target axis without expression values."""
    features = [
        line.strip().split("\t", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not features:
        raise ValueError(f"empty target feature list: {path}")
    if len(features) != len(set(features)):
        raise ValueError(f"duplicate target feature identifiers: {path}")
    return {name: index for index, name in enumerate(features)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genomic-policy-mex-root", type=Path, required=True)
    parser.add_argument("--mapq-off-policy-mex-root", type=Path, required=True)
    target_axis = parser.add_mutually_exclusive_group(required=True)
    target_axis.add_argument("--vendor-h5", type=Path)
    target_axis.add_argument(
        "--target-feature-list",
        type=Path,
        help=(
            "One target feature ID per line. Use when no Space Ranger raw H5 "
            "is available; --width supplies the declared square-grid width."
        ),
    )
    parser.add_argument("--cellpose-segmentation", type=Path, required=True)
    parser.add_argument("--capture-grid-json", type=Path, required=True)
    parser.add_argument("--native-registration-json", type=Path, required=True)
    parser.add_argument(
        "--registration-moving-source-downsample", type=int, default=16,
    )
    parser.add_argument(
        "--registration-moving-sampling",
        choices=("area", "decimate"),
        default="decimate",
    )
    parser.add_argument(
        "--policy", action="append", choices=tuple(POLICY_DIRECTORIES),
        default=None,
    )
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--parent-size", type=int, default=8)
    parser.add_argument("--slide", default="H1-GMHFWPH")
    parser.add_argument("--area", default="D1")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policies = args.policy or ["hard", "soft_expected"]
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    for path in (
        args.genomic_policy_mex_root,
        args.mapq_off_policy_mex_root,
        args.cellpose_segmentation,
    ):
        if not path.is_dir():
            raise SystemExit(f"missing input directory: {path}")
    for path in (
        args.vendor_h5 or args.target_feature_list,
        args.capture_grid_json,
        args.native_registration_json,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")

    geometry_args = argparse.Namespace(
        native_registration_json=args.native_registration_json,
        roi_manifest=None,
        capture_grid_json=args.capture_grid_json,
        tissue_positions=None,
        space_ranger_alignment_json=None,
        cellpose_segmentation=args.cellpose_segmentation,
        barcode_mappings=None,
        registration_moving_source_downsample=(
            args.registration_moving_source_downsample
        ),
        registration_moving_sampling=args.registration_moving_sampling,
        width=args.width,
        height=args.height,
        parent_size=args.parent_size,
    )
    geometry = build_roi_geometry(geometry_args)

    if args.vendor_h5:
        vendor, target_index, grid_width = load_space_ranger_sparse(
            args.vendor_h5, 2,
        )
        if grid_width != args.width:
            raise ValueError("vendor grid width differs from declared width")
        del vendor
        gc.collect()
        target_axis_input = {
            "kind": "space_ranger_raw_h5",
            "path": str(args.vendor_h5.resolve()),
            "sha256": sha256(args.vendor_h5),
        }
    else:
        target_index = load_target_feature_list(args.target_feature_list)
        target_axis_input = {
            "kind": "target_feature_list",
            "path": str(args.target_feature_list.resolve()),
            "sha256": sha256(args.target_feature_list),
            "declared_grid_width": args.width,
        }
    target_features = len(target_index)

    rows: list[dict[str, object]] = []
    reconciliation: list[dict[str, object]] = []
    input_matrices: dict[str, dict[str, list[dict[str, object]]]] = {}
    in_cell = geometry.states > 0
    in_nucleus = geometry.states == 2
    for policy in policies:
        directory = POLICY_DIRECTORIES[policy]
        roots = {
            "genomic": args.genomic_policy_mex_root / directory / "square_002um",
            "mapq_off": args.mapq_off_policy_mex_root / directory / "square_002um",
        }
        matrices: dict[str, sparse.csr_matrix] = {}
        input_matrices[policy] = {}
        for mode, root in roots.items():
            features, mapped_rows, mapped_columns = build_mex_axis(
                root, target_index, geometry.indices,
                width=args.width, height=args.height,
            )
            matrices[mode] = load_mex_roi_sparse(
                root, mapped_rows, mapped_columns,
                target_features=target_features,
                roi_bins=len(geometry.indices),
            )
            input_matrices[policy][mode] = [
                {
                    "path": str(mex_file(root, name).resolve()),
                    "sha256": sha256(mex_file(root, name)),
                }
                for name in ("features.tsv", "barcodes.tsv", "matrix.mtx")
            ]
            del features, mapped_rows, mapped_columns

        genomic = matrices["genomic"]
        mapq_off = matrices["mapq_off"]
        positive, negative = split_sparse_delta(mapq_off, genomic)
        fields = {
            "genomic_whole": matrix_field(genomic),
            "mapq_off_whole": matrix_field(mapq_off),
            "mapq_off_positive_feature_bin_increment": matrix_field(positive),
            "mapq_off_negative_feature_bin_redistribution": matrix_field(negative),
        }
        roles = {
            "genomic_whole": "whole_genomic_mode",
            "mapq_off_whole": "whole_mapq_off_mode",
            "mapq_off_positive_feature_bin_increment": "positive_mapq_increment",
            "mapq_off_negative_feature_bin_redistribution": "negative_mapq_redistribution",
        }
        for name, field in fields.items():
            row = score_field(
                f"{policy}_{name}", roles[name], policy, field, geometry,
            )
            row["scope"] = "registered_cellpose_supported_roi"
            rows.append(row)

        positive_field = fields["mapq_off_positive_feature_bin_increment"]
        negative_field = fields["mapq_off_negative_feature_bin_redistribution"]
        genomic_mass = float(genomic.sum())
        mapq_off_mass = float(mapq_off.sum())
        positive_mass = float(positive.sum())
        negative_mass = float(negative.sum())
        reconciliation.append({
            "policy": policy,
            "scope": "registered_cellpose_supported_roi",
            "genomic_mass": genomic_mass,
            "mapq_off_mass": mapq_off_mass,
            "positive_feature_bin_increment_mass": positive_mass,
            "negative_feature_bin_redistribution_mass": negative_mass,
            "net_increment_mass": mapq_off_mass - genomic_mass,
            "percent_net_increment_over_genomic": (
                100.0 * (mapq_off_mass - genomic_mass) / genomic_mass
            ),
            "positive_increment_in_cell_mass_fraction": (
                float(positive_field[in_cell].sum() / positive_mass)
                if positive_mass else 0.0
            ),
            "positive_increment_in_nucleus_mass_fraction": (
                float(positive_field[in_nucleus].sum() / positive_mass)
                if positive_mass else 0.0
            ),
            "negative_redistribution_in_cell_mass_fraction": (
                float(negative_field[in_cell].sum() / negative_mass)
                if negative_mass else 0.0
            ),
            "negative_redistribution_in_nucleus_mass_fraction": (
                float(negative_field[in_nucleus].sum() / negative_mass)
                if negative_mass else 0.0
            ),
        })
        del matrices, genomic, mapq_off, positive, negative, fields
        gc.collect()

    args.out_dir.mkdir(parents=True)
    fields_path = args.out_dir / "morphology_fields.tsv"
    reconciliation_path = args.out_dir / "delta_reconciliation.tsv"
    write_rows(fields_path, rows)
    write_rows(reconciliation_path, reconciliation)
    summary = {
        "schema": "visium_hd_processing.flex_mapq_increment_morphology.v1",
        "interpretation": (
            "Independent H&E morphology equivalence/localization diagnostic; "
            "raw mass is not normalized and small metric signs do not select a mode."
        ),
        "scope": (
            "Expression restricted to the frozen native-registration capture-grid "
            "ROI with Cellpose pixel support; not a whole-slide mass total."
        ),
        "policies": policies,
        "roi_geometry_audit": geometry.audit,
        "inputs": {
            "target_axis": target_axis_input,
            "cellpose_summary": {
                "path": str((args.cellpose_segmentation / "summary.json").resolve()),
                "sha256": sha256(args.cellpose_segmentation / "summary.json"),
            },
            "capture_grid": {
                "path": str(args.capture_grid_json.resolve()),
                "sha256": sha256(args.capture_grid_json),
            },
            "registration": {
                "path": str(args.native_registration_json.resolve()),
                "sha256": sha256(args.native_registration_json),
            },
            "matrices": input_matrices,
        },
        "outputs": {
            fields_path.name: {
                "bytes": fields_path.stat().st_size,
                "sha256": sha256(fields_path),
            },
            reconciliation_path.name: {
                "bytes": reconciliation_path.stat().st_size,
                "sha256": sha256(reconciliation_path),
            },
        },
        "invariants": {
            "raw_mass_not_normalized": True,
            "positive_and_negative_feature_bin_delta_reported_separately": True,
            "image_or_segmentation_not_used_for_assignment": True,
            "image_artifacts_are_frozen_before_expression_scoring": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (fields_path, reconciliation_path, summary_path)
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
