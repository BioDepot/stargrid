#!/usr/bin/env python3
"""Prepare compact STAR/Space-Ranger coarse-bin fields for H&E scoring."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from compare_hd_flex_h5ad_aggregates import (  # noqa: E402
    barcode_coordinate,
    correlation,
    cosine,
    load_h5ad_totals,
    load_mex_totals,
    normalized_tv,
    sha256,
)
from compare_hd_flex_policy_matrices import space_ranger_feature_indexes  # noqa: E402


def parse_method(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=MEX_DIR")
    return label, Path(raw_path)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def load_space_ranger_h5_totals(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, int]:
    """Load target-gene and barcode totals from a 10x CSC count H5."""
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        feature_group = matrix["features"]
        identifiers = [value.split(".", 1)[0] for value in decode(feature_group["id"][:])]
        target_indexes = np.asarray(
            sorted(int(value) for value in space_ranger_feature_indexes(feature_group)),
            dtype=np.int64,
        )
        target_features = [identifiers[index] for index in target_indexes]
        if len(target_features) != len(set(target_features)):
            raise ValueError("duplicate Space Ranger target gene identifiers")
        barcodes = decode(matrix["barcodes"][:])
        if len(barcodes) != len(set(barcodes)):
            raise ValueError("duplicate Space Ranger barcodes")
        shape = tuple(int(value) for value in matrix["shape"][:])
        if shape != (len(identifiers), len(barcodes)):
            raise ValueError("Space Ranger H5 shape disagrees with its axes")
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        if len(indptr) != len(barcodes) + 1:
            raise ValueError("invalid Space Ranger H5 column pointers")
        compact = np.full(len(identifiers), -1, dtype=np.int32)
        compact[target_indexes] = np.arange(len(target_indexes), dtype=np.int32)
        gene_totals = np.zeros(len(target_indexes), dtype=np.float64)
        bin_totals = np.zeros(len(barcodes), dtype=np.float64)
        target_nnz = 0
        for column_begin in range(0, len(barcodes), 50_000):
            column_end = min(len(barcodes), column_begin + 50_000)
            data_begin = int(indptr[column_begin])
            data_end = int(indptr[column_end])
            data = np.asarray(matrix["data"][data_begin:data_end], dtype=np.float64)
            indices = np.asarray(matrix["indices"][data_begin:data_end], dtype=np.int64)
            if np.any(~np.isfinite(data)) or np.any(data < 0):
                raise ValueError("invalid Space Ranger H5 values")
            local_features = compact[indices]
            selected = local_features >= 0
            target_nnz += int(np.count_nonzero(selected & (data != 0)))
            if np.any(selected):
                gene_totals += np.bincount(
                    local_features[selected],
                    weights=data[selected],
                    minlength=len(target_indexes),
                )
            weights = np.where(selected, data, 0.0)
            prefix = np.empty(len(weights) + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(weights, out=prefix[1:])
            offsets = indptr[column_begin : column_end + 1] - data_begin
            bin_totals[column_begin:column_end] = (
                prefix[offsets[1:]] - prefix[offsets[:-1]]
            )
    return target_features, barcodes, gene_totals, bin_totals, target_nnz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--space-ranger-h5", type=Path)
    source.add_argument("--space-ranger-h5ad", type=Path)
    parser.add_argument(
        "--star-method", action="append", type=parse_method, required=True,
        metavar="LABEL=MEX_DIR",
    )
    parser.add_argument("--scale-um", type=int, default=8)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    labels = [label for label, _ in args.star_method]
    if len(labels) != len(set(labels)):
        raise SystemExit("duplicate --star-method label")
    sr_path = args.space_ranger_h5 or args.space_ranger_h5ad
    if not sr_path.is_file():
        raise SystemExit(f"missing Space Ranger input: {sr_path}")
    for _, root in args.star_method:
        if not root.is_dir():
            raise SystemExit(f"missing STAR MEX directory: {root}")

    if args.space_ranger_h5:
        sr_features, sr_barcodes, sr_gene, sr_bins, sr_nnz = (
            load_space_ranger_h5_totals(args.space_ranger_h5)
        )
        sr_kind = "space_ranger_target_gene_h5"
    else:
        sr_features, sr_barcodes, sr_gene, sr_bins, sr_nnz = load_h5ad_totals(
            args.space_ranger_h5ad, row_chunk=50_000,
        )
        sr_kind = "published_space_ranger_filtered_h5ad"
    sr_coordinates = np.asarray([
        barcode_coordinate(barcode, scale=args.scale_um) for barcode in sr_barcodes
    ], dtype=np.int32)
    sr_feature_index = {feature: index for index, feature in enumerate(sr_features)}
    sr_axis_index = {
        tuple(int(value) for value in coordinate): index
        for index, coordinate in enumerate(sr_coordinates)
    }
    arrays: dict[str, np.ndarray] = {
        "space_ranger_coordinates": sr_coordinates,
        "space_ranger_bin_totals": sr_bins,
    }
    methods: dict[str, object] = {}
    for label, root in args.star_method:
        features, barcodes, gene, bins, nnz, input_paths = load_mex_totals(
            root, scale=args.scale_um,
        )
        unexpected = sorted(set(features) - set(sr_feature_index))
        if unexpected:
            raise ValueError(f"{len(unexpected)} STAR features absent from Space Ranger axis")
        missing = sorted(set(sr_features) - set(features))
        missing_mass = float(sum(sr_gene[sr_feature_index[name]] for name in missing))
        star_gene_on_sr = np.zeros(len(sr_features), dtype=np.float64)
        for feature, value in zip(features, gene, strict=True):
            star_gene_on_sr[sr_feature_index[feature]] += value
        star_coordinates = np.asarray([
            barcode_coordinate(barcode, scale=args.scale_um) for barcode in barcodes
        ], dtype=np.int32)
        star_on_sr = np.zeros(len(sr_barcodes), dtype=np.float64)
        outside_mass = 0.0
        outside_barcodes = 0
        for coordinate, value in zip(star_coordinates, bins, strict=True):
            index = sr_axis_index.get(tuple(int(item) for item in coordinate))
            if index is None:
                outside_mass += float(value)
                outside_barcodes += 1
            else:
                star_on_sr[index] += value
        star_total = float(np.sum(bins, dtype=np.float64))
        star_common = float(np.sum(star_on_sr, dtype=np.float64))
        if not math.isclose(
            star_total, star_common + outside_mass, rel_tol=1e-12, abs_tol=1e-5,
        ):
            raise ValueError("STAR common/outside barcode mass does not reconcile")
        shared = np.minimum(star_on_sr, sr_bins)
        star_only = np.clip(star_on_sr - sr_bins, 0.0, None)
        sr_only = np.clip(sr_bins - star_on_sr, 0.0, None)
        positive_mass = float(np.sum(star_only, dtype=np.float64))
        negative_mass = float(np.sum(sr_only, dtype=np.float64))
        sr_mass = float(np.sum(sr_bins, dtype=np.float64))
        common_detected = (star_gene_on_sr > 0) & (sr_gene > 0)
        methods[label] = {
            "inputs": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in input_paths
            ],
            "star_full_mass": star_total,
            "star_mass_on_space_ranger_axis": star_common,
            "star_mass_outside_space_ranger_axis": outside_mass,
            "star_barcodes_outside_space_ranger_axis": outside_barcodes,
            "space_ranger_mass": sr_mass,
            "signed_mass_difference_on_common_axis": star_common - sr_mass,
            "percent_mass_difference_over_space_ranger_on_common_axis": (
                100.0 * (star_common - sr_mass) / sr_mass if sr_mass else 0.0
            ),
            "shared_spatial_bin_mass": float(np.sum(shared, dtype=np.float64)),
            "star_positive_spatial_bin_residual_mass": positive_mass,
            "space_ranger_positive_spatial_bin_residual_mass": negative_mass,
            "star_to_space_ranger_positive_spatial_residual_mass_ratio": (
                positive_mass / negative_mass if negative_mass else None
            ),
            "space_ranger_features_absent_from_star": len(missing),
            "space_ranger_mass_on_features_absent_from_star": missing_mass,
            "gene_total_pearson": correlation(star_gene_on_sr, sr_gene),
            "gene_log1p_pearson": correlation(
                np.log1p(star_gene_on_sr), np.log1p(sr_gene),
            ),
            "gene_total_spearman_common_detected": correlation(
                star_gene_on_sr[common_detected], sr_gene[common_detected], ranked=True,
            ),
            "spatial_bin_total_pearson_on_common_axis": correlation(star_on_sr, sr_bins),
            "spatial_bin_log1p_pearson_on_common_axis": correlation(
                np.log1p(star_on_sr), np.log1p(sr_bins),
            ),
            "spatial_bin_spearman_on_common_axis": correlation(
                star_on_sr, sr_bins, ranked=True,
            ),
            "spatial_bin_cosine_on_common_axis": cosine(star_on_sr, sr_bins),
            "spatial_bin_normalized_total_variation_on_common_axis": normalized_tv(
                star_on_sr, sr_bins,
            ),
            "star_nnz": nnz,
            "space_ranger_nnz": sr_nnz,
        }
        arrays[f"{label}_star_common_bin_totals"] = star_on_sr
        arrays[f"{label}_star_full_coordinates"] = star_coordinates
        arrays[f"{label}_star_full_bin_totals"] = bins
        del gene, bins, star_gene_on_sr, star_on_sr, shared, star_only, sr_only
        gc.collect()

    args.out_dir.mkdir(parents=True)
    fields_path = args.out_dir / "coarse_fields.npz"
    np.savez_compressed(fields_path, **arrays)
    summary = {
        "schema": "visium_hd_processing.flex_star_sr_prepared_fields.v1",
        "dataset": args.dataset,
        "comparison_scale_um": args.scale_um,
        "space_ranger": {
            "kind": sr_kind,
            "path": str(sr_path.resolve()),
            "bytes": sr_path.stat().st_size,
            "sha256": sha256(sr_path),
            "features": len(sr_features),
            "barcodes": len(sr_barcodes),
            "nnz": sr_nnz,
        },
        "methods": methods,
        "outputs": {
            fields_path.name: {
                "bytes": fields_path.stat().st_size,
                "sha256": sha256(fields_path),
            }
        },
        "invariants": {
            "biological_feature_axis_only": True,
            "diagnostic_non_panel_counts_excluded": True,
            "raw_mass_not_normalized": True,
            "space_ranger_barcode_axis_is_primary": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        f"{sha256(fields_path)}  {fields_path.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
