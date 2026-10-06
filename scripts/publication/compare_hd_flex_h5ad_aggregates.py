#!/usr/bin/env python3
"""Compare STAR Flex MEX gene/bin totals with a retained Space Ranger H5AD."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import mmread
from scipy.stats import rankdata


BARCODE = re.compile(r"^s_(\d{3})um_(\d+)_(\d+)(?:-\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def open_text(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8", newline="")


def barcode_coordinate(barcode: str, *, scale: int) -> tuple[int, int]:
    """Return a formatting-independent Visium HD array coordinate."""
    match = BARCODE.fullmatch(barcode)
    if match is None or int(match.group(1)) != scale:
        raise ValueError(f"invalid {scale} um barcode: {barcode}")
    return int(match.group(2)), int(match.group(3))


def mex_file(root: Path, name: str) -> Path:
    for candidate in (root / name, root / f"{name}.gz"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing MEX component {name}[.gz] in {root}")


def correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool = False) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("correlation arrays must have the same nontrivial shape")
    if ranked:
        left = rankdata(left, method="average")
        right = rankdata(right, method="average")
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator:
        return float(np.dot(left, right) / denominator)
    return 1.0 if np.array_equal(left, right) else 0.0


def normalized_tv(left: np.ndarray, right: np.ndarray) -> float | None:
    left_mass = float(np.sum(left, dtype=np.float64))
    right_mass = float(np.sum(right, dtype=np.float64))
    if not left_mass or not right_mass:
        return None
    return 0.5 * float(
        np.sum(np.abs(left / left_mass - right / right_mass), dtype=np.float64)
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator:
        return float(np.dot(left, right) / denominator)
    return 1.0 if np.array_equal(left, right) else 0.0


def load_h5ad_totals(
    path: Path, *, row_chunk: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, int]:
    """Load H5AD axes and aggregate CSR X without materializing the matrix."""
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        encoding = x.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding != "csr_matrix":
            raise ValueError(f"expected CSR H5AD X, observed {encoding!r}")
        barcodes = decode(handle["obs/_index"][:])
        features = [
            value.split(".", 1)[0] for value in decode(handle["var/gene_ids"][:])
        ]
        if len(features) != len(set(features)) or len(barcodes) != len(set(barcodes)):
            raise ValueError("duplicate H5AD feature or barcode axis")
        shape = tuple(int(value) for value in x.attrs["shape"])
        if shape != (len(barcodes), len(features)):
            raise ValueError(f"H5AD X shape disagrees with axes: {shape}")
        indptr = np.asarray(x["indptr"][:], dtype=np.int64)
        if len(indptr) != len(barcodes) + 1 or int(indptr[-1]) != len(x["data"]):
            raise ValueError("invalid H5AD CSR pointers")
        gene_totals = np.zeros(len(features), dtype=np.float64)
        bin_totals = np.zeros(len(barcodes), dtype=np.float64)
        for row_begin in range(0, len(barcodes), row_chunk):
            row_end = min(len(barcodes), row_begin + row_chunk)
            data_begin, data_end = int(indptr[row_begin]), int(indptr[row_end])
            data = np.asarray(x["data"][data_begin:data_end], dtype=np.float64)
            indices = np.asarray(x["indices"][data_begin:data_end], dtype=np.int64)
            if np.any(~np.isfinite(data)) or np.any(data < 0):
                raise ValueError("invalid H5AD X values")
            gene_totals += np.bincount(indices, weights=data, minlength=len(features))
            prefix = np.empty(len(data) + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(data, out=prefix[1:])
            offsets = indptr[row_begin : row_end + 1] - data_begin
            bin_totals[row_begin:row_end] = prefix[offsets[1:]] - prefix[offsets[:-1]]
        nnz = int(indptr[-1])
    return features, barcodes, gene_totals, bin_totals, nnz


def load_mex_totals(
    root: Path, *, scale: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, int, list[Path]]:
    feature_path = mex_file(root, "features.tsv")
    barcode_path = mex_file(root, "barcodes.tsv")
    matrix_path = mex_file(root, "matrix.mtx")
    with open_text(feature_path) as handle:
        features = [
            line.rstrip("\n").split("\t", 1)[0].split(".", 1)[0]
            for line in handle
        ]
    with open_text(barcode_path) as handle:
        barcodes = [line.rstrip("\n").split("\t", 1)[0] for line in handle]
    if len(features) != len(set(features)) or len(barcodes) != len(set(barcodes)):
        raise ValueError(f"duplicate MEX feature or barcode axis in {root}")
    coordinates = [barcode_coordinate(barcode, scale=scale) for barcode in barcodes]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"duplicate MEX barcode coordinates in {root}")
    matrix = mmread(str(matrix_path)).tocoo(copy=False)
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"MEX matrix shape disagrees with axes in {root}")
    data = np.asarray(matrix.data, dtype=np.float64)
    if np.any(~np.isfinite(data)) or np.any(data < 0):
        raise ValueError(f"invalid MEX values in {root}")
    gene_totals = np.bincount(
        np.asarray(matrix.row, dtype=np.int64), weights=data, minlength=len(features)
    )
    bin_totals = np.bincount(
        np.asarray(matrix.col, dtype=np.int64), weights=data, minlength=len(barcodes)
    )
    return (
        features,
        barcodes,
        gene_totals,
        bin_totals,
        int(matrix.nnz),
        [feature_path, barcode_path, matrix_path],
    )


def compare_method(
    label: str,
    root: Path,
    scale: int,
    sr_features: list[str],
    sr_barcodes: list[str],
    sr_gene_totals: np.ndarray,
    sr_bin_totals: np.ndarray,
) -> tuple[dict[str, object], list[Path]]:
    features, barcodes, gene_totals, bin_totals, nnz, inputs = load_mex_totals(
        root, scale=scale,
    )
    sr_feature_index = {feature: index for index, feature in enumerate(sr_features)}
    unexpected = sorted(set(features) - set(sr_feature_index))
    if unexpected:
        raise ValueError(f"{len(unexpected)} STAR features absent from H5AD")
    star_gene_on_sr = np.zeros(len(sr_features), dtype=np.float64)
    for index, feature in enumerate(features):
        star_gene_on_sr[sr_feature_index[feature]] += gene_totals[index]

    sr_coordinates = [
        barcode_coordinate(barcode, scale=scale) for barcode in sr_barcodes
    ]
    if len(sr_coordinates) != len(set(sr_coordinates)):
        raise ValueError("duplicate H5AD barcode coordinates")
    sr_barcode_index = {
        coordinate: index for index, coordinate in enumerate(sr_coordinates)
    }
    star_bin_on_sr = np.zeros(len(sr_barcodes), dtype=np.float64)
    outside_mass = 0.0
    outside_barcodes = 0
    for barcode, value in zip(barcodes, bin_totals):
        sr_index = sr_barcode_index.get(barcode_coordinate(barcode, scale=scale))
        if sr_index is None:
            outside_mass += float(value)
            outside_barcodes += 1
        else:
            star_bin_on_sr[sr_index] += value

    star_mass = float(np.sum(gene_totals, dtype=np.float64))
    sr_mass = float(np.sum(sr_gene_totals, dtype=np.float64))
    shared_star_mass = float(np.sum(star_bin_on_sr, dtype=np.float64))
    common_detected = (star_gene_on_sr > 0) & (sr_gene_totals > 0)
    star_signal = star_bin_on_sr > 0
    sr_signal = sr_bin_totals > 0
    top_count = min(
        100, np.count_nonzero(star_gene_on_sr), np.count_nonzero(sr_gene_totals)
    )
    star_top = set(np.argsort(star_gene_on_sr)[-top_count:])
    sr_top = set(np.argsort(sr_gene_totals)[-top_count:])
    missing_features = sorted(set(sr_features) - set(features))
    missing_indexes = [sr_feature_index[feature] for feature in missing_features]
    missing_mass = (
        float(np.sum(sr_gene_totals[missing_indexes], dtype=np.float64))
        if missing_indexes else 0.0
    )
    row = {
        "method": label,
        "scale_um": scale,
        "star_features": len(features),
        "space_ranger_features": len(sr_features),
        "space_ranger_features_absent_from_star": len(missing_features),
        "absent_feature_space_ranger_mass": missing_mass,
        "star_barcodes": len(barcodes),
        "space_ranger_barcodes": len(sr_barcodes),
        "star_barcodes_outside_space_ranger_object": outside_barcodes,
        "star_nnz": nnz,
        "star_raw_mass": star_mass,
        "space_ranger_raw_mass": sr_mass,
        "signed_mass_difference": star_mass - sr_mass,
        "mass_ratio": star_mass / sr_mass if sr_mass else None,
        "star_mass_on_space_ranger_barcodes": shared_star_mass,
        "star_mass_outside_space_ranger_barcodes": outside_mass,
        "star_mass_fraction_on_space_ranger_barcodes": (
            shared_star_mass / star_mass if star_mass else None
        ),
        "gene_total_pearson": correlation(star_gene_on_sr, sr_gene_totals),
        "gene_log1p_pearson": correlation(
            np.log1p(star_gene_on_sr), np.log1p(sr_gene_totals)
        ),
        "gene_total_spearman_common_detected": correlation(
            star_gene_on_sr[common_detected],
            sr_gene_totals[common_detected],
            ranked=True,
        ),
        "gene_normalized_total_variation": normalized_tv(
            star_gene_on_sr, sr_gene_totals
        ),
        "top100_gene_overlap": len(star_top & sr_top),
        "space_ranger_barcodes_with_star_signal": int(np.count_nonzero(star_signal)),
        "space_ranger_barcode_signal_coverage": float(np.mean(star_signal)),
        "spatial_bin_total_pearson_on_space_ranger_axis": correlation(
            star_bin_on_sr, sr_bin_totals
        ),
        "spatial_bin_log1p_pearson_on_space_ranger_axis": correlation(
            np.log1p(star_bin_on_sr), np.log1p(sr_bin_totals)
        ),
        "spatial_bin_spearman_on_space_ranger_axis": correlation(
            star_bin_on_sr, sr_bin_totals, ranked=True
        ),
        "spatial_bin_cosine_on_space_ranger_axis": cosine(
            star_bin_on_sr, sr_bin_totals
        ),
        "spatial_bin_normalized_total_variation_on_space_ranger_axis": (
            normalized_tv(star_bin_on_sr, sr_bin_totals)
        ),
        "space_ranger_nonzero_barcodes": int(np.count_nonzero(sr_signal)),
    }
    return row, inputs


def parse_methods(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    labels = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in labels:
            raise ValueError(f"expected unique LABEL=PATH for --method: {value}")
        labels.add(label)
        result.append((label, Path(raw_path)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-h5ad", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, metavar="LABEL=MEX_DIR")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--umi-mode", default="1mm_cr")
    parser.add_argument("--h5ad-row-chunk", type=int, default=50_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    methods = parse_methods(args.method)
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    if args.scale < 1 or args.h5ad_row_chunk < 1:
        parser.error("--scale and --h5ad-row-chunk must be positive")
    if not args.vendor_h5ad.is_file():
        raise SystemExit(f"missing input: {args.vendor_h5ad}")

    sr_features, sr_barcodes, sr_gene_totals, sr_bin_totals, sr_nnz = (
        load_h5ad_totals(args.vendor_h5ad, row_chunk=args.h5ad_row_chunk)
    )
    rows = []
    method_inputs = {}
    for label, root in methods:
        row, inputs = compare_method(
            label,
            root,
            args.scale,
            sr_features,
            sr_barcodes,
            sr_gene_totals,
            sr_bin_totals,
        )
        rows.append({"umi_mode": args.umi_mode, **row})
        method_inputs[label] = [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ]

    args.out_dir.mkdir(parents=True)
    table = args.out_dir / "aggregate_concordance.tsv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "visium_hd_processing.flex_h5ad_aggregate_concordance.v1",
        "comparison_axis": {
            "gene": "union of H5AD target genes and STAR features",
            "spatial": "retained Space Ranger H5AD barcode axis",
        },
        "interpretation": "Space Ranger is a compatibility comparator, not truth.",
        "umi_mode": args.umi_mode,
        "scale_um": args.scale,
        "space_ranger": {
            "path": str(args.vendor_h5ad.resolve()),
            "bytes": args.vendor_h5ad.stat().st_size,
            "sha256": sha256(args.vendor_h5ad),
            "features": len(sr_features),
            "barcodes": len(sr_barcodes),
            "nnz": sr_nnz,
        },
        "methods": method_inputs,
        "output": {
            "path": str(table.resolve()),
            "bytes": table.stat().st_size,
            "sha256": sha256(table),
        },
        "invariants": {
            "raw_mass_not_normalized": True,
            "space_ranger_not_used_for_star_assignment": True,
            "mass_outside_retained_h5ad_barcode_axis_reported": True,
            "normalized_tv_is_secondary_shape_metric": True,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
