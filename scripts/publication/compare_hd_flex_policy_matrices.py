#!/usr/bin/env python3
"""Compare open Flex policy MEX matrices with Space Ranger on the target axis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import io as scipy_io
from scipy import sparse
from scipy.stats import rankdata


POLICIES = (
    ("strict", "strict"),
    ("postcollapse_soft", "soft_expected"),
    ("postcollapse_hard", "hard"),
    ("gated_hard", "gated_hard"),
)
SCALES = (2, 8, 16)
BARCODE = re.compile(r"^s_(\d{3})um_(\d+)_(\d+)(?:-\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_barcode(value: str, scale: int) -> str:
    match = BARCODE.fullmatch(value)
    if match is None or int(match.group(1)) != scale:
        raise ValueError(f"invalid {scale} um barcode: {value}")
    return f"s_{scale:03d}um_{int(match.group(2))}_{int(match.group(3))}"


def open_text(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8", newline="")


def mex_file(root: Path, name: str) -> Path:
    compressed = root / f"{name}.gz"
    plain = root / name
    if compressed.is_file():
        return compressed
    if plain.is_file():
        return plain
    raise FileNotFoundError(f"missing MEX component {name}[.gz] in {root}")


def read_lines(path: Path) -> list[str]:
    with open_text(path) as handle:
        return [line.rstrip("\n") for line in handle]


def load_mex(root: Path, scale: int) -> dict[tuple[str, str], float]:
    features = [
        line.split("\t", 1)[0]
        for line in read_lines(mex_file(root, "features.tsv"))
    ]
    barcodes = [
        canonical_barcode(line.split("\t", 1)[0], scale)
        for line in read_lines(mex_file(root, "barcodes.tsv"))
    ]
    if len(features) != len(set(features)) or len(barcodes) != len(set(barcodes)):
        raise ValueError(f"duplicate MEX axes in {root}")
    with open_text(mex_file(root, "matrix.mtx")) as handle:
        lines = (line.strip() for line in handle)
        header = next(lines, "")
        if not header.startswith("%%MatrixMarket matrix coordinate "):
            raise ValueError(f"invalid Matrix Market header in {root}")
        dimensions = next((line for line in lines if line and not line.startswith("%")), "")
        try:
            feature_count, barcode_count, entry_count = map(int, dimensions.split())
        except ValueError as exc:
            raise ValueError(f"invalid Matrix Market dimensions in {root}") from exc
        if (feature_count, barcode_count) != (len(features), len(barcodes)):
            raise ValueError(f"MEX axes disagree with matrix dimensions in {root}")
        result: dict[tuple[str, str], float] = {}
        observed = 0
        for line in lines:
            if not line:
                continue
            row, column, raw_value = line.split()
            key = (features[int(row) - 1], barcodes[int(column) - 1])
            if key in result:
                raise ValueError(f"duplicate MEX entry in {root}: {key}")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid MEX value in {root}: {raw_value}")
            if value:
                result[key] = value
            observed += 1
        if observed != entry_count:
            raise ValueError(f"MEX entry count disagrees with header in {root}")
    return result


def space_ranger_feature_indexes(features: object) -> np.ndarray:
    """Select the assay feature axis without assuming a Flex target set."""
    if "target_sets" in features:
        target_sets = features["target_sets"]
        if len(target_sets) != 1:
            raise ValueError("expected exactly one Space Ranger target set")
        indexes = np.asarray(next(iter(target_sets.values()))[:], dtype=np.int64)
    else:
        if "feature_type" not in features:
            raise ValueError("Space Ranger features lack target_sets and feature_type")
        feature_types = np.asarray(features["feature_type"][:])
        indexes = np.flatnonzero(feature_types == b"Gene Expression").astype(np.int64)
        if not len(indexes):
            raise ValueError("Space Ranger H5 has no Gene Expression features")
    if len(np.unique(indexes)) != len(indexes):
        raise ValueError("duplicate Space Ranger feature indexes")
    return indexes


def load_space_ranger_h5(path: Path, scale: int) -> tuple[dict[tuple[str, str], float], set[str]]:
    import h5py

    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        features = matrix["features"]
        identifiers = [value.decode() for value in features["id"][:]]
        target_indexes = {int(value) for value in space_ranger_feature_indexes(features)}
        target_genes = {identifiers[index] for index in target_indexes}
        barcodes = [canonical_barcode(value.decode(), scale) for value in matrix["barcodes"][:]]
        shape = tuple(int(value) for value in matrix["shape"][:])
        if shape != (len(identifiers), len(barcodes)):
            raise ValueError(f"Space Ranger H5 shape disagrees with axes: {path}")
        indptr = matrix["indptr"][:]
        indices = matrix["indices"][:]
        data = matrix["data"][:]
        result: dict[tuple[str, str], float] = {}
        for column, barcode in enumerate(barcodes):
            for offset in range(int(indptr[column]), int(indptr[column + 1])):
                feature_index = int(indices[offset])
                if feature_index not in target_indexes:
                    continue
                value = float(data[offset])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid Space Ranger H5 value: {value}")
                if not value:
                    continue
                key = (identifiers[feature_index], barcode)
                result[key] = result.get(key, 0.0) + value
    return result, target_genes


def pearson(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 1.0
    x = [left.get(key, 0.0) for key in keys]
    y = [right.get(key, 0.0) for key in keys]
    mean_x = math.fsum(x) / len(x)
    mean_y = math.fsum(y) / len(y)
    covariance = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    variance_x = math.fsum((a - mean_x) ** 2 for a in x)
    variance_y = math.fsum((b - mean_y) ** 2 for b in y)
    denominator = math.sqrt(variance_x * variance_y)
    if denominator:
        return covariance / denominator
    return 1.0 if x == y else 0.0


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[ordered[position]] = rank
        start = end
    return result


def spearman(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 1.0
    x = ranks([left.get(key, 0.0) for key in keys])
    y = ranks([right.get(key, 0.0) for key in keys])
    return pearson(dict(enumerate(x)), dict(enumerate(y)))


def aggregate(values: dict[tuple[str, str], float], axis: int) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for key in sorted(values):
        totals[key[axis]].append(values[key])
    return {key: math.fsum(parts) for key, parts in sorted(totals.items())}


def compare(
    open_values: dict[tuple[str, str], float],
    vendor_values: dict[tuple[str, str], float],
) -> dict[str, object]:
    keys = sorted(set(open_values) | set(vendor_values))
    open_mass = math.fsum(open_values.values())
    vendor_mass = math.fsum(vendor_values.values())
    pairs = [(open_values.get(key, 0.0), vendor_values.get(key, 0.0)) for key in keys]
    dot = math.fsum(left * right for left, right in pairs)
    norm = math.sqrt(
        math.fsum(left * left for left, _ in pairs)
        * math.fsum(right * right for _, right in pairs)
    )
    tolerance = 1e-12
    open_genes, vendor_genes = aggregate(open_values, 0), aggregate(vendor_values, 0)
    open_bins, vendor_bins = aggregate(open_values, 1), aggregate(vendor_values, 1)
    return {
        "open_raw_mass": open_mass,
        "space_ranger_raw_mass": vendor_mass,
        "signed_mass_difference": open_mass - vendor_mass,
        "absolute_mass_difference": abs(open_mass - vendor_mass),
        "mass_ratio": open_mass / vendor_mass if vendor_mass else None,
        "union_nonzero_entries": len(keys),
        "union_axis_l1": math.fsum(abs(left - right) for left, right in pairs),
        "exact_entries": sum(abs(left - right) <= tolerance for left, right in pairs),
        "exact_entry_fraction": (
            sum(abs(left - right) <= tolerance for left, right in pairs) / len(keys)
            if keys else 1.0
        ),
        "gene_total_pearson": pearson(open_genes, vendor_genes),
        "gene_total_spearman": spearman(open_genes, vendor_genes),
        "bin_total_pearson": pearson(open_bins, vendor_bins),
        "bin_total_spearman": spearman(open_bins, vendor_bins),
        "cosine_similarity": dot / norm if norm else (1.0 if not keys else 0.0),
        "normalized_total_variation": 0.5 * math.fsum(
            abs(
                (left / open_mass if open_mass else 0.0)
                - (right / vendor_mass if vendor_mass else 0.0)
            )
            for left, right in pairs
        ),
    }


def load_space_ranger_sparse(
    path: Path, scale: int,
) -> tuple[sparse.csr_matrix, dict[str, int], int]:
    """Load the target portion of a Space Ranger CSC matrix without Python rows."""
    import h5py

    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        feature_group = matrix["features"]
        identifiers = [value.decode() for value in feature_group["id"][:]]
        target_indexes = space_ranger_feature_indexes(feature_group)
        target_names = [identifiers[int(index)] for index in target_indexes]
        if len(target_names) != len(set(target_names)):
            raise ValueError("duplicate Space Ranger target feature identifiers")
        shape = tuple(int(value) for value in matrix["shape"][:])
        barcode_count = shape[1]
        grid_width = math.isqrt(barcode_count)
        if grid_width * grid_width != barcode_count:
            raise ValueError(f"Space Ranger barcode axis is not a square grid: {path}")
        barcodes = matrix["barcodes"]
        expected_first = f"s_{scale:03d}um_00000_00000-1".encode()
        expected_last = (
            f"s_{scale:03d}um_{grid_width - 1:05d}_{grid_width - 1:05d}-1".encode()
        )
        if len(barcodes) != barcode_count or barcodes[0] != expected_first or barcodes[-1] != expected_last:
            raise ValueError(f"Space Ranger barcode axis is not the declared row-major grid: {path}")
        values = np.asarray(matrix["data"][:], dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"invalid Space Ranger H5 values: {path}")
        full = sparse.csc_matrix(
            (
                values,
                np.asarray(matrix["indices"][:], dtype=np.int32),
                np.asarray(matrix["indptr"][:], dtype=np.int64),
            ),
            shape=shape,
        )
    full.sum_duplicates()
    full.eliminate_zeros()
    target = full[target_indexes, :].tocsr()
    return target, {name: index for index, name in enumerate(target_names)}, grid_width


def load_mex_sparse(
    root: Path,
    scale: int,
    target_index: dict[str, int],
    grid_width: int,
    expected_features: list[str] | None = None,
) -> sparse.csr_matrix:
    """Load MEX and map its subset axes directly onto the Space Ranger grid."""
    features = [
        line.split("\t", 1)[0]
        for line in read_lines(mex_file(root, "features.tsv"))
    ]
    if len(features) != len(set(features)):
        raise ValueError(f"duplicate MEX feature axis in {root}")
    if expected_features is not None and features != expected_features:
        raise ValueError(f"open policy feature axis differs in {root}")
    unexpected = sorted(set(features) - set(target_index))
    if unexpected:
        raise ValueError(f"open matrix contains {len(unexpected)} off-target features")

    raw = scipy_io.mmread(str(mex_file(root, "matrix.mtx"))).tocoo()
    if raw.shape[0] != len(features):
        raise ValueError(f"MEX feature axis disagrees with matrix dimensions in {root}")
    if np.any(~np.isfinite(raw.data)) or np.any(raw.data < 0.0):
        raise ValueError(f"invalid MEX values in {root}")
    canonical = raw.tocsr()
    if canonical.nnz != raw.nnz:
        raise ValueError(f"duplicate MEX entry in {root}")
    raw = canonical.tocoo()

    mapped_columns = np.empty(raw.shape[1], dtype=np.int64)
    observed_barcodes = 0
    with open_text(mex_file(root, "barcodes.tsv")) as handle:
        for observed_barcodes, line in enumerate(handle, start=1):
            if observed_barcodes > len(mapped_columns):
                raise ValueError(f"MEX barcode axis exceeds matrix dimensions in {root}")
            value = line.rstrip("\n").split("\t", 1)[0]
            match = BARCODE.fullmatch(value)
            if match is None or int(match.group(1)) != scale:
                raise ValueError(f"invalid {scale} um barcode: {value}")
            row, column = int(match.group(2)), int(match.group(3))
            if row >= grid_width or column >= grid_width:
                raise ValueError(f"MEX barcode is outside the Space Ranger grid: {value}")
            mapped_columns[observed_barcodes - 1] = row * grid_width + column
    if observed_barcodes != len(mapped_columns):
        raise ValueError(f"MEX barcode axis disagrees with matrix dimensions in {root}")
    if len(np.unique(mapped_columns)) != len(mapped_columns):
        raise ValueError(f"duplicate MEX barcode axis in {root}")

    mapped_rows = np.asarray([target_index[name] for name in features], dtype=np.int32)
    positive = raw.data != 0.0
    aligned = sparse.coo_matrix(
        (
            np.asarray(raw.data[positive], dtype=np.float64),
            (mapped_rows[raw.row[positive]], mapped_columns[raw.col[positive]]),
        ),
        shape=(len(target_index), grid_width * grid_width),
    ).tocsr()
    aligned.sum_duplicates()
    aligned.eliminate_zeros()
    return aligned


def _dense_correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool) -> float:
    present = (left != 0.0) | (right != 0.0)
    if not np.any(present):
        return 1.0
    x = left[present]
    y = right[present]
    if ranked:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)))
    if denominator:
        return float(np.dot(centered_x, centered_y) / denominator)
    return 1.0 if np.array_equal(x, y) else 0.0


def compare_sparse(
    open_values: sparse.csr_matrix,
    vendor_values: sparse.csr_matrix,
) -> dict[str, object]:
    """Sparse equivalent of compare(), bounded by matrix NNZ rather than Python objects."""
    if open_values.shape != vendor_values.shape:
        raise ValueError("open and Space Ranger sparse axes differ")
    open_mass = float(np.sum(open_values.data, dtype=np.float64))
    vendor_mass = float(np.sum(vendor_values.data, dtype=np.float64))
    product = open_values.multiply(vendor_values)
    union_entries = open_values.nnz + vendor_values.nnz - product.nnz
    difference = open_values - vendor_values
    difference.sum_duplicates()
    difference.eliminate_zeros()
    tolerance = 1e-12
    nonexact = int(np.count_nonzero(np.abs(difference.data) > tolerance))
    dot = float(np.sum(product.data, dtype=np.float64))
    norm = math.sqrt(
        float(np.dot(open_values.data, open_values.data))
        * float(np.dot(vendor_values.data, vendor_values.data))
    )
    open_genes = np.asarray(open_values.sum(axis=1)).ravel()
    vendor_genes = np.asarray(vendor_values.sum(axis=1)).ravel()
    open_bins = np.asarray(open_values.sum(axis=0)).ravel()
    vendor_bins = np.asarray(vendor_values.sum(axis=0)).ravel()
    normalized = open_values.copy()
    if open_mass:
        normalized.data /= open_mass
    normalized_vendor = vendor_values.copy()
    if vendor_mass:
        normalized_vendor.data /= vendor_mass
    normalized -= normalized_vendor
    normalized.eliminate_zeros()
    return {
        "open_raw_mass": open_mass,
        "space_ranger_raw_mass": vendor_mass,
        "signed_mass_difference": open_mass - vendor_mass,
        "absolute_mass_difference": abs(open_mass - vendor_mass),
        "mass_ratio": open_mass / vendor_mass if vendor_mass else None,
        "union_nonzero_entries": int(union_entries),
        "union_axis_l1": float(np.sum(np.abs(difference.data), dtype=np.float64)),
        "exact_entries": int(union_entries - nonexact),
        "exact_entry_fraction": (
            (union_entries - nonexact) / union_entries if union_entries else 1.0
        ),
        "gene_total_pearson": _dense_correlation(open_genes, vendor_genes, ranked=False),
        "gene_total_spearman": _dense_correlation(open_genes, vendor_genes, ranked=True),
        "bin_total_pearson": _dense_correlation(open_bins, vendor_bins, ranked=False),
        "bin_total_spearman": _dense_correlation(open_bins, vendor_bins, ranked=True),
        "cosine_similarity": dot / norm if norm else (1.0 if not union_entries else 0.0),
        "normalized_total_variation": 0.5 * float(
            np.sum(np.abs(normalized.data), dtype=np.float64)
        ),
    }


def parse_vendor_h5(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        raw_scale, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"expected SCALE=PATH for --vendor-h5: {value}")
        scale = int(raw_scale)
        if scale not in SCALES or scale in result:
            raise ValueError(f"invalid or duplicate vendor scale: {scale}")
        result[scale] = Path(raw_path)
    if set(result) != set(SCALES):
        raise ValueError("--vendor-h5 must provide 2, 8, and 16 um matrices")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-mex-root", type=Path, required=True)
    parser.add_argument("--vendor-h5", action="append", required=True, metavar="SCALE=PATH")
    parser.add_argument("--umi-mode", choices=("1mm_cr", "exact"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    vendor_paths = parse_vendor_h5(args.vendor_h5)
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    for path in vendor_paths.values():
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    rows = []
    target_axis: set[str] | None = None
    open_feature_axis: list[str] | None = None
    feature_axis_audit: dict[str, dict[str, object]] = {}
    for scale in SCALES:
        vendor_values, target_index, grid_width = load_space_ranger_sparse(
            vendor_paths[scale], scale,
        )
        target_genes = set(target_index)
        if target_axis is None:
            target_axis = target_genes
        elif target_axis != target_genes:
            raise SystemExit("Space Ranger target feature axes differ across scales")
        strict_root = args.policy_mex_root / "strict" / f"square_{scale:03d}um"
        scale_features = [
            line.split("\t", 1)[0]
            for line in read_lines(mex_file(strict_root, "features.tsv"))
        ]
        if open_feature_axis is None:
            open_feature_axis = scale_features
        elif open_feature_axis != scale_features:
            raise SystemExit("open feature axes differ across scales")
        missing_features = sorted(target_genes - set(scale_features))
        missing_rows = [target_index[name] for name in missing_features]
        missing_mass = float(vendor_values[missing_rows, :].sum()) if missing_rows else 0.0
        feature_axis_audit[str(scale)] = {
            "open_features": len(scale_features),
            "space_ranger_target_features": len(target_genes),
            "space_ranger_features_absent_from_open_axis": missing_features,
            "absent_feature_space_ranger_mass": missing_mass,
            "absent_feature_space_ranger_mass_fraction": (
                missing_mass / float(vendor_values.sum()) if vendor_values.nnz else 0.0
            ),
        }
        for policy, directory in POLICIES:
            root = args.policy_mex_root / directory / f"square_{scale:03d}um"
            open_values = load_mex_sparse(
                root, scale, target_index, grid_width, expected_features=scale_features,
            )
            rows.append({"umi_mode": args.umi_mode, "policy": policy, "scale_um": scale,
                         **compare_sparse(open_values, vendor_values)})

    args.out_dir.mkdir(parents=True)
    table = args.out_dir / "matrix_concordance.tsv"
    fields = tuple(rows[0])
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "visium_hd_processing.flex_policy_matrix_concordance.v1",
        "umi_mode": args.umi_mode,
        "comparison_axis": "union of nonzero feature-by-bin entries on the declared target set",
        "inputs": {
            "policy_mex_root": str(args.policy_mex_root.resolve()),
            "vendor_h5": {
                str(scale): {"path": str(path.resolve()), "sha256": sha256(path)}
                for scale, path in sorted(vendor_paths.items())
            },
        },
        "feature_axis_audit": feature_axis_audit,
        "output": {"path": str(table.resolve()), "sha256": sha256(table)},
        "invariants": {
            "target_axis_identical_across_scales": True,
            "open_policy_feature_axes_identical": True,
            "space_ranger_mass_on_target_features_absent_from_open_axis_reported_separately": True,
            "raw_mass_not_normalized": True,
            "normalized_tv_is_secondary_shape_metric": True,
            "occupancy_jaccard_not_computed": True,
            "vendor_not_used_for_open_assignment": True,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
