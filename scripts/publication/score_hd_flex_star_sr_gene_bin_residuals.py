#!/usr/bin/env python3
"""Localize STAR-versus-Space-Ranger Flex residuals at gene-by-bin resolution.

The comparison uses the shared biological-gene and Space Ranger barcode axes.
For every gene and 8-um bin it separates shared, STAR-positive, and
Space-Ranger-positive count mass.  Independently derived H&E cell and nucleus
fractions are then used only to score where those residuals land.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import io as scipy_io
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_hd_flex_h5ad_aggregates import (  # noqa: E402
    barcode_coordinate,
    correlation,
    decode,
    mex_file,
    normalized_tv,
    open_text,
    sha256,
)
from compare_hd_flex_policy_matrices import space_ranger_feature_indexes  # noqa: E402


@dataclass
class MatrixBundle:
    matrix: sparse.csr_matrix
    gene_ids: list[str]
    gene_names: list[str]
    coordinates: np.ndarray
    inputs: list[dict[str, object]]
    kind: str


def parse_method(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=MEX_DIR")
    return label, Path(raw_path)


def normalize_gene_id(value: str) -> str:
    return value.split(".", 1)[0]


def input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_axes(
    gene_ids: list[str], gene_names: list[str], barcodes: list[str], *, scale_um: int,
) -> np.ndarray:
    if len(gene_ids) != len(gene_names) or len(gene_ids) != len(set(gene_ids)):
        raise ValueError("duplicate or inconsistent Space Ranger gene axis")
    if len(barcodes) != len(set(barcodes)):
        raise ValueError("duplicate Space Ranger barcode axis")
    coordinates = np.asarray(
        [barcode_coordinate(value, scale=scale_um) for value in barcodes],
        dtype=np.int32,
    )
    if len({tuple(value) for value in coordinates.tolist()}) != len(coordinates):
        raise ValueError("duplicate Space Ranger barcode coordinates")
    return coordinates


def load_space_ranger_h5(path: Path, *, scale_um: int) -> MatrixBundle:
    """Load the biological target panel from a 10x CSC count H5."""
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        features = group["features"]
        all_ids = [normalize_gene_id(value) for value in decode(features["id"][:])]
        all_names = decode(features["name"][:])
        target = np.asarray(
            sorted(int(value) for value in space_ranger_feature_indexes(features)),
            dtype=np.int64,
        )
        gene_ids = [all_ids[index] for index in target]
        gene_names = [all_names[index] for index in target]
        barcodes = decode(group["barcodes"][:])
        shape = tuple(int(value) for value in group["shape"][:])
        if shape != (len(all_ids), len(barcodes)):
            raise ValueError("Space Ranger H5 matrix shape disagrees with its axes")
        raw_indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        raw_indices = np.asarray(group["indices"][:], dtype=np.int64)
        raw_data = np.asarray(group["data"][:])
    coordinates = _validate_axes(
        gene_ids, gene_names, barcodes, scale_um=scale_um,
    )
    compact = np.full(len(all_ids), -1, dtype=np.int32)
    compact[target] = np.arange(len(target), dtype=np.int32)
    mapped = compact[raw_indices]
    selected = mapped >= 0
    prefix = np.empty(len(selected) + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(selected, dtype=np.int64, out=prefix[1:])
    filtered_indptr = prefix[raw_indptr]
    matrix_csc = sparse.csc_matrix(
        (raw_data[selected], mapped[selected], filtered_indptr),
        shape=(len(gene_ids), len(barcodes)),
    )
    matrix = matrix_csc.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    del raw_data, raw_indices, raw_indptr, mapped, selected, prefix, matrix_csc
    gc.collect()
    return MatrixBundle(
        matrix=matrix,
        gene_ids=gene_ids,
        gene_names=gene_names,
        coordinates=coordinates,
        inputs=[input_record(path)],
        kind="space_ranger_target_gene_h5",
    )


def load_space_ranger_h5ad(path: Path, *, scale_um: int) -> MatrixBundle:
    """Load a retained Space Ranger CSR H5AD as gene-by-barcode CSR."""
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        encoding = x.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding != "csr_matrix":
            raise ValueError(f"expected CSR H5AD X, observed {encoding!r}")
        barcodes = decode(handle["obs/_index"][:])
        gene_ids = [
            normalize_gene_id(value) for value in decode(handle["var/gene_ids"][:])
        ]
        gene_names = decode(handle["var/_index"][:])
        shape = tuple(int(value) for value in x.attrs["shape"])
        if shape != (len(barcodes), len(gene_ids)):
            raise ValueError("Space Ranger H5AD matrix shape disagrees with its axes")
        data = np.asarray(x["data"][:])
        indices = np.asarray(x["indices"][:], dtype=np.int32)
        indptr = np.asarray(x["indptr"][:], dtype=np.int64)
    coordinates = _validate_axes(
        gene_ids, gene_names, barcodes, scale_um=scale_um,
    )
    barcode_by_gene = sparse.csr_matrix(
        (data, indices, indptr), shape=shape,
    )
    matrix = barcode_by_gene.transpose().tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    del barcode_by_gene, data, indices, indptr
    gc.collect()
    return MatrixBundle(
        matrix=matrix,
        gene_ids=gene_ids,
        gene_names=gene_names,
        coordinates=coordinates,
        inputs=[input_record(path)],
        kind="published_space_ranger_filtered_h5ad",
    )


def load_he_weights(
    root: Path, expected_coordinates: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    summary_path = root / "summary.json"
    arrays_path = root / "he_bin_weights.npz"
    if not summary_path.is_file() or not arrays_path.is_file():
        raise ValueError(f"incomplete H&E bin weights: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "visium_hd_processing.flex_he_bin_weights.v1":
        raise ValueError("unexpected H&E bin-weight schema")
    if sha256(arrays_path) != summary["outputs"]["he_bin_weights.npz"]["sha256"]:
        raise ValueError("H&E bin-weight array differs from its declared hash")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        coordinates = np.asarray(arrays["coordinates"], dtype=np.int32)
        weights = {
            name: np.asarray(arrays[name], dtype=np.float64)
            for name in ("cell_fraction", "nucleus_fraction", "extracellular_fraction")
        }
        children = np.asarray(arrays["he_supported_children"], dtype=np.int32)
    if not np.array_equal(coordinates, expected_coordinates):
        raise ValueError("H&E weights and Space Ranger barcode axes differ")
    if any(value.shape != (len(coordinates),) for value in weights.values()):
        raise ValueError("invalid H&E bin-weight shape")
    if np.any(children < 0) or (np.any(children == 0) and not summary.get("allow_unsupported_coarse_barcodes", False)) or np.any(weights["nucleus_fraction"] > weights["cell_fraction"]):
        raise ValueError("invalid H&E bin weights")
    if not np.allclose(
        weights["cell_fraction"] + weights["extracellular_fraction"],
        (children > 0).astype(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("H&E cell/extracellular weights do not reconcile")
    weights["supported"] = (children > 0).astype(float)
    metadata = {
        "summary": input_record(summary_path),
        "arrays": input_record(arrays_path),
    }
    return weights, metadata


def load_star_mex(
    root: Path,
    *,
    sr_gene_ids: list[str],
    sr_coordinates: np.ndarray,
    scale_um: int,
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    feature_path = mex_file(root, "features.tsv")
    barcode_path = mex_file(root, "barcodes.tsv")
    matrix_path = mex_file(root, "matrix.mtx")
    with open_text(feature_path) as handle:
        feature_rows = [line.rstrip("\n").split("\t") for line in handle]
    star_gene_ids = [normalize_gene_id(row[0]) for row in feature_rows]
    if len(star_gene_ids) != len(set(star_gene_ids)):
        raise ValueError(f"duplicate STAR gene axis in {root}")
    sr_gene_index = {value: index for index, value in enumerate(sr_gene_ids)}
    unexpected = sorted(set(star_gene_ids) - set(sr_gene_index))
    if unexpected:
        raise ValueError(f"{len(unexpected)} STAR genes absent from Space Ranger axis")
    row_map = np.asarray([sr_gene_index[value] for value in star_gene_ids], dtype=np.int32)
    with open_text(barcode_path) as handle:
        star_barcodes = [line.rstrip("\n").split("\t", 1)[0] for line in handle]
    star_coordinates = [
        barcode_coordinate(value, scale=scale_um) for value in star_barcodes
    ]
    if len(star_coordinates) != len(set(star_coordinates)):
        raise ValueError(f"duplicate STAR barcode coordinates in {root}")
    sr_coordinate_index = {
        tuple(int(value) for value in coordinate): index
        for index, coordinate in enumerate(sr_coordinates)
    }
    column_map = np.asarray(
        [sr_coordinate_index.get(value, -1) for value in star_coordinates],
        dtype=np.int32,
    )
    raw = scipy_io.mmread(str(matrix_path)).tocoo(copy=False)
    if raw.shape != (len(star_gene_ids), len(star_barcodes)):
        raise ValueError(f"STAR MEX matrix shape disagrees with its axes in {root}")
    if np.any(~np.isfinite(raw.data)) or np.any(raw.data < 0):
        raise ValueError(f"invalid STAR MEX values in {root}")
    mapped_columns = column_map[raw.col]
    nonzero = raw.data != 0
    on_axis = mapped_columns >= 0
    selected = on_axis & nonzero
    selected_count = int(np.count_nonzero(selected))
    matrix = sparse.coo_matrix(
        (
            raw.data[selected],
            (row_map[raw.row[selected]], mapped_columns[selected]),
        ),
        shape=(len(sr_gene_ids), len(sr_coordinates)),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.nnz != selected_count:
        raise ValueError(f"duplicate STAR gene-bin entries in {root}")
    outside_nnz = int(np.count_nonzero((~on_axis) & nonzero))
    explicit_zero_entries = int(np.count_nonzero(~nonzero))
    outside_mass = float(np.sum(raw.data[~on_axis], dtype=np.float64))
    full_mass = float(np.sum(raw.data, dtype=np.float64))
    missing = sorted(set(sr_gene_ids) - set(star_gene_ids))
    del raw, mapped_columns, selected, row_map, column_map
    gc.collect()
    return matrix, {
        "inputs": [input_record(path) for path in (feature_path, barcode_path, matrix_path)],
        "star_genes": len(star_gene_ids),
        "space_ranger_genes_absent_from_star": len(missing),
        "space_ranger_gene_ids_absent_from_star": missing,
        "star_barcodes": len(star_barcodes),
        "star_gene_bin_nnz_on_space_ranger_axis": int(matrix.nnz),
        "star_gene_bin_nnz_outside_space_ranger_axis": outside_nnz,
        "star_matrix_market_explicit_zero_entries": explicit_zero_entries,
        "star_mass_all_axes": full_mass,
        "star_mass_outside_space_ranger_axis": outside_mass,
    }


def axis_sum(matrix: sparse.csr_matrix, axis: int) -> np.ndarray:
    return np.asarray(matrix.sum(axis=axis, dtype=np.float64)).ravel()


def matrix_vector(matrix: sparse.csr_matrix, vector: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ np.asarray(vector, dtype=np.float64)).ravel()


def safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(np.asarray(numerator, dtype=np.float64))
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) > 0)
    return result


def positive_difference(
    source: sparse.csr_matrix, shared: sparse.csr_matrix,
) -> sparse.csr_matrix:
    result = (source - shared).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    if result.nnz and float(np.min(result.data)) < -1e-7:
        raise ValueError("negative gene-bin residual")
    if result.nnz:
        result.data[result.data < 0] = 0
        result.eliminate_zeros()
    result.sort_indices()
    return result


def weighted_gene_reference(
    residual_mass: np.ndarray,
    reference_fraction: np.ndarray,
    reference_total: np.ndarray,
) -> tuple[float, float]:
    covered = reference_total > 0
    covered_mass = float(np.sum(residual_mass[covered], dtype=np.float64))
    total = float(np.sum(residual_mass, dtype=np.float64))
    value = (
        float(np.dot(residual_mass[covered], reference_fraction[covered])) / covered_mass
        if covered_mass else 0.0
    )
    return value, covered_mass / total if total else 0.0


def _top_fraction(values: np.ndarray, count: int) -> float:
    total = float(np.sum(values, dtype=np.float64))
    if not total:
        return 0.0
    selected = np.sort(np.asarray(values, dtype=np.float64))[-min(count, len(values)):]
    return float(np.sum(selected, dtype=np.float64) / total)


def analyse_method(
    *,
    dataset: str,
    method: str,
    gene_ids: list[str],
    gene_names: list[str],
    star: sparse.csr_matrix,
    space_ranger: sparse.csr_matrix,
    weights: dict[str, np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if star.shape != space_ranger.shape:
        raise ValueError("STAR and Space Ranger gene-bin axes differ")
    if star.shape != (len(gene_ids), len(weights["cell_fraction"])):
        raise ValueError("gene or H&E barcode axis differs from matrix shape")
    shared = star.minimum(space_ranger).tocsr()
    shared.eliminate_zeros()
    star_positive = positive_difference(star, shared)
    sr_positive = positive_difference(space_ranger, shared)

    star_total = axis_sum(star, 1)
    sr_total = axis_sum(space_ranger, 1)
    shared_total = axis_sum(shared, 1)
    star_positive_total = axis_sum(star_positive, 1)
    sr_positive_total = axis_sum(sr_positive, 1)
    signed_delta = star_total - sr_total
    if not np.allclose(shared_total + star_positive_total, star_total, rtol=0, atol=1e-5):
        raise ValueError("STAR gene-bin residuals do not reconcile")
    if not np.allclose(shared_total + sr_positive_total, sr_total, rtol=0, atol=1e-5):
        raise ValueError("Space Ranger gene-bin residuals do not reconcile")

    cell = weights["cell_fraction"]
    nucleus = weights["nucleus_fraction"]
    star_cell = matrix_vector(star, cell)
    star_nucleus = matrix_vector(star, nucleus)
    sr_cell = matrix_vector(space_ranger, cell)
    sr_nucleus = matrix_vector(space_ranger, nucleus)
    star_positive_cell = matrix_vector(star_positive, cell)
    star_positive_nucleus = matrix_vector(star_positive, nucleus)
    sr_positive_cell = matrix_vector(sr_positive, cell)
    sr_positive_nucleus = matrix_vector(sr_positive, nucleus)

    support = weights.get("supported", np.ones(star.shape[1]))
    star_he_total = matrix_vector(star, support)
    sr_he_total = matrix_vector(space_ranger, support)
    star_positive_he_total = matrix_vector(star_positive, support)
    sr_positive_he_total = matrix_vector(sr_positive, support)
    star_he_mass = float(np.sum(star_he_total))
    sr_he_mass = float(np.sum(sr_he_total))
    residual_he_mass = float(np.sum(star_positive_he_total))

    star_cell_fraction = safe_fraction(star_cell, star_he_total)
    star_nucleus_fraction = safe_fraction(star_nucleus, star_he_total)
    sr_cell_fraction = safe_fraction(sr_cell, sr_he_total)
    sr_nucleus_fraction = safe_fraction(sr_nucleus, sr_he_total)
    star_positive_cell_fraction = safe_fraction(star_positive_cell, star_positive_he_total)
    star_positive_nucleus_fraction = safe_fraction(
        star_positive_nucleus, star_positive_he_total,
    )
    sr_positive_cell_fraction = safe_fraction(sr_positive_cell, sr_positive_he_total)
    sr_positive_nucleus_fraction = safe_fraction(sr_positive_nucleus, sr_positive_he_total)

    rows: list[dict[str, object]] = []
    residual_mass = float(np.sum(star_positive_total, dtype=np.float64))
    for index, (gene_id, gene_name) in enumerate(zip(gene_ids, gene_names, strict=True)):
        rows.append({
            "dataset": dataset,
            "method": method,
            "gene_id": gene_id,
            "gene_name": gene_name,
            "star_mass": float(star_total[index]),
            "space_ranger_mass": float(sr_total[index]),
            "signed_star_minus_space_ranger_mass": float(signed_delta[index]),
            "shared_gene_bin_mass": float(shared_total[index]),
            "star_positive_gene_bin_residual_mass": float(star_positive_total[index]),
            "space_ranger_positive_gene_bin_residual_mass": float(sr_positive_total[index]),
            "star_positive_residual_mass_fraction": (
                float(star_positive_total[index] / residual_mass) if residual_mass else 0.0
            ),
            "star_cell_mass_fraction": float(star_cell_fraction[index]),
            "space_ranger_cell_mass_fraction": float(sr_cell_fraction[index]),
            "star_minus_space_ranger_cell_fraction_pp": float(
                100.0 * (star_cell_fraction[index] - sr_cell_fraction[index])
            ),
            "star_nucleus_mass_fraction": float(star_nucleus_fraction[index]),
            "space_ranger_nucleus_mass_fraction": float(sr_nucleus_fraction[index]),
            "star_minus_space_ranger_nucleus_fraction_pp": float(
                100.0 * (star_nucleus_fraction[index] - sr_nucleus_fraction[index])
            ),
            "star_positive_residual_cell_mass": float(star_positive_cell[index]),
            "star_positive_residual_cell_mass_fraction": float(
                star_positive_cell_fraction[index]
            ),
            "star_positive_residual_nucleus_mass": float(star_positive_nucleus[index]),
            "star_positive_residual_nucleus_mass_fraction": float(
                star_positive_nucleus_fraction[index]
            ),
            "space_ranger_positive_residual_cell_mass": float(sr_positive_cell[index]),
            "space_ranger_positive_residual_cell_mass_fraction": float(
                sr_positive_cell_fraction[index]
            ),
            "space_ranger_positive_residual_nucleus_mass": float(sr_positive_nucleus[index]),
            "space_ranger_positive_residual_nucleus_mass_fraction": float(
                sr_positive_nucleus_fraction[index]
            ),
            "star_positive_residual_gene_bin_nnz": int(
                star_positive.indptr[index + 1] - star_positive.indptr[index]
            ),
            "space_ranger_positive_residual_gene_bin_nnz": int(
                sr_positive.indptr[index + 1] - sr_positive.indptr[index]
            ),
        })

    star_mass = float(np.sum(star_total, dtype=np.float64))
    sr_mass = float(np.sum(sr_total, dtype=np.float64))
    shared_mass = float(np.sum(shared_total, dtype=np.float64))
    sr_positive_mass = float(np.sum(sr_positive_total, dtype=np.float64))
    star_bin = axis_sum(star, 0)
    sr_bin = axis_sum(space_ranger, 0)
    star_spatial_positive = float(np.sum(np.clip(star_bin - sr_bin, 0, None)))
    sr_spatial_positive = float(np.sum(np.clip(sr_bin - star_bin, 0, None)))
    sr_same_gene_cell, sr_reference_coverage = weighted_gene_reference(
        star_positive_he_total, sr_cell_fraction, sr_he_total,
    )
    sr_same_gene_nucleus, sr_nucleus_reference_coverage = weighted_gene_reference(
        star_positive_he_total, sr_nucleus_fraction, sr_he_total,
    )
    probabilities = star_positive_total / residual_mass if residual_mass else star_positive_total
    effective_genes = (
        float(1.0 / np.sum(probabilities * probabilities)) if residual_mass else 0.0
    )
    summary = {
        "dataset": dataset,
        "method": method,
        "genes": len(gene_ids),
        "barcodes": star.shape[1],
        "star_mass_on_space_ranger_axis": star_mass,
        "space_ranger_mass": sr_mass,
        "signed_mass_difference": star_mass - sr_mass,
        "star_percent_excess_over_space_ranger": (
            100.0 * (star_mass - sr_mass) / sr_mass if sr_mass else 0.0
        ),
        "shared_gene_bin_mass": shared_mass,
        "star_positive_gene_bin_residual_mass": residual_mass,
        "space_ranger_positive_gene_bin_residual_mass": sr_positive_mass,
        "star_to_space_ranger_positive_gene_bin_residual_mass_ratio": (
            residual_mass / sr_positive_mass if sr_positive_mass else None
        ),
        "star_positive_spatial_bin_residual_mass": star_spatial_positive,
        "space_ranger_positive_spatial_bin_residual_mass": sr_spatial_positive,
        "feature_bin_to_spatial_bin_star_positive_residual_ratio": (
            residual_mass / star_spatial_positive if star_spatial_positive else None
        ),
        "genes_with_star_positive_gene_bin_residual": int(
            np.count_nonzero(star_positive_total)
        ),
        "genes_with_positive_net_star_delta": int(np.count_nonzero(signed_delta > 0)),
        "genes_with_negative_net_star_delta": int(np.count_nonzero(signed_delta < 0)),
        "star_positive_residual_effective_gene_count": effective_genes,
        "star_positive_residual_top10_gene_fraction": _top_fraction(star_positive_total, 10),
        "star_positive_residual_top100_gene_fraction": _top_fraction(star_positive_total, 100),
        "star_positive_residual_vs_space_ranger_gene_total_pearson": correlation(
            star_positive_total, sr_total,
        ),
        "star_positive_residual_vs_space_ranger_gene_total_spearman": correlation(
            star_positive_total, sr_total, ranked=True,
        ),
        "star_vs_space_ranger_gene_total_pearson": correlation(star_total, sr_total),
        "star_vs_space_ranger_gene_normalized_total_variation": normalized_tv(
            star_total, sr_total,
        ),
        "space_ranger_whole_cell_mass_fraction": (
            float(np.sum(sr_cell) / sr_he_mass) if sr_he_mass else 0.0
        ),
        "star_whole_cell_mass_fraction": (
            float(np.sum(star_cell) / star_he_mass) if star_he_mass else 0.0
        ),
        "star_positive_residual_cell_mass_fraction": (
            float(np.sum(star_positive_cell) / residual_he_mass) if residual_he_mass else 0.0
        ),
        "same_gene_space_ranger_cell_fraction_reference": sr_same_gene_cell,
        "star_positive_minus_same_gene_space_ranger_cell_fraction_pp": 100.0 * (
            (float(np.sum(star_positive_cell) / residual_he_mass) if residual_he_mass else 0.0)
            - sr_same_gene_cell
        ),
        "same_gene_space_ranger_cell_reference_residual_mass_coverage": sr_reference_coverage,
        "space_ranger_whole_nucleus_mass_fraction": (
            float(np.sum(sr_nucleus) / sr_he_mass) if sr_he_mass else 0.0
        ),
        "star_whole_nucleus_mass_fraction": (
            float(np.sum(star_nucleus) / star_he_mass) if star_he_mass else 0.0
        ),
        "star_positive_residual_nucleus_mass_fraction": (
            float(np.sum(star_positive_nucleus) / residual_he_mass) if residual_he_mass else 0.0
        ),
        "same_gene_space_ranger_nucleus_fraction_reference": sr_same_gene_nucleus,
        "star_positive_minus_same_gene_space_ranger_nucleus_fraction_pp": 100.0 * (
            (float(np.sum(star_positive_nucleus) / residual_he_mass) if residual_he_mass else 0.0)
            - sr_same_gene_nucleus
        ),
        "same_gene_space_ranger_nucleus_reference_residual_mass_coverage": (
            sr_nucleus_reference_coverage
        ),
        "star_gene_bin_nnz": int(star.nnz),
        "space_ranger_gene_bin_nnz": int(space_ranger.nnz),
        "shared_gene_bin_nnz": int(shared.nnz),
        "star_positive_gene_bin_residual_nnz": int(star_positive.nnz),
        "space_ranger_positive_gene_bin_residual_nnz": int(sr_positive.nnz),
        "raw_mass_normalized": False,
    }
    del shared, star_positive, sr_positive
    gc.collect()
    return summary, rows


def load_prior_reconciliation(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or len(rows) != len({row["method"] for row in rows}):
        raise ValueError("invalid prior morphology reconciliation table")
    return {row["method"]: row for row in rows}


def assert_prior_reconciliation(
    summary: dict[str, object], prior: dict[str, str], *, tolerance: float = 1e-5,
) -> None:
    pairs = {
        "star_mass_on_space_ranger_axis": "star_mass_on_space_ranger_axis",
        "space_ranger_mass": "space_ranger_mass",
        "star_positive_spatial_bin_residual_mass": (
            "star_positive_spatial_bin_residual_mass"
        ),
        "space_ranger_positive_spatial_bin_residual_mass": (
            "space_ranger_positive_spatial_bin_residual_mass"
        ),
    }
    for observed_key, prior_key in pairs.items():
        observed = float(summary[observed_key])
        expected = float(prior[prior_key])
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=tolerance):
            raise ValueError(
                f"gene-bin analysis disagrees with prior {prior_key}: "
                f"observed {observed}, expected {expected}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--space-ranger-h5", type=Path)
    source.add_argument("--space-ranger-h5ad", type=Path)
    parser.add_argument(
        "--star-method", action="append", type=parse_method, required=True,
        metavar="LABEL=MEX_DIR",
    )
    parser.add_argument("--he-bin-weights", type=Path, required=True)
    parser.add_argument("--prior-morphology", type=Path, required=True)
    parser.add_argument("--scale-um", type=int, default=8)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-genes", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    labels = [label for label, _ in args.star_method]
    if len(labels) != len(set(labels)):
        raise SystemExit("duplicate --star-method label")
    if args.scale_um < 2 or args.top_genes < 1:
        raise SystemExit("--scale-um and --top-genes must be positive")
    sr_path = args.space_ranger_h5 or args.space_ranger_h5ad
    if not sr_path or not sr_path.is_file():
        raise SystemExit(f"missing Space Ranger input: {sr_path}")
    for _, root in args.star_method:
        if not root.is_dir():
            raise SystemExit(f"missing STAR MEX directory: {root}")
    prior_path = args.prior_morphology / "mass_reconciliation.tsv"
    if not prior_path.is_file():
        raise SystemExit(f"missing prior morphology reconciliation: {prior_path}")

    space_ranger = (
        load_space_ranger_h5(args.space_ranger_h5, scale_um=args.scale_um)
        if args.space_ranger_h5
        else load_space_ranger_h5ad(args.space_ranger_h5ad, scale_um=args.scale_um)
    )
    weights, weight_inputs = load_he_weights(
        args.he_bin_weights, space_ranger.coordinates,
    )
    prior = load_prior_reconciliation(prior_path)
    method_summaries: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    star_inputs: dict[str, object] = {}
    sr_gene_totals = axis_sum(space_ranger.matrix, 1)
    for label, root in args.star_method:
        if label not in prior:
            raise ValueError(f"method {label!r} absent from prior morphology result")
        star, metadata = load_star_mex(
            root,
            sr_gene_ids=space_ranger.gene_ids,
            sr_coordinates=space_ranger.coordinates,
            scale_um=args.scale_um,
        )
        summary, rows = analyse_method(
            dataset=args.dataset,
            method=label,
            gene_ids=space_ranger.gene_ids,
            gene_names=space_ranger.gene_names,
            star=star,
            space_ranger=space_ranger.matrix,
            weights=weights,
        )
        assert_prior_reconciliation(summary, prior[label])
        absent_gene_ids = set(metadata["space_ranger_gene_ids_absent_from_star"])
        missing_indexes = [
            index for index, gene_id in enumerate(space_ranger.gene_ids)
            if gene_id in absent_gene_ids
        ]
        metadata["space_ranger_mass_on_genes_absent_from_star"] = float(
            np.sum(sr_gene_totals[missing_indexes], dtype=np.float64)
        )
        method_summaries.append({**summary, **{
            key: value for key, value in metadata.items() if key != "inputs"
        }})
        gene_rows.extend(rows)
        star_inputs[label] = metadata["inputs"]
        del star, rows
        gc.collect()

    args.out_dir.mkdir(parents=True)
    method_path = args.out_dir / "method_summary.tsv"
    gene_path = args.out_dir / "gene_residuals.tsv"
    top_path = args.out_dir / "top_gene_residuals.tsv"
    write_rows(method_path, method_summaries)
    write_rows(gene_path, gene_rows)
    top_rows: list[dict[str, object]] = []
    for label in labels:
        selected = [row for row in gene_rows if row["method"] == label]
        selected.sort(
            key=lambda row: (
                -float(row["star_positive_gene_bin_residual_mass"]),
                str(row["gene_id"]),
            )
        )
        cumulative = 0.0
        total = sum(float(row["star_positive_gene_bin_residual_mass"]) for row in selected)
        for rank, row in enumerate(selected[: args.top_genes], start=1):
            cumulative += float(row["star_positive_gene_bin_residual_mass"])
            top_rows.append({
                "rank": rank,
                "cumulative_star_positive_residual_fraction": (
                    cumulative / total if total else 0.0
                ),
                **row,
            })
    write_rows(top_path, top_rows)
    summary = {
        "schema": "visium_hd_processing.flex_star_sr_gene_bin_he_residuals.v1",
        "dataset": args.dataset,
        "comparison_scale_um": args.scale_um,
        "he_unsupported_barcodes": int(np.count_nonzero(weights["supported"] == 0)),
        "he_support_policy": "Exclude unsupported bins from H&E only; preserve full count axis and totals",
        "primary_axis": "shared_biological_gene_by_space_ranger_barcode_axis",
        "residual_definition": (
            "For each biological gene and Space Ranger-axis 8-um bin: shared=min(STAR,SR), "
            "STAR-positive=STAR-shared, SR-positive=SR-shared."
        ),
        "molecule_identity_compared": False,
        "molecule_identity_limitation": (
            "The shared published SPATCH comparator is a count H5AD without molecule identities; "
            "gene-by-bin is therefore the finest common comparison level."
        ),
        "morphology_projection": (
            "Each gene-bin residual is weighted by the independently segmented H&E cell and "
            "nucleus fractions among that 8-um bin's supported 2-um child centers."
        ),
        "methods": {row["method"]: row for row in method_summaries},
        "inputs": {
            "space_ranger": space_ranger.inputs,
            "star_methods": star_inputs,
            "he_bin_weights": weight_inputs,
            "prior_morphology_reconciliation": input_record(prior_path),
        },
        "outputs": {},
        "invariants": {
            "biological_feature_axis_only": True,
            "diagnostic_non_panel_counts_excluded": True,
            "space_ranger_barcode_axis_is_primary": True,
            "prior_total_bin_residual_result_reconciled": True,
            "image_or_segmentation_not_used_for_assignment": True,
            "raw_mass_not_normalized": True,
            "space_ranger_is_comparator_not_truth": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    for path in (method_path, gene_path, top_path):
        summary["outputs"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    checksum_paths = (method_path, gene_path, top_path, summary_path)
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
