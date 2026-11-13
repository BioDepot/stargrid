#!/usr/bin/env python3
"""Compare Visium count products with an adjacent official Xenium section.

This generator deliberately separates raw molecule mass from normalized shape
metrics.  Registration is expression-blind and the comparison cohort is fixed
from Xenium cell support plus the Visium H&E tissue flag, not from a count
matrix.  Xenium graph clusters are treated as unnamed spatial domains; the
script does not infer biological cell-type labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.io import mmread
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA = "spatial_suite.adjacent_xenium_policy_comparison.v1"
STAR_BARCODE_RE = re.compile(r"^s_016um_(\d+)_(\d+)-1$")
SHAPE_METRICS = [
    "eligible_gene_count",
    "gene_total_log1p_pearson",
    "gene_total_spearman",
    "median_gene_spatial_pearson",
    "median_gene_spatial_spearman",
    "median_patch_profile_pearson",
    "macro_cluster_fraction_pearson",
    "macro_cluster_fraction_spearman",
    "macro_cluster_dominant_roc_auc",
    "macro_cluster_dominant_average_precision",
    "evaluated_dominant_cluster_count",
]


class ComparisonError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-h5", required=True, type=Path)
    parser.add_argument(
        "--policy-mex",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="A named STAR policy MEX directory; may be repeated.",
    )
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--scalefactors", required=True, type=Path)
    parser.add_argument("--xenium-h5", required=True, type=Path)
    parser.add_argument("--xenium-cells", required=True, type=Path)
    parser.add_argument("--xenium-clusters", required=True, type=Path)
    parser.add_argument("--xenium-diffexp", required=True, type=Path)
    parser.add_argument("--xenium-experiment", required=True, type=Path)
    parser.add_argument("--xenium-he-alignment", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patch-um", type=float, default=128.0)
    parser.add_argument("--xenium-offset-x-um", type=float, default=0.0)
    parser.add_argument("--xenium-offset-y-um", type=float, default=0.0)
    parser.add_argument("--min-xenium-cells", type=int, default=10)
    parser.add_argument("--min-visium-tissue-bins", type=int, default=16)
    parser.add_argument("--markers-per-cluster", type=int, default=20)
    parser.add_argument("--min-dominant-patches", type=int, default=10)
    parser.add_argument("--min-gene-counts", type=float, default=100.0)
    parser.add_argument("--xenium-split-seed", type=int, default=1729)
    parser.add_argument(
        "--reference-panel-json",
        type=Path,
        help="Optional official Xenium panel JSON defining a biological gene subset.",
    )
    parser.add_argument("--reference-panel-source-category", default="current")
    parser.add_argument("--reference-gene-set-name", default="reference_panel")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: Iterable[Any]) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def read_10x_h5(path: Path) -> tuple[sparse.csc_matrix, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(x) for x in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        )
        feature_ids = decode(group["features/id"][:])
        barcodes = decode(group["barcodes"][:])
    if matrix.shape != (len(feature_ids), len(barcodes)):
        raise ComparisonError(f"inconsistent H5 axes at {path}: {matrix.shape}")
    return matrix, feature_ids, barcodes


def read_h5_feature_ids(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return decode(handle["matrix/features/id"][:])


def _resolve_mex_file(directory: Path, name: str) -> Path:
    plain = directory / name
    gzipped = directory / f"{name}.gz"
    if plain.is_file():
        return plain
    if gzipped.is_file():
        return gzipped
    raise ComparisonError(f"missing {name}[.gz] under {directory}")


def read_mex(directory: Path) -> tuple[sparse.csc_matrix, list[str], list[str]]:
    matrix = mmread(_resolve_mex_file(directory, "matrix.mtx")).tocsc()
    with _resolve_mex_file(directory, "features.tsv").open("rt", encoding="utf-8") as handle:
        feature_ids = [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]
    with _resolve_mex_file(directory, "barcodes.tsv").open("rt", encoding="utf-8") as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    if matrix.shape != (len(feature_ids), len(barcodes)):
        raise ComparisonError(f"inconsistent MEX axes at {directory}: {matrix.shape}")
    return matrix, feature_ids, barcodes


def read_mex_feature_ids(directory: Path) -> list[str]:
    with _resolve_mex_file(directory, "features.tsv").open("rt", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]


def parse_policy_args(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ComparisonError(f"--policy-mex must be NAME=DIR: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in seen:
            raise ComparisonError(f"duplicate or empty policy name: {name!r}")
        seen.add(name)
        result.append((name, Path(raw_path)))
    return result


def xenium_to_visium_transform(
    he_alignment: np.ndarray,
    moving_fullres_to_fixed_fullres: np.ndarray,
    pixel_size_um: float,
) -> np.ndarray:
    if he_alignment.shape != (3, 3) or moving_fullres_to_fixed_fullres.shape != (3, 3):
        raise ComparisonError("alignment transforms must be 3x3 matrices")
    if pixel_size_um <= 0:
        raise ComparisonError("Xenium pixel size must be positive")
    micron_to_dapi = np.diag([1.0 / pixel_size_um, 1.0 / pixel_size_um, 1.0])
    # The 10x image-alignment file maps imported H&E pixels to the fixed
    # Xenium/DAPI pixel system.  Cells start in Xenium microns, so invert it.
    return moving_fullres_to_fixed_fullres @ np.linalg.inv(he_alignment) @ micron_to_dapi


def transform_points(x: np.ndarray, y: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([x.astype(float), y.astype(float), np.ones(len(x), dtype=float)])
    mapped = transform @ points
    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def patch_keys(x_fullres: np.ndarray, y_fullres: np.ndarray, microns_per_pixel: float, patch_um: float) -> list[tuple[int, int]]:
    if microns_per_pixel <= 0 or patch_um <= 0:
        raise ComparisonError("spatial scales must be positive")
    x_index = np.floor(x_fullres * microns_per_pixel / patch_um).astype(np.int64)
    y_index = np.floor(y_fullres * microns_per_pixel / patch_um).astype(np.int64)
    return list(zip(x_index.tolist(), y_index.tolist()))


def make_patch_axis(*key_sets: Iterable[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    keys = sorted(set().union(*(set(values) for values in key_sets)))
    return keys, {key: index for index, key in enumerate(keys)}


def mapping_vector(keys: Iterable[tuple[int, int]], lookup: dict[tuple[int, int], int]) -> np.ndarray:
    return np.asarray([lookup.get(key, -1) for key in keys], dtype=np.int64)


def grouping_matrix(groups: np.ndarray, n_groups: int) -> sparse.csr_matrix:
    columns = np.flatnonzero(groups >= 0)
    return sparse.coo_matrix(
        (np.ones(len(columns), dtype=np.float32), (columns, groups[columns])),
        shape=(len(groups), n_groups),
    ).tocsr()


def target_row_indices(source_ids: list[str], target_ids: list[str]) -> np.ndarray:
    lookup: dict[str, int] = {}
    for index, value in enumerate(source_ids):
        if value in lookup:
            raise ComparisonError(f"duplicate feature ID in source matrix: {value}")
        lookup[value] = index
    missing = [value for value in target_ids if value not in lookup]
    if missing:
        raise ComparisonError(f"source matrix is missing {len(missing)} target features; first={missing[0]}")
    return np.asarray([lookup[value] for value in target_ids], dtype=np.int64)


def aggregate_matrix(
    matrix: sparse.spmatrix,
    source_ids: list[str],
    target_ids: list[str],
    column_groups: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if matrix.shape[1] != len(column_groups):
        raise ComparisonError("matrix columns do not match spatial mapping")
    rows = target_row_indices(source_ids, target_ids)
    group_matrix = grouping_matrix(column_groups, n_groups)
    selected = matrix[rows, :]
    aggregated = (selected @ group_matrix).toarray().astype(np.float64, copy=False)
    all_column_totals = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    valid = column_groups >= 0
    cohort_columns = np.zeros(len(column_groups), dtype=bool)
    return aggregated, {
        "total_mass": float(matrix.sum()),
        "panel_mass": float(selected.sum()),
        "mapped_total_mass": float(all_column_totals[valid].sum()),
        "mapped_panel_mass": float(aggregated.sum()),
        "_valid_column_count": float(np.count_nonzero(valid)),
    }


def visium_column_groups(
    barcodes: list[str],
    positions: pd.DataFrame,
    position_patch_lookup: dict[tuple[int, int], int],
) -> np.ndarray:
    by_barcode = dict(zip(positions["barcode"].astype(str), positions["_patch_index"].astype(int)))
    by_grid = {
        (int(row), int(column)): int(patch)
        for row, column, patch in zip(
            positions["array_row"], positions["array_col"], positions["_patch_index"]
        )
    }
    groups = np.full(len(barcodes), -1, dtype=np.int64)
    for index, barcode in enumerate(barcodes):
        if barcode in by_barcode:
            groups[index] = by_barcode[barcode]
            continue
        match = STAR_BARCODE_RE.match(barcode)
        if match:
            groups[index] = by_grid.get((int(match.group(1)), int(match.group(2))), -1)
    if np.count_nonzero(groups >= 0) != len(barcodes):
        missing = len(barcodes) - int(np.count_nonzero(groups >= 0))
        raise ComparisonError(f"could not map {missing}/{len(barcodes)} Visium barcodes")
    if not set(groups.tolist()).issubset(set(position_patch_lookup.values())):
        raise ComparisonError("Visium columns mapped outside the declared patch axis")
    return groups


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(stats.pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(stats.spearmanr(x, y).statistic)


def rowwise_pearson(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    if matrix_a.shape != matrix_b.shape or matrix_a.ndim != 2:
        raise ComparisonError("rowwise Pearson inputs must have identical two-dimensional shapes")
    observations = matrix_a.shape[1]
    sum_a = matrix_a.sum(axis=1)
    sum_b = matrix_b.sum(axis=1)
    covariance = np.einsum("ij,ij->i", matrix_a, matrix_b) - sum_a * sum_b / observations
    variance_a = np.einsum("ij,ij->i", matrix_a, matrix_a) - sum_a * sum_a / observations
    variance_b = np.einsum("ij,ij->i", matrix_b, matrix_b) - sum_b * sum_b / observations
    denominator = np.sqrt(np.maximum(variance_a, 0.0) * np.maximum(variance_b, 0.0))
    return np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )


def log_normalize(matrix: np.ndarray, scale: float = 10_000.0) -> np.ndarray:
    totals = matrix.sum(axis=0)
    normalized = np.divide(matrix, totals[None, :], out=np.zeros_like(matrix, dtype=float), where=totals[None, :] > 0)
    return np.log1p(normalized * scale)


def xenium_split_half_reliability(
    xenium_counts: np.ndarray,
    cohort: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = xenium_counts[:, cohort]
    rounded = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, rounded):
        raise ComparisonError("Xenium count matrix must be integer-valued for split-half reliability")
    rng = np.random.Generator(np.random.PCG64(seed))
    first = rng.binomial(rounded, 0.5)
    second = rounded - first
    split_correlation = rowwise_pearson(log_normalize(first), log_normalize(second))
    reliability = np.zeros_like(split_correlation)
    valid = np.isfinite(split_correlation) & (split_correlation > -0.999999)
    reliability[valid] = 2.0 * split_correlation[valid] / (1.0 + split_correlation[valid])
    reliability = np.clip(reliability, 0.0, 1.0)
    return split_correlation, reliability


def reliability_weighted_metrics(
    visium_counts: np.ndarray,
    xenium_counts: np.ndarray,
    cohort: np.ndarray,
    reliability: np.ndarray,
) -> tuple[float, float]:
    if reliability.shape != (visium_counts.shape[0],):
        raise ComparisonError("Xenium reliability weights do not match the gene axis")
    v = log_normalize(visium_counts[:, cohort])
    x = log_normalize(xenium_counts[:, cohort])
    gene_correlations = rowwise_pearson(v, x)
    reference_valid = reliability > 0
    if not np.any(reference_valid):
        return math.nan, math.nan
    # Eligibility is defined exclusively by the independent Xenium split.
    # A method with no spatial variation for a reliable Xenium gene receives
    # zero for that gene; it must not improve its score by dropping the gene.
    gene_scores = np.nan_to_num(
        gene_correlations[reference_valid],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    gene_spatial = float(np.average(gene_scores, weights=reliability[reference_valid]))
    weights = reliability[reference_valid]
    weighted_patch_correlations: list[float] = []
    for patch in range(v.shape[1]):
        vx = v[reference_valid, patch]
        xx = x[reference_valid, patch]
        v_mean = float(np.average(vx, weights=weights))
        x_mean = float(np.average(xx, weights=weights))
        v_centered = vx - v_mean
        x_centered = xx - x_mean
        denominator = math.sqrt(
            float(np.dot(weights, v_centered * v_centered))
            * float(np.dot(weights, x_centered * x_centered))
        )
        weighted_patch_correlations.append(
            float(np.dot(weights, v_centered * x_centered) / denominator)
            if denominator > 0
            else math.nan
        )
    return gene_spatial, float(np.nanmedian(weighted_patch_correlations))


def xenium_reliability_mass_curve(
    name: str,
    visium_counts: np.ndarray,
    xenium_counts: np.ndarray,
    gene_ids: list[str],
    cohort: np.ndarray,
    reliability: np.ndarray,
    steps: int = 20,
    selection_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Score fixed method outputs across an independent Xenium reliability axis.

    Xenium split-half reproducibility determines both ordering and weight.  All
    methods therefore see exactly the same genes at every point.  A missing or
    constant method profile receives zero gene-spatial correlation rather than
    disappearing from the denominator.
    """
    if steps < 2:
        raise ComparisonError("reliability-mass curve requires at least two steps")
    if reliability.shape != (len(gene_ids),):
        raise ComparisonError("Xenium reliability weights do not match the gene axis")
    selected = np.ones(len(gene_ids), dtype=bool)
    if selection_mask is not None:
        selected = np.asarray(selection_mask, dtype=bool)
        if selected.shape != (len(gene_ids),):
            raise ComparisonError("reference gene-set mask does not match the gene axis")
    positive = np.flatnonzero((reliability > 0) & selected)
    if positive.size < 3:
        raise ComparisonError("fewer than three genes have positive Xenium reliability")

    v = log_normalize(visium_counts[:, cohort])
    x = log_normalize(xenium_counts[:, cohort])
    gene_spatial = np.nan_to_num(rowwise_pearson(v, x), nan=0.0, posinf=0.0, neginf=0.0)
    reference_totals = xenium_counts[:, cohort].sum(axis=1)
    order = np.asarray(
        sorted(
            positive.tolist(),
            key=lambda index: (-reliability[index], -reference_totals[index], gene_ids[index]),
        ),
        dtype=np.int64,
    )
    ordered_weights = reliability[order]
    cumulative_weight = np.cumsum(ordered_weights)
    total_weight = float(cumulative_weight[-1])
    requested = np.linspace(0.05, 1.0, steps)
    rows: list[dict[str, Any]] = []
    for target in requested:
        rank = min(
            max(3, int(np.searchsorted(cumulative_weight, target * total_weight, side="left")) + 1),
            len(order),
        )
        selected = order[:rank]
        weights = reliability[selected]
        weighted_gene_spatial = float(np.average(gene_spatial[selected], weights=weights))

        patch_correlations: list[float] = []
        for patch in range(v.shape[1]):
            vx = v[selected, patch]
            xx = x[selected, patch]
            v_centered = vx - float(np.average(vx, weights=weights))
            x_centered = xx - float(np.average(xx, weights=weights))
            denominator = math.sqrt(
                float(np.dot(weights, v_centered * v_centered))
                * float(np.dot(weights, x_centered * x_centered))
            )
            patch_correlations.append(
                float(np.dot(weights, v_centered * x_centered) / denominator)
                if denominator > 0
                else 0.0
            )
        rows.append(
            {
                "method": name,
                "gene_count": rank,
                "target_reliability_mass_fraction": float(target),
                "reliability_mass_fraction": float(cumulative_weight[rank - 1] / total_weight),
                "reliability_weighted_gene_spatial_pearson": weighted_gene_spatial,
                "median_reliability_weighted_patch_profile_pearson": float(np.median(patch_correlations)),
            }
        )

    fractions = np.asarray([row["target_reliability_mass_fraction"] for row in rows], dtype=float)
    spatial_scores = np.asarray(
        [row["reliability_weighted_gene_spatial_pearson"] for row in rows], dtype=float
    )
    profile_scores = np.asarray(
        [row["median_reliability_weighted_patch_profile_pearson"] for row in rows], dtype=float
    )
    width = fractions[-1] - fractions[0]
    spatial_auc = trapezoid(spatial_scores, fractions) / width
    profile_auc = trapezoid(profile_scores, fractions) / width
    return rows, spatial_auc, profile_auc


def reference_gene_metric_rows(
    name: str,
    gene_set_name: str,
    visium_counts: np.ndarray,
    xenium_counts: np.ndarray,
    gene_ids: list[str],
    gene_names: dict[str, str],
    cohort: np.ndarray,
    reliability: np.ndarray,
    selection_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Emit the complete declared biological subset, retaining zero scores."""
    v_raw = visium_counts[:, cohort]
    x_raw = xenium_counts[:, cohort]
    correlations = rowwise_pearson(log_normalize(v_raw), log_normalize(x_raw))
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(selection_mask):
        defined = bool(np.isfinite(correlations[index]))
        rows.append(
            {
                "gene_set": gene_set_name,
                "method": name,
                "gene_id": gene_ids[index],
                "gene_name": gene_names.get(gene_ids[index], gene_ids[index]),
                "xenium_counts": float(x_raw[index].sum()),
                "visium_counts": float(v_raw[index].sum()),
                "xenium_reliability": float(reliability[index]),
                "spatial_pearson_defined": int(defined),
                "spatial_pearson_zero_if_undefined": float(correlations[index]) if defined else 0.0,
            }
        )
    return rows


def marker_sets(diffexp: pd.DataFrame, gene_ids: list[str], markers_per_cluster: int) -> dict[int, list[int]]:
    id_to_row = {value: index for index, value in enumerate(gene_ids)}
    result: dict[int, list[int]] = {}
    cluster_ids = sorted(
        int(match.group(1))
        for column in diffexp.columns
        if (match := re.fullmatch(r"Cluster (\d+) Log2 fold change", column))
    )
    for cluster in cluster_ids:
        mean_col = f"Cluster {cluster} Mean Counts"
        fold_col = f"Cluster {cluster} Log2 fold change"
        p_col = f"Cluster {cluster} Adjusted p value"
        candidates = diffexp[
            (diffexp["Feature ID"].astype(str).isin(id_to_row))
            & (diffexp[fold_col] > 0)
            & (diffexp[mean_col] > 0)
        ].copy()
        # Some otherwise well-populated official graph clusters have no genes
        # below the vendor table's adjusted-p threshold.  A p-value gate would
        # silently remove those domains.  Rank all positive effects instead,
        # balancing specificity and abundance so that very rare high-LFC genes
        # do not dominate the spatial module.
        candidates["_marker_rank"] = candidates[fold_col] * np.log1p(candidates[mean_col])
        candidates = candidates.sort_values(
            ["_marker_rank", fold_col, mean_col, "Feature ID"],
            ascending=[False, False, False, True],
        )
        selected = candidates.head(markers_per_cluster)["Feature ID"].astype(str).tolist()
        if len(selected) < min(5, markers_per_cluster):
            raise ComparisonError(f"cluster {cluster} has only {len(selected)} usable marker genes")
        result[cluster] = [id_to_row[value] for value in selected]
    return result


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return np.divide(matrix - mean, std, out=np.zeros_like(matrix), where=std > 0)


def evaluate_method(
    name: str,
    visium_counts: np.ndarray,
    xenium_counts: np.ndarray,
    gene_ids: list[str],
    cohort: np.ndarray,
    marker_rows: dict[int, list[int]],
    cluster_counts: np.ndarray,
    cluster_ids: list[int],
    min_gene_counts: float,
    min_dominant_patches: int,
    mass: dict[str, float],
    eligible_gene_mask: np.ndarray | None = None,
    gene_reliability: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    v_raw = visium_counts[:, cohort]
    x_raw = xenium_counts[:, cohort]
    mass = dict(mass)
    mass["cohort_panel_mass"] = float(v_raw.sum())
    v = log_normalize(v_raw)
    x = log_normalize(x_raw)
    if eligible_gene_mask is None:
        eligible_genes = (
            (v_raw.sum(axis=1) >= min_gene_counts)
            & (x_raw.sum(axis=1) >= min_gene_counts)
            & (v.std(axis=1) > 0)
            & (x.std(axis=1) > 0)
        )
    else:
        eligible_genes = np.asarray(eligible_gene_mask, dtype=bool)
        if eligible_genes.shape != (len(gene_ids),):
            raise ComparisonError("fixed eligible-gene mask does not match the gene axis")
    gene_metrics: list[dict[str, Any]] = []
    for index in np.flatnonzero(eligible_genes):
        gene_metrics.append(
            {
                "method": name,
                "gene_id": gene_ids[index],
                "visium_counts": float(v_raw[index].sum()),
                "xenium_counts": float(x_raw[index].sum()),
                "spatial_pearson": safe_pearson(v[index], x[index]),
                "spatial_spearman": safe_spearman(v[index], x[index]),
            }
        )
    patch_profile = [safe_pearson(v[eligible_genes, i], x[eligible_genes, i]) for i in range(v.shape[1])]
    gene_total_pearson = safe_pearson(np.log1p(v_raw.sum(axis=1)), np.log1p(x_raw.sum(axis=1)))
    gene_total_spearman = safe_spearman(v_raw.sum(axis=1), x_raw.sum(axis=1))

    v_z = zscore_rows(v)
    cohort_cluster_counts = cluster_counts[:, cohort]
    cluster_fractions = np.divide(
        cohort_cluster_counts,
        cohort_cluster_counts.sum(axis=0, keepdims=True),
        out=np.zeros_like(cohort_cluster_counts, dtype=float),
        where=cohort_cluster_counts.sum(axis=0, keepdims=True) > 0,
    )
    dominant = np.argmax(cohort_cluster_counts, axis=0)
    cluster_metrics: list[dict[str, Any]] = []
    scores: list[np.ndarray] = []
    for row, cluster in enumerate(cluster_ids):
        score = v_z[marker_rows[cluster]].mean(axis=0)
        scores.append(score)
        label = dominant == row
        positive_count = int(label.sum())
        dominant_eligible = (
            positive_count >= min_dominant_patches
            and len(label) - positive_count >= min_dominant_patches
        )
        auc = float(roc_auc_score(label, score)) if dominant_eligible else math.nan
        ap = float(average_precision_score(label, score)) if dominant_eligible else math.nan
        cluster_metrics.append(
            {
                "method": name,
                "cluster": cluster,
                "marker_gene_count": len(marker_rows[cluster]),
                "dominant_patch_count": positive_count,
                "dominant_metric_eligible": int(dominant_eligible),
                "fraction_pearson": safe_pearson(score, cluster_fractions[row]),
                "fraction_spearman": safe_spearman(score, cluster_fractions[row]),
                "dominant_roc_auc": auc,
                "dominant_average_precision": ap,
            }
        )
    score_matrix = np.vstack(scores)
    predicted = np.argmax(score_matrix, axis=0)
    summary = {
        "method": name,
        **mass,
        "comparison_patch_count": int(np.count_nonzero(cohort)),
        "eligible_gene_count": int(np.count_nonzero(eligible_genes)),
        "gene_total_log1p_pearson": gene_total_pearson,
        "gene_total_spearman": gene_total_spearman,
        "median_gene_spatial_pearson": float(np.nanmedian([x["spatial_pearson"] for x in gene_metrics])),
        "median_gene_spatial_spearman": float(np.nanmedian([x["spatial_spearman"] for x in gene_metrics])),
        "median_patch_profile_pearson": float(np.nanmedian(patch_profile)),
        "macro_cluster_fraction_pearson": float(np.nanmean([x["fraction_pearson"] for x in cluster_metrics])),
        "macro_cluster_fraction_spearman": float(np.nanmean([x["fraction_spearman"] for x in cluster_metrics])),
        "macro_cluster_dominant_roc_auc": float(np.nanmean([x["dominant_roc_auc"] for x in cluster_metrics])),
        "macro_cluster_dominant_average_precision": float(
            np.nanmean([x["dominant_average_precision"] for x in cluster_metrics])
        ),
        "evaluated_dominant_cluster_count": int(
            sum(x["dominant_metric_eligible"] for x in cluster_metrics)
        ),
        "dominant_cluster_accuracy": float(np.mean(predicted == dominant)),
    }
    if gene_reliability is not None:
        weighted_gene, weighted_patch = reliability_weighted_metrics(
            visium_counts,
            xenium_counts,
            cohort,
            gene_reliability,
        )
        summary["xenium_reliability_weighted_gene_spatial_pearson"] = weighted_gene
        summary["xenium_reliability_weighted_patch_profile_pearson"] = weighted_patch
    return summary, gene_metrics, cluster_metrics


def fixed_eligible_gene_mask(
    method_counts: dict[str, np.ndarray],
    xenium_counts: np.ndarray,
    cohort: np.ndarray,
    min_gene_counts: float,
) -> np.ndarray:
    x_raw = xenium_counts[:, cohort]
    x_normalized = log_normalize(x_raw)
    eligible = (x_raw.sum(axis=1) >= min_gene_counts) & (x_normalized.std(axis=1) > 0)
    del x_normalized
    for counts in method_counts.values():
        raw = counts[:, cohort]
        normalized = log_normalize(raw)
        eligible &= (raw.sum(axis=1) >= min_gene_counts) & (normalized.std(axis=1) > 0)
        del normalized
    if not np.any(eligible):
        raise ComparisonError("no genes pass the fixed cross-method eligibility rule")
    return eligible


def reference_ordered_gene_coverage_curve(
    name: str,
    visium_counts: np.ndarray,
    xenium_counts: np.ndarray,
    gene_ids: list[str],
    cohort: np.ndarray,
    steps: int = 20,
) -> tuple[list[dict[str, Any]], float, float]:
    if steps < 2:
        raise ComparisonError("gene-coverage curve requires at least two steps")
    v = log_normalize(visium_counts[:, cohort])
    x = log_normalize(xenium_counts[:, cohort])
    gene_spatial_correlations = rowwise_pearson(v, x)
    reference_totals = xenium_counts[:, cohort].sum(axis=1)
    order = np.asarray(
        sorted(range(len(gene_ids)), key=lambda index: (-reference_totals[index], gene_ids[index])),
        dtype=np.int64,
    )
    requested = np.linspace(0.05, 1.0, steps)
    counts = sorted(set(max(3, int(math.ceil(fraction * len(gene_ids)))) for fraction in requested))
    count_set = set(counts)
    sx = np.zeros(v.shape[1], dtype=np.float64)
    sy = np.zeros(v.shape[1], dtype=np.float64)
    sxx = np.zeros(v.shape[1], dtype=np.float64)
    syy = np.zeros(v.shape[1], dtype=np.float64)
    sxy = np.zeros(v.shape[1], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for rank, gene_index in enumerate(order, start=1):
        vx = v[gene_index]
        xy = x[gene_index]
        sx += vx
        sy += xy
        sxx += vx * vx
        syy += xy * xy
        sxy += vx * xy
        if rank not in count_set:
            continue
        covariance = sxy - sx * sy / rank
        variance_v = np.maximum(sxx - sx * sx / rank, 0.0)
        variance_x = np.maximum(syy - sy * sy / rank, 0.0)
        denominator = np.sqrt(variance_v * variance_x)
        correlations = np.divide(
            covariance,
            denominator,
            out=np.full_like(covariance, np.nan),
            where=denominator > 0,
        )
        rows.append(
            {
                "method": name,
                "gene_count": rank,
                "gene_coverage_fraction": rank / len(gene_ids),
                "median_patch_profile_pearson": float(np.nanmedian(correlations)),
                "median_gene_spatial_pearson": float(
                    np.nanmedian(gene_spatial_correlations[order[:rank]])
                ),
            }
        )
    fractions = np.asarray([row["gene_coverage_fraction"] for row in rows], dtype=float)
    profile_correlations = np.asarray([row["median_patch_profile_pearson"] for row in rows], dtype=float)
    spatial_correlations = np.asarray([row["median_gene_spatial_pearson"] for row in rows], dtype=float)
    width = fractions[-1] - fractions[0]
    profile_auc = trapezoid(profile_correlations, fractions) / width
    spatial_auc = trapezoid(spatial_correlations, fractions) / width
    return rows, profile_auc, spatial_auc


def evaluate_matrix_delta(
    name: str,
    policy_counts: np.ndarray,
    sr_counts: np.ndarray,
    xenium_counts: np.ndarray,
    gene_ids: list[str],
    cohort: np.ndarray,
    marker_rows: dict[int, list[int]],
    cluster_counts: np.ndarray,
    cluster_ids: list[int],
    min_gene_counts: float,
    min_dominant_patches: int,
) -> dict[str, Any]:
    positive = np.maximum(policy_counts - sr_counts, 0.0)
    negative = np.maximum(sr_counts - policy_counts, 0.0)
    intersection = np.minimum(policy_counts, sr_counts)
    delta_eligible = fixed_eligible_gene_mask(
        {"positive_delta": positive, "matrix_intersection": intersection},
        xenium_counts,
        cohort,
        min_gene_counts,
    )
    positive_summary, _, _ = evaluate_method(
        f"{name}:positive_matrix_delta",
        positive,
        xenium_counts,
        gene_ids,
        cohort,
        marker_rows,
        cluster_counts,
        cluster_ids,
        min_gene_counts,
        min_dominant_patches,
        {"total_mass": float(positive.sum())},
        delta_eligible,
    )
    intersection_summary, _, _ = evaluate_method(
        f"{name}:matrix_intersection",
        intersection,
        xenium_counts,
        gene_ids,
        cohort,
        marker_rows,
        cluster_counts,
        cluster_ids,
        min_gene_counts,
        min_dominant_patches,
        {"total_mass": float(intersection.sum())},
        delta_eligible,
    )
    row: dict[str, Any] = {
        "policy": name,
        "matrix_delta_unit": "shared_gene_by_registered_patch",
        "all_patch_positive_delta_mass": float(positive.sum()),
        "all_patch_negative_delta_mass": float(negative.sum()),
        "all_patch_net_delta_mass": float(policy_counts.sum() - sr_counts.sum()),
        "cohort_positive_delta_mass": float(positive[:, cohort].sum()),
        "cohort_negative_delta_mass": float(negative[:, cohort].sum()),
        "cohort_net_delta_mass": float(policy_counts[:, cohort].sum() - sr_counts[:, cohort].sum()),
        "cohort_matrix_intersection_mass": float(intersection[:, cohort].sum()),
    }
    for metric in SHAPE_METRICS:
        row[f"positive_delta_{metric}"] = positive_summary[metric]
        row[f"matrix_intersection_{metric}"] = intersection_summary[metric]
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ComparisonError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate with both NumPy 1.x and NumPy 2.x."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return float(implementation(values, coordinates))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ComparisonError(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_paths = parse_policy_args(args.policy_mex)
    input_paths = [
        args.sr_h5,
        args.positions,
        args.scalefactors,
        args.xenium_h5,
        args.xenium_cells,
        args.xenium_clusters,
        args.xenium_diffexp,
        args.xenium_experiment,
        args.xenium_he_alignment,
        args.registration,
    ]
    if args.reference_panel_json is not None:
        input_paths.append(args.reference_panel_json)
    for path in input_paths:
        if not path.is_file():
            raise ComparisonError(f"missing input: {path}")

    scalefactors = json.loads(args.scalefactors.read_text(encoding="utf-8"))
    microns_per_pixel = float(scalefactors["microns_per_pixel"])
    positions = pd.read_parquet(args.positions)
    required_position_columns = {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    if not required_position_columns.issubset(positions.columns):
        raise ComparisonError("Visium tissue positions do not have the required columns")

    experiment = json.loads(args.xenium_experiment.read_text(encoding="utf-8"))
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    if registration.get("status") != "pass" or registration.get("interpretation", {}).get("expression_used") is not False:
        raise ComparisonError("registration must be passing and expression-blind")
    he_alignment = np.loadtxt(args.xenium_he_alignment, delimiter=",")
    x_to_v = xenium_to_visium_transform(
        he_alignment,
        np.asarray(registration["moving_fullres_to_fixed_fullres"], dtype=float),
        float(experiment["pixel_size"]),
    )

    cells = pd.read_parquet(args.xenium_cells)
    if not {"cell_id", "x_centroid", "y_centroid"}.issubset(cells.columns):
        raise ComparisonError("Xenium cell table lacks centroid columns")
    mapped_x, mapped_y = transform_points(
        cells["x_centroid"].to_numpy(dtype=float),
        cells["y_centroid"].to_numpy(dtype=float),
        x_to_v,
    )
    mapped_x += args.xenium_offset_x_um / microns_per_pixel
    mapped_y += args.xenium_offset_y_um / microns_per_pixel
    position_keys = patch_keys(
        positions["pxl_col_in_fullres"].to_numpy(dtype=float),
        positions["pxl_row_in_fullres"].to_numpy(dtype=float),
        microns_per_pixel,
        args.patch_um,
    )
    cell_keys = patch_keys(mapped_x, mapped_y, microns_per_pixel, args.patch_um)
    patch_axis, patch_lookup = make_patch_axis(position_keys, cell_keys)
    positions["_patch_index"] = mapping_vector(position_keys, patch_lookup)
    cell_groups = mapping_vector(cell_keys, patch_lookup)
    n_patches = len(patch_axis)

    tissue_bin_counts = np.bincount(
        positions.loc[positions["in_tissue"].astype(bool), "_patch_index"].to_numpy(dtype=np.int64),
        minlength=n_patches,
    )
    xenium_cell_counts = np.bincount(cell_groups, minlength=n_patches)
    cohort = (tissue_bin_counts >= args.min_visium_tissue_bins) & (xenium_cell_counts >= args.min_xenium_cells)
    if np.count_nonzero(cohort) < 20:
        raise ComparisonError(f"only {np.count_nonzero(cohort)} comparison patches passed fixed cohort gates")

    clusters = pd.read_csv(args.xenium_clusters)
    cluster_by_cell = dict(zip(clusters["Barcode"].astype(str), clusters["Cluster"].astype(int)))
    cluster_ids = sorted(set(cluster_by_cell.values()))
    cluster_row = {value: index for index, value in enumerate(cluster_ids)}
    cluster_counts = np.zeros((len(cluster_ids), n_patches), dtype=np.int32)
    for cell_id, patch in zip(cells["cell_id"].astype(str), cell_groups):
        cluster = cluster_by_cell.get(cell_id)
        if cluster is not None:
            cluster_counts[cluster_row[cluster], patch] += 1

    diffexp_all = pd.read_csv(args.xenium_diffexp)
    panel_gene_ids = diffexp_all["Feature ID"].astype(str).tolist()
    if len(panel_gene_ids) != len(set(panel_gene_ids)):
        raise ComparisonError("Xenium differential-expression gene axis contains duplicates")
    source_feature_sets: dict[str, set[str]] = {
        "xenium": set(read_h5_feature_ids(args.xenium_h5)),
        "space_ranger": set(read_h5_feature_ids(args.sr_h5)),
    }
    for name, directory in policy_paths:
        source_feature_sets[name] = set(read_mex_feature_ids(directory))
    target_gene_ids = [
        gene_id for gene_id in panel_gene_ids if all(gene_id in values for values in source_feature_sets.values())
    ]
    excluded_features = [
        {
            "gene_id": gene_id,
            "missing_from": ",".join(name for name, values in source_feature_sets.items() if gene_id not in values),
        }
        for gene_id in panel_gene_ids
        if gene_id not in set(target_gene_ids)
    ]
    if not target_gene_ids:
        raise ComparisonError("the matrices have no shared Xenium panel genes")
    diffexp = diffexp_all[diffexp_all["Feature ID"].astype(str).isin(target_gene_ids)].copy()
    diffexp["Feature ID"] = pd.Categorical(diffexp["Feature ID"].astype(str), categories=target_gene_ids, ordered=True)
    diffexp = diffexp.sort_values("Feature ID").reset_index(drop=True)
    diffexp["Feature ID"] = diffexp["Feature ID"].astype(str)
    gene_name_by_id = dict(zip(diffexp["Feature ID"].astype(str), diffexp["Feature Name"].astype(str)))
    reference_gene_names: set[str] | None = None
    reference_gene_mask: np.ndarray | None = None
    if args.reference_panel_json is not None:
        panel_payload = json.loads(args.reference_panel_json.read_text(encoding="utf-8"))["payload"]
        reference_gene_names = {
            str(target["type"]["data"]["name"])
            for target in panel_payload["targets"]
            if target.get("source", {}).get("category") == args.reference_panel_source_category
            and target.get("type", {}).get("descriptor") == "gene"
        }
        if not reference_gene_names:
            raise ComparisonError(
                f"reference panel has no genes in source category {args.reference_panel_source_category!r}"
            )
        reference_gene_mask = np.asarray(
            [gene_name_by_id.get(gene_id, gene_id) in reference_gene_names for gene_id in target_gene_ids],
            dtype=bool,
        )
        if np.count_nonzero(reference_gene_mask) < 3:
            raise ComparisonError("fewer than three reference-panel genes occur on the shared axis")
    markers = marker_sets(diffexp, target_gene_ids, args.markers_per_cluster)
    if sorted(markers) != cluster_ids:
        raise ComparisonError("Xenium cluster and differential-expression axes disagree")

    xenium_matrix, xenium_features, xenium_barcodes = read_10x_h5(args.xenium_h5)
    if xenium_barcodes != cells["cell_id"].astype(str).tolist():
        raise ComparisonError("Xenium matrix and cell table axes disagree")
    xenium_counts, xenium_mass = aggregate_matrix(
        xenium_matrix,
        xenium_features,
        target_gene_ids,
        cell_groups,
        n_patches,
    )
    del xenium_matrix
    gc.collect()
    split_correlation, gene_reliability = xenium_split_half_reliability(
        xenium_counts,
        cohort,
        args.xenium_split_seed,
    )
    xenium_gene_totals = xenium_counts[:, cohort].sum(axis=1)
    reliability_rows = [
        {
            "gene_id": gene_id,
            "gene_name": gene_name_by_id.get(gene_id, gene_id),
            "xenium_cohort_counts": float(xenium_gene_totals[index]),
            "split_half_spatial_pearson": float(split_correlation[index]),
            "spearman_brown_reliability": float(gene_reliability[index]),
        }
        for index, gene_id in enumerate(target_gene_ids)
    ]

    method_counts: dict[str, np.ndarray] = {}
    method_masses: dict[str, dict[str, float]] = {}
    sr_matrix, sr_features, sr_barcodes = read_10x_h5(args.sr_h5)
    sr_groups = visium_column_groups(sr_barcodes, positions, patch_lookup)
    sr_counts, sr_mass = aggregate_matrix(sr_matrix, sr_features, target_gene_ids, sr_groups, n_patches)
    method_counts["space_ranger"] = sr_counts
    method_masses["space_ranger"] = sr_mass
    del sr_matrix
    gc.collect()

    for name, directory in policy_paths:
        matrix, features, barcodes = read_mex(directory)
        groups = visium_column_groups(barcodes, positions, patch_lookup)
        counts, mass = aggregate_matrix(matrix, features, target_gene_ids, groups, n_patches)
        method_counts[name] = counts
        method_masses[name] = mass
        del matrix
        gc.collect()

    common_eligible_genes = fixed_eligible_gene_mask(
        method_counts,
        xenium_counts,
        cohort,
        args.min_gene_counts,
    )
    summaries: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    matrix_delta_rows: list[dict[str, Any]] = []
    gene_coverage_rows: list[dict[str, Any]] = []
    reliability_mass_rows: list[dict[str, Any]] = []
    reference_gene_set_rows: list[dict[str, Any]] = []
    reference_gene_set_curve_rows: list[dict[str, Any]] = []
    for name, counts in method_counts.items():
        summary, genes, cluster_metrics = evaluate_method(
            name,
            counts,
            xenium_counts,
            target_gene_ids,
            cohort,
            markers,
            cluster_counts,
            cluster_ids,
            args.min_gene_counts,
            args.min_dominant_patches,
            method_masses[name],
            common_eligible_genes,
            gene_reliability,
        )
        summaries.append(summary)
        gene_rows.extend(genes)
        cluster_rows.extend(cluster_metrics)
        coverage_rows, profile_coverage_auc, spatial_coverage_auc = reference_ordered_gene_coverage_curve(
            name,
            counts,
            xenium_counts,
            target_gene_ids,
            cohort,
        )
        summary["gene_profile_coverage_auc_5_to_100"] = profile_coverage_auc
        summary["gene_spatial_coverage_auc_5_to_100"] = spatial_coverage_auc
        summary["full_shared_axis_patch_profile_pearson"] = coverage_rows[-1][
            "median_patch_profile_pearson"
        ]
        summary["full_shared_axis_median_gene_spatial_pearson"] = coverage_rows[-1][
            "median_gene_spatial_pearson"
        ]
        gene_coverage_rows.extend(coverage_rows)
        reliability_rows_for_method, reliability_spatial_auc, reliability_profile_auc = (
            xenium_reliability_mass_curve(
                name,
                counts,
                xenium_counts,
                target_gene_ids,
                cohort,
                gene_reliability,
            )
        )
        summary["xenium_reliability_mass_gene_spatial_auc_5_to_100"] = reliability_spatial_auc
        summary["xenium_reliability_mass_patch_profile_auc_5_to_100"] = reliability_profile_auc
        reliability_mass_rows.extend(reliability_rows_for_method)
        if reference_gene_mask is not None:
            reference_curve, reference_spatial_auc, reference_profile_auc = xenium_reliability_mass_curve(
                name,
                counts,
                xenium_counts,
                target_gene_ids,
                cohort,
                gene_reliability,
                selection_mask=reference_gene_mask,
            )
            for row in reference_curve:
                row["gene_set"] = args.reference_gene_set_name
            summary["reference_gene_set_reliability_mass_gene_spatial_auc_5_to_100"] = (
                reference_spatial_auc
            )
            summary["reference_gene_set_reliability_mass_patch_profile_auc_5_to_100"] = (
                reference_profile_auc
            )
            reference_gene_set_curve_rows.extend(reference_curve)
            reference_gene_set_rows.extend(
                reference_gene_metric_rows(
                    name,
                    args.reference_gene_set_name,
                    counts,
                    xenium_counts,
                    target_gene_ids,
                    gene_name_by_id,
                    cohort,
                    gene_reliability,
                    reference_gene_mask,
                )
            )
        if name != "space_ranger":
            matrix_delta_rows.append(
                evaluate_matrix_delta(
                    name,
                    counts,
                    sr_counts,
                    xenium_counts,
                    target_gene_ids,
                    cohort,
                    markers,
                    cluster_counts,
                    cluster_ids,
                    args.min_gene_counts,
                    args.min_dominant_patches,
                )
            )

    sr_summary = summaries[0]
    for row in summaries:
        row["total_mass_ratio_vs_space_ranger"] = row["total_mass"] / sr_summary["total_mass"]
        row["cohort_panel_mass_ratio_vs_space_ranger"] = (
            row["cohort_panel_mass"] / sr_summary["cohort_panel_mass"]
        )

    patch_rows = [
        {
            "patch_index": index,
            "patch_x": key[0],
            "patch_y": key[1],
            "x_start_um": key[0] * args.patch_um,
            "y_start_um": key[1] * args.patch_um,
            "visium_tissue_bin_count": int(tissue_bin_counts[index]),
            "xenium_cell_count": int(xenium_cell_counts[index]),
            "xenium_clustered_cell_count": int(cluster_counts[:, index].sum()),
            "in_comparison_cohort": int(cohort[index]),
        }
        for index, key in enumerate(patch_axis)
    ]
    write_tsv(args.out_dir / "method_summary.tsv", summaries)
    write_tsv(args.out_dir / "gene_spatial_metrics.tsv", gene_rows)
    write_tsv(args.out_dir / "cluster_domain_metrics.tsv", cluster_rows)
    write_tsv(args.out_dir / "patch_cohort.tsv", patch_rows)
    write_tsv(args.out_dir / "gene_coverage_curve.tsv", gene_coverage_rows)
    write_tsv(args.out_dir / "xenium_reliability_mass_curve.tsv", reliability_mass_rows)
    if reference_gene_set_rows:
        write_tsv(args.out_dir / "reference_gene_set_metrics.tsv", reference_gene_set_rows)
        write_tsv(args.out_dir / "reference_gene_set_reliability_mass_curve.tsv", reference_gene_set_curve_rows)
    write_tsv(args.out_dir / "xenium_gene_reliability.tsv", reliability_rows)
    if matrix_delta_rows:
        write_tsv(args.out_dir / "matrix_delta_summary.tsv", matrix_delta_rows)
    if excluded_features:
        write_tsv(args.out_dir / "excluded_panel_features.tsv", excluded_features)

    input_hashes = {str(path): sha256(path) for path in input_paths}
    for name, directory in policy_paths:
        for filename in ("matrix.mtx", "features.tsv", "barcodes.tsv"):
            path = _resolve_mex_file(directory, filename)
            input_hashes[str(path)] = sha256(path)
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "interpretation": {
            "xenium_role": "adjacent-section orthogonal spatial reference",
            "xenium_clusters": "official unnamed graph-cluster domains, not authoritative cell types",
            "registration_scope": "coarse adjacent-section comparison; not cell-to-cell correspondence",
            "raw_mass_normalized": False,
            "expression_used_for_registration_or_cohort": False,
        },
        "parameters": {
            "patch_um": args.patch_um,
            "xenium_offset_x_um": args.xenium_offset_x_um,
            "xenium_offset_y_um": args.xenium_offset_y_um,
            "min_xenium_cells": args.min_xenium_cells,
            "min_visium_tissue_bins": args.min_visium_tissue_bins,
            "markers_per_cluster": args.markers_per_cluster,
            "min_dominant_patches": args.min_dominant_patches,
            "min_gene_counts": args.min_gene_counts,
            "xenium_split_seed": args.xenium_split_seed,
            "gene_eligibility": "fixed intersection across Xenium, Space Ranger, and every STAR policy",
            "gene_coverage_curve": "Xenium-abundance order on the full shared axis; 5% to 100% in 20 steps",
            "xenium_reliability_mass_curve": (
                "Xenium split-half reliability order and weight; identical genes by cumulative reliability "
                "mass; missing or constant method signal scores zero"
            ),
            "reference_gene_set_name": args.reference_gene_set_name if reference_gene_mask is not None else None,
            "reference_panel_source_category": (
                args.reference_panel_source_category if reference_gene_mask is not None else None
            ),
        },
        "axes": {
            "xenium_panel_gene_count": len(panel_gene_ids),
            "shared_gene_count": len(target_gene_ids),
            "fixed_eligible_gene_count": int(np.count_nonzero(common_eligible_genes)),
            "excluded_panel_gene_count": len(excluded_features),
            "xenium_cell_count": len(cells),
            "xenium_clustered_cell_count": len(clusters),
            "xenium_cluster_count": len(cluster_ids),
            "patch_count": n_patches,
            "comparison_patch_count": int(np.count_nonzero(cohort)),
            "reference_panel_requested_gene_count": len(reference_gene_names) if reference_gene_names else None,
            "reference_panel_shared_gene_count": (
                int(np.count_nonzero(reference_gene_mask)) if reference_gene_mask is not None else None
            ),
            "reference_panel_positive_reliability_gene_count": (
                int(np.count_nonzero(reference_gene_mask & (gene_reliability > 0)))
                if reference_gene_mask is not None
                else None
            ),
        },
        "registration": {
            "source_status": registration["status"],
            "source_mask_iou": registration["overlap"]["iou"],
            "xenium_microns_to_visium_fullres": x_to_v.tolist(),
        },
        "xenium_mass": xenium_mass,
        "xenium_reliability": {
            "weight_sum": float(gene_reliability.sum()),
            "median": float(np.median(gene_reliability)),
            "genes_at_least_0_5": int(np.count_nonzero(gene_reliability >= 0.5)),
            "genes_at_least_0_8": int(np.count_nonzero(gene_reliability >= 0.8)),
        },
        "methods": summaries,
        "input_sha256": input_hashes,
        "artifacts": {
            "method_summary": "method_summary.tsv",
            "gene_spatial_metrics": "gene_spatial_metrics.tsv",
            "cluster_domain_metrics": "cluster_domain_metrics.tsv",
            "patch_cohort": "patch_cohort.tsv",
            "gene_coverage_curve": "gene_coverage_curve.tsv",
            "xenium_reliability_mass_curve": "xenium_reliability_mass_curve.tsv",
            "xenium_gene_reliability": "xenium_gene_reliability.tsv",
            "reference_gene_set_metrics": (
                "reference_gene_set_metrics.tsv" if reference_gene_set_rows else None
            ),
            "reference_gene_set_reliability_mass_curve": (
                "reference_gene_set_reliability_mass_curve.tsv" if reference_gene_set_curve_rows else None
            ),
            "matrix_delta_summary": "matrix_delta_summary.tsv" if matrix_delta_rows else None,
            "excluded_panel_features": "excluded_panel_features.tsv" if excluded_features else None,
        },
    }
    result = json_safe(result)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    try:
        result = run(parse_args())
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps({"status": result["status"], "axes": result["axes"], "artifacts": result["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
