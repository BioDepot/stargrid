#!/usr/bin/env python3
"""Compare ovarian-panel differential signal in Xenium-defined cancer regions.

The cancer-cell label and the registered patch cohort are defined only from
Xenium and morphology-derived registration.  Every Visium method is evaluated
on the same cancer-rich and cancer-poor patches.  Raw mass is retained; shape
metrics use within-method shared-axis library normalization and rank AUCs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score

from compare_visium_to_adjacent_xenium import (
    ComparisonError,
    aggregate_matrix,
    make_patch_axis,
    mapping_vector,
    parse_policy_args,
    patch_keys,
    read_10x_h5,
    read_h5_feature_ids,
    read_mex,
    read_mex_feature_ids,
    transform_points,
    visium_column_groups,
    xenium_to_visium_transform,
)


SCHEMA = "spatial_suite.ovarian_panel_xenium_cancer_cell_comparison.v1"
DEFAULT_CANCER_CLUSTERS = (1, 2, 3, 5, 6, 8, 9, 14)
DEFAULT_LABEL_GENES = ("MUC16", "PAX8", "EPCAM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-h5", required=True, type=Path)
    parser.add_argument("--policy-mex", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--scalefactors", required=True, type=Path)
    parser.add_argument("--xenium-h5", required=True, type=Path)
    parser.add_argument("--xenium-cells", required=True, type=Path)
    parser.add_argument("--xenium-clusters", required=True, type=Path)
    parser.add_argument("--xenium-diffexp", required=True, type=Path)
    parser.add_argument("--xenium-experiment", required=True, type=Path)
    parser.add_argument("--xenium-he-alignment", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--reference-panel-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patch-um", type=float, default=128.0)
    parser.add_argument("--min-xenium-cells", type=int, default=10)
    parser.add_argument("--min-visium-tissue-bins", type=int, default=16)
    parser.add_argument("--cancer-rich-min-fraction", type=float, default=0.75)
    parser.add_argument("--cancer-poor-max-fraction", type=float, default=0.25)
    parser.add_argument(
        "--cancer-clusters",
        default=",".join(str(value) for value in DEFAULT_CANCER_CLUSTERS),
    )
    parser.add_argument("--label-genes", default=",".join(DEFAULT_LABEL_GENES))
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item) for item in value.split(",") if item}))
    except ValueError as exc:
        raise ComparisonError(f"invalid comma-separated integer list: {value!r}") from exc
    if not result:
        raise ComparisonError("cancer cluster list is empty")
    return result


def comma_names(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not result:
        raise ComparisonError("label gene list is empty")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: Any) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def cancer_cell_membership(
    matrix: sparse.spmatrix,
    feature_names: list[str],
    cell_clusters: np.ndarray,
    cancer_clusters: tuple[int, ...],
    label_genes: tuple[str, ...],
) -> np.ndarray:
    if matrix.shape[0] != len(feature_names) or matrix.shape[1] != len(cell_clusters):
        raise ComparisonError("Xenium feature or cell axis is inconsistent")
    lookup: dict[str, int] = {}
    for index, name in enumerate(feature_names):
        if name in lookup:
            raise ComparisonError(f"duplicate Xenium feature name: {name}")
        lookup[name] = index
    missing = [gene for gene in label_genes if gene not in lookup]
    if missing:
        raise ComparisonError(f"Xenium matrix is missing label genes: {','.join(missing)}")
    rows = [lookup[gene] for gene in label_genes]
    positive = np.asarray(matrix[rows, :].sum(axis=0)).ravel() > 0
    return np.isin(cell_clusters, cancer_clusters) & positive


def log_normalize_shared_axis(counts: np.ndarray, scale: float = 10_000.0) -> np.ndarray:
    totals = counts.sum(axis=0)
    normalized = np.divide(
        counts,
        totals[None, :],
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals[None, :] > 0,
    )
    return np.log1p(scale * normalized)


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return math.nan
    return float(stats.pearsonr(first, second).statistic)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return math.nan
    return float(stats.spearmanr(first, second).statistic)


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=1, keepdims=True)
    standard = matrix.std(axis=1, keepdims=True)
    return np.divide(matrix - means, standard, out=np.zeros_like(matrix), where=standard > 0)


def evaluate_scope(
    scope: str,
    method_counts: dict[str, np.ndarray],
    xenium_counts: np.ndarray,
    gene_ids: list[str],
    gene_names: list[str],
    selected_genes: np.ndarray,
    cancer_rich: np.ndarray,
    cancer_poor: np.ndarray,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows = np.flatnonzero(selected_genes)
    if len(selected_rows) < 3:
        raise ComparisonError(f"scope {scope!r} has fewer than three shared genes")
    evaluation_patches = cancer_rich | cancer_poor
    labels = cancer_rich[evaluation_patches]
    if labels.sum() < 10 or (~labels).sum() < 10:
        raise ComparisonError(f"scope {scope!r} has too few cancer-rich or cancer-poor patches")

    all_counts = {"xenium": xenium_counts, **method_counts}
    normalized = {
        name: log_normalize_shared_axis(counts[:, evaluation_patches])
        for name, counts in all_counts.items()
    }
    reference = normalized["xenium"][selected_rows]
    reference_effect = reference[:, labels].mean(axis=1) - reference[:, ~labels].mean(axis=1)
    reference_positive = reference_effect > 0
    if not np.any(reference_positive):
        raise ComparisonError(f"scope {scope!r} has no Xenium-positive cancer genes")

    summaries: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    auc_by_method: dict[str, np.ndarray] = {}
    for name, counts in all_counts.items():
        values = normalized[name][selected_rows]
        effect = values[:, labels].mean(axis=1) - values[:, ~labels].mean(axis=1)
        auc = np.asarray([roc_auc_score(labels, row) for row in values], dtype=float)
        oriented_auc = np.where(reference_effect >= 0, auc, 1.0 - auc)
        auc_by_method[name] = oriented_auc

        signed_score = (zscore_rows(values) * np.sign(reference_effect)[:, None]).mean(axis=0)
        positive_score = zscore_rows(values[reference_positive]).mean(axis=0)
        panel_counts = counts[selected_rows]
        rich_mass = float(panel_counts[:, cancer_rich].sum())
        poor_mass = float(panel_counts[:, cancer_poor].sum())
        summary = {
            "scope": scope,
            "method": name,
            "gene_count": len(selected_rows),
            "xenium_positive_gene_count": int(reference_positive.sum()),
            "cancer_rich_patch_count": int(cancer_rich.sum()),
            "cancer_poor_patch_count": int(cancer_poor.sum()),
            "raw_cancer_rich_panel_mass": rich_mass,
            "raw_cancer_poor_panel_mass": poor_mass,
            "raw_cancer_rich_panel_mass_per_patch": rich_mass / cancer_rich.sum(),
            "raw_cancer_poor_panel_mass_per_patch": poor_mass / cancer_poor.sum(),
            "raw_cancer_rich_to_poor_mass_per_patch": (
                (rich_mass / cancer_rich.sum()) / (poor_mass / cancer_poor.sum())
                if poor_mass > 0
                else math.nan
            ),
            "effect_pearson_vs_xenium": safe_pearson(effect, reference_effect),
            "effect_spearman_vs_xenium": safe_spearman(effect, reference_effect),
            "mean_absolute_effect_error_vs_xenium": float(np.mean(np.abs(effect - reference_effect))),
            "effect_direction_agreement_vs_xenium": float(np.mean(effect * reference_effect > 0)),
            "mean_reference_oriented_gene_auc": float(np.mean(oriented_auc)),
            "median_reference_oriented_gene_auc": float(np.median(oriented_auc)),
            "mean_xenium_positive_gene_auc": float(np.mean(auc[reference_positive])),
            "median_xenium_positive_gene_auc": float(np.median(auc[reference_positive])),
            "signed_panel_score_auc": float(roc_auc_score(labels, signed_score)),
            "positive_panel_score_auc": float(roc_auc_score(labels, positive_score)),
        }
        summaries.append(summary)
        for local_index, global_index in enumerate(selected_rows):
            gene_rows.append(
                {
                    "scope": scope,
                    "method": name,
                    "gene_id": gene_ids[global_index],
                    "gene_name": gene_names[global_index],
                    "xenium_effect": float(reference_effect[local_index]),
                    "xenium_positive": int(reference_positive[local_index]),
                    "method_effect": float(effect[local_index]),
                    "raw_cancer_rich_mass": float(counts[global_index, cancer_rich].sum()),
                    "raw_cancer_poor_mass": float(counts[global_index, cancer_poor].sum()),
                    "cancer_patch_auc": float(auc[local_index]),
                    "reference_oriented_auc": float(oriented_auc[local_index]),
                }
            )

    sr_auc = auc_by_method["space_ranger"]
    sr_summary = next(row for row in summaries if row["method"] == "space_ranger")
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    for summary in summaries:
        method = summary["method"]
        delta = auc_by_method[method] - sr_auc
        positive_delta = delta[reference_positive]
        summary["raw_cancer_rich_panel_mass_ratio_vs_space_ranger"] = (
            summary["raw_cancer_rich_panel_mass"] / sr_summary["raw_cancer_rich_panel_mass"]
        )
        summary["raw_cancer_poor_panel_mass_ratio_vs_space_ranger"] = (
            summary["raw_cancer_poor_panel_mass"] / sr_summary["raw_cancer_poor_panel_mass"]
        )
        summary["mean_oriented_gene_auc_delta_vs_space_ranger"] = float(delta.mean())
        summary["median_oriented_gene_auc_delta_vs_space_ranger"] = float(np.median(delta))
        summary["gene_auc_wins_vs_space_ranger"] = int(np.count_nonzero(delta > 1e-12))
        summary["gene_auc_ties_vs_space_ranger"] = int(np.count_nonzero(np.abs(delta) <= 1e-12))
        summary["gene_auc_losses_vs_space_ranger"] = int(np.count_nonzero(delta < -1e-12))
        if method == "space_ranger" or bootstrap_replicates <= 0:
            low = high = float(delta.mean())
        else:
            samples = rng.integers(0, len(delta), size=(bootstrap_replicates, len(delta)))
            boot = delta[samples].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975]).tolist()
        summary["mean_oriented_gene_auc_delta_vs_space_ranger_ci95_low"] = float(low)
        summary["mean_oriented_gene_auc_delta_vs_space_ranger_ci95_high"] = float(high)
        summary["mean_positive_gene_auc_delta_vs_space_ranger"] = float(positive_delta.mean())
        summary["median_positive_gene_auc_delta_vs_space_ranger"] = float(
            np.median(positive_delta)
        )
        summary["positive_gene_auc_wins_vs_space_ranger"] = int(
            np.count_nonzero(positive_delta > 1e-12)
        )
        summary["positive_gene_auc_ties_vs_space_ranger"] = int(
            np.count_nonzero(np.abs(positive_delta) <= 1e-12)
        )
        summary["positive_gene_auc_losses_vs_space_ranger"] = int(
            np.count_nonzero(positive_delta < -1e-12)
        )
        if method == "space_ranger" or bootstrap_replicates <= 0:
            positive_low = positive_high = float(positive_delta.mean())
        else:
            samples = rng.integers(
                0,
                len(positive_delta),
                size=(bootstrap_replicates, len(positive_delta)),
            )
            positive_boot = positive_delta[samples].mean(axis=1)
            positive_low, positive_high = np.quantile(positive_boot, [0.025, 0.975]).tolist()
        summary["mean_positive_gene_auc_delta_vs_space_ranger_ci95_low"] = float(
            positive_low
        )
        summary["mean_positive_gene_auc_delta_vs_space_ranger_ci95_high"] = float(
            positive_high
        )
    return summaries, gene_rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ComparisonError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ComparisonError(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.cancer_poor_max_fraction < args.cancer_rich_min_fraction <= 1:
        raise ComparisonError("cancer fraction thresholds must satisfy 0 <= poor < rich <= 1")
    policy_paths = parse_policy_args(args.policy_mex)
    cancer_clusters = comma_ints(args.cancer_clusters)
    label_genes = comma_names(args.label_genes)
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
        args.reference_panel_json,
    ]
    for path in input_paths:
        if not path.is_file():
            raise ComparisonError(f"missing input: {path}")

    scalefactors = json.loads(args.scalefactors.read_text(encoding="utf-8"))
    microns_per_pixel = float(scalefactors["microns_per_pixel"])
    positions = pd.read_parquet(args.positions)
    cells = pd.read_parquet(args.xenium_cells)
    experiment = json.loads(args.xenium_experiment.read_text(encoding="utf-8"))
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    if registration.get("status") != "pass" or registration.get("interpretation", {}).get("expression_used") is not False:
        raise ComparisonError("registration must be passing and expression-blind")
    transform = xenium_to_visium_transform(
        np.loadtxt(args.xenium_he_alignment, delimiter=","),
        np.asarray(registration["moving_fullres_to_fixed_fullres"], dtype=float),
        float(experiment["pixel_size"]),
    )
    mapped_x, mapped_y = transform_points(
        cells["x_centroid"].to_numpy(float), cells["y_centroid"].to_numpy(float), transform
    )
    position_keys = patch_keys(
        positions["pxl_col_in_fullres"].to_numpy(float),
        positions["pxl_row_in_fullres"].to_numpy(float),
        microns_per_pixel,
        args.patch_um,
    )
    cell_keys = patch_keys(mapped_x, mapped_y, microns_per_pixel, args.patch_um)
    patch_axis, patch_lookup = make_patch_axis(position_keys, cell_keys)
    positions["_patch_index"] = mapping_vector(position_keys, patch_lookup)
    cell_groups = mapping_vector(cell_keys, patch_lookup)
    n_patches = len(patch_axis)
    tissue_counts = np.bincount(
        positions.loc[positions["in_tissue"].astype(bool), "_patch_index"].to_numpy(np.int64),
        minlength=n_patches,
    )
    xenium_cell_counts = np.bincount(cell_groups, minlength=n_patches)
    cohort = (tissue_counts >= args.min_visium_tissue_bins) & (
        xenium_cell_counts >= args.min_xenium_cells
    )

    clusters = pd.read_csv(args.xenium_clusters)
    cluster_by_cell = dict(zip(clusters["Barcode"].astype(str), clusters["Cluster"].astype(int)))
    cell_clusters = np.asarray([cluster_by_cell.get(cell, 0) for cell in cells["cell_id"].astype(str)])
    xenium_matrix, xenium_features, xenium_barcodes = read_10x_h5(args.xenium_h5)
    if xenium_barcodes != cells["cell_id"].astype(str).tolist():
        raise ComparisonError("Xenium matrix and cell table axes disagree")
    with h5py.File(args.xenium_h5, "r") as handle:
        xenium_feature_names = decode(handle["matrix/features/name"][:])
    cancer_cells = cancer_cell_membership(
        xenium_matrix,
        xenium_feature_names,
        cell_clusters,
        cancer_clusters,
        label_genes,
    )
    cancer_cell_counts = np.bincount(
        cell_groups, weights=cancer_cells.astype(float), minlength=n_patches
    )
    cancer_fraction = np.divide(
        cancer_cell_counts,
        xenium_cell_counts,
        out=np.zeros(n_patches, dtype=float),
        where=xenium_cell_counts > 0,
    )
    cancer_rich = cohort & (cancer_fraction >= args.cancer_rich_min_fraction)
    cancer_poor = cohort & (cancer_fraction <= args.cancer_poor_max_fraction)

    diffexp = pd.read_csv(args.xenium_diffexp)
    panel_gene_ids = diffexp["Feature ID"].astype(str).tolist()
    gene_name_by_id = dict(zip(diffexp["Feature ID"].astype(str), diffexp["Feature Name"].astype(str)))
    source_features: dict[str, set[str]] = {
        "xenium": set(xenium_features),
        "space_ranger": set(read_h5_feature_ids(args.sr_h5)),
    }
    for name, directory in policy_paths:
        source_features[name] = set(read_mex_feature_ids(directory))
    target_gene_ids = [
        gene for gene in panel_gene_ids if all(gene in feature_set for feature_set in source_features.values())
    ]
    target_gene_names = [gene_name_by_id[gene] for gene in target_gene_ids]

    panel_payload = json.loads(args.reference_panel_json.read_text(encoding="utf-8"))["payload"]
    reference_names = {
        str(target["type"]["data"]["name"])
        for target in panel_payload["targets"]
        if target.get("source", {}).get("category") == "current"
        and target.get("type", {}).get("descriptor") == "gene"
    }
    reference_mask = np.asarray([name in reference_names for name in target_gene_names])
    scopes = {
        "full_reference_panel": reference_mask,
        "exclude_mecom": reference_mask & (np.asarray(target_gene_names) != "MECOM"),
        "label_leakage_controlled": reference_mask
        & ~np.isin(np.asarray(target_gene_names), ("MECOM", *label_genes)),
    }

    xenium_counts, _ = aggregate_matrix(
        xenium_matrix, xenium_features, target_gene_ids, cell_groups, n_patches
    )
    del xenium_matrix
    gc.collect()
    method_counts: dict[str, np.ndarray] = {}
    sr_matrix, sr_features, sr_barcodes = read_10x_h5(args.sr_h5)
    sr_groups = visium_column_groups(sr_barcodes, positions, patch_lookup)
    method_counts["space_ranger"], _ = aggregate_matrix(
        sr_matrix, sr_features, target_gene_ids, sr_groups, n_patches
    )
    del sr_matrix
    gc.collect()
    for name, directory in policy_paths:
        matrix, features, barcodes = read_mex(directory)
        groups = visium_column_groups(barcodes, positions, patch_lookup)
        method_counts[name], _ = aggregate_matrix(
            matrix, features, target_gene_ids, groups, n_patches
        )
        del matrix
        gc.collect()

    summaries: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for scope, mask in scopes.items():
        scope_summary, scope_genes = evaluate_scope(
            scope,
            method_counts,
            xenium_counts,
            target_gene_ids,
            target_gene_names,
            mask,
            cancer_rich,
            cancer_poor,
            args.bootstrap_seed,
            args.bootstrap_replicates,
        )
        summaries.extend(scope_summary)
        gene_rows.extend(scope_genes)

    patch_rows = [
        {
            "patch_index": index,
            "patch_x": key[0],
            "patch_y": key[1],
            "in_comparison_cohort": int(cohort[index]),
            "xenium_cell_count": int(xenium_cell_counts[index]),
            "xenium_cancer_cell_count": int(cancer_cell_counts[index]),
            "xenium_cancer_cell_fraction": float(cancer_fraction[index]),
            "cancer_rich": int(cancer_rich[index]),
            "cancer_poor": int(cancer_poor[index]),
        }
        for index, key in enumerate(patch_axis)
    ]
    write_tsv(args.out_dir / "method_summary.tsv", summaries)
    write_tsv(args.out_dir / "gene_metrics.tsv", gene_rows)
    write_tsv(args.out_dir / "patch_labels.tsv", patch_rows)

    input_hashes = {str(path): sha256(path) for path in input_paths}
    for _, directory in policy_paths:
        for filename in ("matrix.mtx", "features.tsv", "barcodes.tsv"):
            path = directory / filename
            if not path.is_file():
                path = directory / f"{filename}.gz"
            input_hashes[str(path)] = sha256(path)
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "parameters": {
            "patch_um": args.patch_um,
            "cancer_clusters": list(cancer_clusters),
            "label_genes": list(label_genes),
            "cancer_cell_definition": "listed Xenium graph cluster AND positive for at least one label gene",
            "cancer_rich_min_fraction": args.cancer_rich_min_fraction,
            "cancer_poor_max_fraction": args.cancer_poor_max_fraction,
            "shape_normalization": "log1p(10000 * counts / full shared-gene-axis patch mass)",
            "raw_mass_normalized": False,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "axes": {
            "xenium_cell_count": len(cells),
            "xenium_cancer_cell_count": int(cancer_cells.sum()),
            "comparison_patch_count": int(cohort.sum()),
            "cancer_rich_patch_count": int(cancer_rich.sum()),
            "cancer_poor_patch_count": int(cancer_poor.sum()),
            "shared_gene_count": len(target_gene_ids),
            "reference_panel_shared_gene_count": int(reference_mask.sum()),
            "full_reference_panel_gene_count": int(scopes["full_reference_panel"].sum()),
            "exclude_mecom_gene_count": int(scopes["exclude_mecom"].sum()),
            "label_leakage_controlled_gene_count": int(scopes["label_leakage_controlled"].sum()),
        },
        "interpretation": {
            "xenium_role": "adjacent-section noisy biological oracle",
            "registration_scope": "coarse adjacent-section comparison; not cell-to-cell correspondence",
            "cohort_and_labels_independent_of_visium_expression": True,
        },
        "methods": summaries,
        "input_sha256": input_hashes,
        "artifacts": {
            "method_summary": "method_summary.tsv",
            "gene_metrics": "gene_metrics.tsv",
            "patch_labels": "patch_labels.tsv",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    try:
        result = run(parse_args())
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps({"status": result["status"], "axes": result["axes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
