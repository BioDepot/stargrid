#!/usr/bin/env python3
"""Locate the gene and spatial-bin mass changed by a Flex MAPQ-mode toggle.

The comparison is post-seal and read-only.  It operates on matched materialized
MEX matrices, preserves raw mass, and never treats the sign of a small oracle
difference as a policy-selection criterion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np

from compare_hd_flex_h5ad_aggregates import (
    barcode_coordinate,
    correlation,
    load_mex_totals,
    mex_file,
    open_text,
    sha256,
)


def align_totals(
    labels: list[str],
    totals: np.ndarray,
    union_index: dict[str, int],
) -> np.ndarray:
    result = np.zeros(len(union_index), dtype=np.float64)
    for label, value in zip(labels, totals, strict=True):
        result[union_index[label]] += value
    return result


def split_delta(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta, dtype=np.float64)
    return np.clip(delta, 0.0, None), np.clip(-delta, 0.0, None)


def rank_deciles(baseline: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Assign deterministic equal-count deciles within the selected axis."""
    indexes = np.flatnonzero(selected)
    result = np.full(len(baseline), -1, dtype=np.int8)
    if not len(indexes):
        return result
    order = indexes[np.argsort(baseline[indexes], kind="stable")]
    result[order] = np.minimum(
        9, np.arange(len(order), dtype=np.int64) * 10 // len(order)
    )
    return result


def concentration(values: np.ndarray, count: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if not total:
        return 0.0
    return float(np.sort(values)[-min(count, len(values)) :].sum() / total)


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    return correlation(left, right)


def load_h5ad_metadata(
    path: Path,
    *,
    scale: int,
    columns: list[str],
) -> dict[tuple[int, int], dict[str, str]]:
    dataset = ad.read_h5ad(path, backed="r")
    try:
        missing = sorted(set(columns) - set(dataset.obs.columns))
        if missing:
            raise ValueError(f"H5AD lacks requested annotation columns: {missing}")
        values = {
            column: dataset.obs[column].astype(str).to_numpy()
            for column in columns
        }
        result: dict[tuple[int, int], dict[str, str]] = {}
        for index, barcode in enumerate(dataset.obs_names):
            coordinate = barcode_coordinate(str(barcode), scale=scale)
            if coordinate in result:
                raise ValueError(f"duplicate H5AD coordinate: {coordinate}")
            result[coordinate] = {
                column: str(values[column][index]) for column in columns
            }
        return result
    finally:
        dataset.file.close()


def load_feature_names(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with open_text(mex_file(root, "features.tsv")) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            gene_id = fields[0].split(".", 1)[0]
            name = fields[1] if len(fields) > 1 else gene_id
            if gene_id in result and result[gene_id] != name:
                raise ValueError(f"conflicting names for feature {gene_id}")
            result[gene_id] = name
    return result


def decile_rows(
    axis: str,
    baseline: np.ndarray,
    mapq_off: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, object]]:
    assignments = rank_deciles(baseline, selected)
    rows: list[dict[str, object]] = []
    delta = mapq_off - baseline
    positive, negative = split_delta(delta)
    for decile in range(10):
        mask = assignments == decile
        rows.append({
            "axis": axis,
            "genomic_mass_decile": decile + 1,
            "axis_elements": int(mask.sum()),
            "genomic_mass": float(baseline[mask].sum()),
            "mapq_off_mass": float(mapq_off[mask].sum()),
            "signed_delta_mass": float(delta[mask].sum()),
            "positive_net_increment_mass": float(positive[mask].sum()),
            "negative_net_redistribution_mass": float(negative[mask].sum()),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genomic-mex", type=Path, required=True)
    parser.add_argument("--mapq-off-mex", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--h5ad", type=Path)
    parser.add_argument(
        "--annotation-column", action="append",
        default=["annotation", "spatial_cluster", "high_quality", "codex_common"],
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy", default="hard")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    for path in (args.genomic_mex, args.mapq_off_mex):
        if not path.is_dir():
            raise SystemExit(f"missing MEX directory: {path}")
    if args.h5ad is not None and not args.h5ad.is_file():
        raise SystemExit(f"missing H5AD: {args.h5ad}")

    (
        genomic_features, genomic_barcodes, genomic_gene, genomic_bin,
        genomic_nnz, genomic_inputs,
    ) = load_mex_totals(args.genomic_mex, scale=args.scale)
    (
        off_features, off_barcodes, off_gene, off_bin, off_nnz, off_inputs,
    ) = load_mex_totals(args.mapq_off_mex, scale=args.scale)

    genes = sorted(set(genomic_features) | set(off_features))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    genomic_names = load_feature_names(args.genomic_mex)
    off_names = load_feature_names(args.mapq_off_mex)
    conflicts = sorted(
        gene for gene in set(genomic_names) & set(off_names)
        if genomic_names[gene] != off_names[gene]
    )
    if conflicts:
        raise ValueError(f"feature names differ between modes: {conflicts[:5]}")
    gene_names = genomic_names | off_names
    genomic_gene_aligned = align_totals(genomic_features, genomic_gene, gene_index)
    off_gene_aligned = align_totals(off_features, off_gene, gene_index)

    genomic_coordinates = [
        barcode_coordinate(barcode, scale=args.scale) for barcode in genomic_barcodes
    ]
    off_coordinates = [
        barcode_coordinate(barcode, scale=args.scale) for barcode in off_barcodes
    ]
    coordinates = sorted(set(genomic_coordinates) | set(off_coordinates))
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    genomic_bin_aligned = align_totals(
        [f"{row}:{column}" for row, column in genomic_coordinates],
        genomic_bin,
        {
            f"{row}:{column}": index
            for (row, column), index in coordinate_index.items()
        },
    )
    off_bin_aligned = align_totals(
        [f"{row}:{column}" for row, column in off_coordinates],
        off_bin,
        {
            f"{row}:{column}": index
            for (row, column), index in coordinate_index.items()
        },
    )

    gene_delta = off_gene_aligned - genomic_gene_aligned
    bin_delta = off_bin_aligned - genomic_bin_aligned
    gene_positive, gene_negative = split_delta(gene_delta)
    bin_positive, bin_negative = split_delta(bin_delta)
    if not math.isclose(
        float(gene_delta.sum()), float(bin_delta.sum()), rel_tol=1e-12, abs_tol=1e-5,
    ):
        raise ValueError("gene and spatial axes disagree on net MAPQ increment")

    metadata = (
        load_h5ad_metadata(
            args.h5ad, scale=args.scale, columns=args.annotation_column,
        )
        if args.h5ad is not None else {}
    )
    args.out_dir.mkdir(parents=True)

    gene_order = np.argsort(-gene_positive, kind="stable")
    gene_rows: list[dict[str, object]] = []
    for rank, index in enumerate(gene_order, start=1):
        baseline = genomic_gene_aligned[index]
        gene_rows.append({
            "positive_increment_rank": rank,
            "gene_id": genes[index],
            "gene_name": gene_names[genes[index]],
            "genomic_mass": float(baseline),
            "mapq_off_mass": float(off_gene_aligned[index]),
            "signed_delta_mass": float(gene_delta[index]),
            "positive_net_increment_mass": float(gene_positive[index]),
            "negative_net_redistribution_mass": float(gene_negative[index]),
            "signed_delta_fraction_of_genomic": (
                float(gene_delta[index] / baseline) if baseline else "NA"
            ),
        })
    gene_path = args.out_dir / "gene_localization.tsv"
    write_rows(gene_path, gene_rows, list(gene_rows[0]))

    category_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    bin_rows: list[dict[str, object]] = []
    for index, coordinate in enumerate(coordinates):
        meta = metadata.get(coordinate)
        row: dict[str, object] = {
            "array_row": coordinate[0],
            "array_col": coordinate[1],
            "on_retained_h5ad_axis": meta is not None if metadata else "NA",
            "genomic_mass": float(genomic_bin_aligned[index]),
            "mapq_off_mass": float(off_bin_aligned[index]),
            "signed_delta_mass": float(bin_delta[index]),
            "positive_net_increment_mass": float(bin_positive[index]),
            "negative_net_redistribution_mass": float(bin_negative[index]),
        }
        for column in args.annotation_column:
            row[column] = meta[column] if meta is not None else "__outside_h5ad__"
            key = (column, str(row[column]))
            category_totals[key]["bins"] += 1
            category_totals[key]["genomic_mass"] += genomic_bin_aligned[index]
            category_totals[key]["mapq_off_mass"] += off_bin_aligned[index]
            category_totals[key]["signed_delta_mass"] += bin_delta[index]
            category_totals[key]["positive_net_increment_mass"] += bin_positive[index]
            category_totals[key]["negative_net_redistribution_mass"] += bin_negative[index]
        bin_rows.append(row)
    bin_path = args.out_dir / "bin_localization.tsv"
    write_rows(bin_path, bin_rows, list(bin_rows[0]))

    category_rows = []
    for (column, value), totals in sorted(category_totals.items()):
        genomic_mass = totals["genomic_mass"]
        positive_mass = totals["positive_net_increment_mass"]
        category_rows.append({
            "annotation_column": column,
            "annotation_value": value,
            **totals,
            "signed_delta_fraction_of_genomic": (
                totals["signed_delta_mass"] / genomic_mass
                if genomic_mass else "NA"
            ),
            "positive_increment_fraction_within_annotation_column": (
                positive_mass / float(bin_positive.sum())
                if float(bin_positive.sum()) else 0.0
            ),
        })
    category_path = args.out_dir / "annotation_localization.tsv"
    write_rows(
        category_path,
        category_rows,
        [
            "annotation_column", "annotation_value", "bins", "genomic_mass",
            "mapq_off_mass", "signed_delta_mass",
            "positive_net_increment_mass", "negative_net_redistribution_mass",
            "signed_delta_fraction_of_genomic",
            "positive_increment_fraction_within_annotation_column",
        ],
    )

    deciles = [
        *decile_rows(
            "gene", genomic_gene_aligned, off_gene_aligned,
            (genomic_gene_aligned > 0) | (off_gene_aligned > 0),
        ),
        *decile_rows(
            "spatial_bin", genomic_bin_aligned, off_bin_aligned,
            (genomic_bin_aligned > 0) | (off_bin_aligned > 0),
        ),
    ]
    decile_path = args.out_dir / "baseline_deciles.tsv"
    write_rows(decile_path, deciles, list(deciles[0]))

    selected_genes = (genomic_gene_aligned > 0) | (off_gene_aligned > 0)
    selected_bins = (genomic_bin_aligned > 0) | (off_bin_aligned > 0)
    on_h5ad = np.asarray(
        [coordinate in metadata for coordinate in coordinates], dtype=bool,
    )
    summary = {
        "schema": "visium_hd_processing.flex_mapq_increment_localization.v1",
        "dataset": args.dataset,
        "policy": args.policy,
        "scale_um": args.scale,
        "interpretation": (
            "Descriptive localization of a toggleable MAPQ-mode increment; "
            "small oracle-delta signs are not policy-selection evidence."
        ),
        "mass": {
            "genomic": float(genomic_gene_aligned.sum()),
            "mapq_off": float(off_gene_aligned.sum()),
            "net_increment": float(gene_delta.sum()),
            "percent_increment_over_genomic": float(
                100.0 * gene_delta.sum() / genomic_gene_aligned.sum()
            ),
        },
        "gene_axis": {
            "union_features": len(genes),
            "genomic_observed_features": len(genomic_features),
            "mapq_off_observed_features": len(off_features),
            "newly_detected_features": int(np.count_nonzero(
                (genomic_gene_aligned == 0) & (off_gene_aligned > 0)
            )),
            "positive_net_increment_mass": float(gene_positive.sum()),
            "negative_net_redistribution_mass": float(gene_negative.sum()),
            "positive_increment_top10_share": concentration(gene_positive, 10),
            "positive_increment_top100_share": concentration(gene_positive, 100),
            "delta_vs_genomic_pearson_detected_union": safe_correlation(
                gene_delta[selected_genes], genomic_gene_aligned[selected_genes]
            ),
        },
        "spatial_axis": {
            "union_bins": len(coordinates),
            "newly_detected_bins": int(np.count_nonzero(
                (genomic_bin_aligned == 0) & (off_bin_aligned > 0)
            )),
            "positive_net_increment_mass": float(bin_positive.sum()),
            "negative_net_redistribution_mass": float(bin_negative.sum()),
            "positive_increment_top1000_bin_share": concentration(bin_positive, 1000),
            "delta_vs_genomic_pearson_detected_union": safe_correlation(
                bin_delta[selected_bins], genomic_bin_aligned[selected_bins]
            ),
            "positive_increment_on_retained_h5ad_axis": (
                float(bin_positive[on_h5ad].sum()) if metadata else None
            ),
            "positive_increment_outside_retained_h5ad_axis": (
                float(bin_positive[~on_h5ad].sum()) if metadata else None
            ),
        },
        "inputs": {
            "genomic_mex": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in genomic_inputs
            ],
            "mapq_off_mex": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in off_inputs
            ],
            "h5ad": (
                {"path": str(args.h5ad.resolve()), "sha256": sha256(args.h5ad)}
                if args.h5ad is not None else None
            ),
            "genomic_nnz": genomic_nnz,
            "mapq_off_nnz": off_nnz,
        },
    }
    output_paths = [gene_path, bin_path, category_path, decile_path]
    summary["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in output_paths
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [*output_paths, summary_path]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
