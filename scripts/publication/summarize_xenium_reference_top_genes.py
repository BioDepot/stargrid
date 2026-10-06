#!/usr/bin/env python3
"""Summarize the Xenium-abundant genes in a named reference gene set.

The adjacent-section Xenium scorer emits one row per method and gene in
``reference_gene_set_metrics.tsv``.  This utility selects a single shared
top-N gene axis by Xenium abundance and pivots the Visium counts and spatial
correlations into a compact, auditable companion table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-gene-set-metrics", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    # Use the same numeric parsing convention as the original companion audit.
    return pd.read_csv(path, sep="\t").to_dict(orient="records")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    rows = read_rows(args.reference_gene_set_metrics)
    if not rows:
        raise ValueError("reference gene-set metrics table is empty")

    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    by_method: dict[str, dict[str, dict[str, object]]] = {method: {} for method in methods}
    gene_metadata: dict[str, tuple[str, float]] = {}
    for row in rows:
        method = str(row["method"])
        gene_id = str(row["gene_id"])
        if gene_id in by_method[method]:
            raise ValueError(f"duplicate method/gene row: {method}/{gene_id}")
        by_method[method][gene_id] = row
        metadata = (str(row["gene_name"]), float(row["xenium_counts"]))
        if gene_id in gene_metadata and gene_metadata[gene_id] != metadata:
            raise ValueError(f"inconsistent gene metadata: {gene_id}")
        gene_metadata[gene_id] = metadata

    gene_axes = [set(by_method[method]) for method in methods]
    if any(axis != gene_axes[0] for axis in gene_axes[1:]):
        raise ValueError("methods do not share an identical reference gene axis")

    ranked_gene_ids = sorted(
        gene_axes[0],
        key=lambda gene_id: (-gene_metadata[gene_id][1], gene_id),
    )[: args.top_n]

    detail_rows: list[dict[str, object]] = []
    for rank, gene_id in enumerate(ranked_gene_ids, start=1):
        gene_name, xenium_counts = gene_metadata[gene_id]
        output: dict[str, object] = {
            "xenium_abundance_rank": rank,
            "gene_id": gene_id,
            "gene_name": gene_name,
            "xenium_counts": xenium_counts,
        }
        for method in methods:
            row = by_method[method][gene_id]
            output[f"{method}_visium_counts"] = float(row["visium_counts"])
            output[f"{method}_spatial_pearson"] = float(
                row["spatial_pearson_zero_if_undefined"]
            )
        detail_rows.append(output)

    summary_rows: list[dict[str, object]] = []
    for method in methods:
        counts = [float(by_method[method][gene_id]["visium_counts"]) for gene_id in ranked_gene_ids]
        correlations = [
            float(by_method[method][gene_id]["spatial_pearson_zero_if_undefined"])
            for gene_id in ranked_gene_ids
        ]
        summary_rows.append(
            {
                "method": method,
                "top_gene_count": len(ranked_gene_ids),
                "median_spatial_pearson": float(np.median(correlations)),
                "summed_visium_counts": sum(counts),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=False)
    detail_fields = ["xenium_abundance_rank", "gene_id", "gene_name", "xenium_counts"]
    for method in methods:
        detail_fields.extend([f"{method}_visium_counts", f"{method}_spatial_pearson"])
    write_rows(args.out_dir / "top_genes.tsv", detail_fields, detail_rows)
    write_rows(
        args.out_dir / "summary.tsv",
        ["method", "top_gene_count", "median_spatial_pearson", "summed_visium_counts"],
        summary_rows,
    )


if __name__ == "__main__":
    main()
