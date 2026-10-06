#!/usr/bin/env python3
"""Report the top Xenium-abundance genes across identical method axes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(
    gene_metrics: Path,
    xenium_diffexp: Path,
    top_n: int,
    selected_gene_names: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    metrics = pd.read_csv(gene_metrics, sep="\t")
    spatial_column = (
        "spatial_pearson"
        if "spatial_pearson" in metrics.columns
        else "spatial_pearson_zero_if_undefined"
    )
    if spatial_column not in metrics.columns:
        raise ValueError("gene metrics have no recognized spatial-Pearson column")
    methods = metrics["method"].drop_duplicates().astype(str).tolist()
    if not methods or methods[0] != "space_ranger":
        raise ValueError("gene metrics must start with the Space Ranger method")
    expected_genes = set(metrics.loc[metrics["method"] == methods[0], "gene_id"].astype(str))
    for method in methods[1:]:
        observed = set(metrics.loc[metrics["method"] == method, "gene_id"].astype(str))
        if observed != expected_genes:
            raise ValueError(f"gene axis differs for {method}")
    names = pd.read_csv(xenium_diffexp)[["Feature ID", "Feature Name"]].drop_duplicates()
    name_by_id = dict(zip(names["Feature ID"].astype(str), names["Feature Name"].astype(str)))
    if selected_gene_names is not None:
        selected_ids = {
            gene_id for gene_id, gene_name in name_by_id.items() if gene_name in selected_gene_names
        }
        metrics = metrics[metrics["gene_id"].astype(str).isin(selected_ids)].copy()
        if metrics.empty:
            raise ValueError("none of the selected gene names are present on the evaluated axis")
    reference = metrics.groupby("gene_id", sort=False)["xenium_counts"].agg(["min", "max"])
    if not np.allclose(reference["min"], reference["max"]):
        raise ValueError("Xenium reference totals differ across methods")
    ordered = (
        reference.rename_axis("gene_id").reset_index()
        .sort_values(["max", "gene_id"], ascending=[False, True])
        .head(top_n)
        .set_index("gene_id")
    )
    indexed = {method: metrics[metrics["method"] == method].set_index("gene_id") for method in methods}
    rows: list[dict[str, object]] = []
    for rank, (gene_id, source) in enumerate(ordered.iterrows(), start=1):
        row: dict[str, object] = {
            "xenium_abundance_rank": rank,
            "gene_id": gene_id,
            "gene_name": name_by_id.get(gene_id, gene_id),
            "xenium_counts": source["max"],
        }
        for method in methods:
            current = indexed[method].loc[gene_id]
            row[f"{method}_visium_counts"] = current["visium_counts"]
            row[f"{method}_spatial_pearson"] = current[spatial_column]
        rows.append(row)
    summary_rows = [
        {
            "method": method,
            "top_gene_count": len(rows),
            "median_spatial_pearson": float(
                np.median([float(row[f"{method}_spatial_pearson"]) for row in rows])
            ),
            "summed_visium_counts": float(
                sum(float(row[f"{method}_visium_counts"]) for row in rows)
            ),
        }
        for method in methods
    ]
    return rows, summary_rows


def panel_gene_names(path: Path, source_category: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    selected = {
        str(target["type"]["data"]["name"])
        for target in payload["targets"]
        if target.get("source", {}).get("category") == source_category
        and target.get("type", {}).get("descriptor") == "gene"
    }
    if not selected:
        raise ValueError(f"no gene targets have source category {source_category!r}")
    return selected


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-metrics", required=True, type=Path)
    parser.add_argument("--xenium-diffexp", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--selected-panel-json",
        type=Path,
        help="Optionally restrict the Xenium-ranked table to genes in this official panel JSON.",
    )
    parser.add_argument("--selected-source-category", default="current")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    args = parser.parse_args()
    selected = (
        panel_gene_names(args.selected_panel_json, args.selected_source_category)
        if args.selected_panel_json
        else None
    )
    rows, summary = summarize(args.gene_metrics, args.xenium_diffexp, args.top_n, selected)
    write_tsv(args.out, rows)
    write_tsv(args.summary_out, summary)
    print(args.out)
    print(args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
