#!/usr/bin/env python3
"""Compare reference versions on union/shared genes and a fixed barcode axis.

Genes absent from a reference are structural zeros only for the union-axis
comparison. Their raw mass is reported separately; it is not discarded or
interpreted as observed absence of expression.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_hd_flex_h5ad_aggregates import (
    barcode_coordinate, correlation, load_h5ad_totals, load_mex_totals,
    normalized_tv, parse_methods, sha256,
)


def compare_aggregates(label, scale, star, vendor):
    features, barcodes, genes, bins, nnz, _ = star
    sr_features, sr_barcodes, sr_genes, sr_bins, sr_nnz = vendor
    union = sorted(set(features) | set(sr_features))
    index = {gene: i for i, gene in enumerate(union)}
    left, right = np.zeros(len(union)), np.zeros(len(union))
    left[[index[g] for g in features]] = genes
    right[[index[g] for g in sr_features]] = sr_genes
    shared = sorted(set(features) & set(sr_features))
    star_only = sorted(set(features) - set(sr_features))
    sr_only = sorted(set(sr_features) - set(features))
    shared_indices = [index[g] for g in shared]
    coordinates = [barcode_coordinate(b, scale=scale) for b in sr_barcodes]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate H5AD barcode coordinates")
    barcode_index = {b: i for i, b in enumerate(coordinates)}
    star_on_sr = np.zeros(len(sr_barcodes))
    outside_mass, outside_barcodes = 0.0, 0
    for barcode, mass in zip(barcodes, bins):
        target = barcode_index.get(barcode_coordinate(barcode, scale=scale))
        if target is None:
            outside_mass += float(mass)
            outside_barcodes += 1
        else:
            star_on_sr[target] += mass
    star_mass, sr_mass = float(np.sum(genes)), float(np.sum(sr_genes))
    row = dict(
        method=label, scale_um=scale, star_features=len(features),
        space_ranger_features=len(sr_features), union_features=len(union),
        shared_features=len(shared), star_features_absent_from_space_ranger=len(star_only),
        space_ranger_features_absent_from_star=len(sr_only),
        star_mass_on_features_absent_from_space_ranger=float(np.sum(left[[index[g] for g in star_only]])),
        absent_feature_space_ranger_mass=float(np.sum(right[[index[g] for g in sr_only]])),
        star_mass_on_shared_features=float(np.sum(left[shared_indices])),
        space_ranger_mass_on_shared_features=float(np.sum(right[shared_indices])),
        star_raw_mass=star_mass, space_ranger_raw_mass=sr_mass,
        signed_mass_difference=star_mass-sr_mass,
        mass_ratio=star_mass/sr_mass if sr_mass else None,
        gene_total_pearson=correlation(left, right),
        gene_total_pearson_on_shared_features=(
            correlation(left[shared_indices], right[shared_indices]) if len(shared) > 1 else None),
        gene_log1p_pearson=correlation(np.log1p(left), np.log1p(right)),
        gene_normalized_total_variation=normalized_tv(left, right),
        star_barcodes=len(barcodes), space_ranger_barcodes=len(sr_barcodes),
        star_nnz=nnz, space_ranger_nnz=sr_nnz,
        star_mass_on_space_ranger_barcodes=float(np.sum(star_on_sr)),
        star_mass_outside_space_ranger_barcodes=outside_mass,
        star_barcodes_outside_space_ranger_object=outside_barcodes,
        spatial_bin_total_pearson_on_space_ranger_axis=correlation(star_on_sr, sr_bins),
        spatial_bin_log1p_pearson_on_space_ranger_axis=correlation(np.log1p(star_on_sr), np.log1p(sr_bins)),
        spatial_bin_normalized_total_variation_on_space_ranger_axis=normalized_tv(star_on_sr, sr_bins),
    )
    return row, dict(star_only_feature_ids=star_only, space_ranger_only_feature_ids=sr_only,
                     shared_feature_ids=shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-h5ad", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--umi-mode", default="1mm_cr")
    parser.add_argument("--h5ad-row-chunk", type=int, default=50_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing output: {args.out_dir}")
    if args.scale < 1 or args.h5ad_row_chunk < 1:
        parser.error("scale and row chunk must be positive")
    vendor = load_h5ad_totals(args.vendor_h5ad, row_chunk=args.h5ad_row_chunk)
    rows, inputs, axes = [], {}, {}
    for label, root in parse_methods(args.method):
        star = load_mex_totals(root, scale=args.scale)
        row, axes[label] = compare_aggregates(label, args.scale, star, vendor)
        rows.append(dict(umi_mode=args.umi_mode, **row))
        inputs[label] = {str(p.resolve()): sha256(p) for p in star[-1]}
    args.out_dir.mkdir(parents=True)
    table = args.out_dir / "aggregate_concordance.tsv"
    with table.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        schema="visium_hd_processing.cross_reference_h5ad_aggregate_concordance.v1",
        comparison_axis=dict(gene="union and intersection of declared reference feature IDs",
                             spatial="retained vendor barcode axis; all genes in each reference"),
        interpretation="Reference scope differs. Structural zeros on the union axis are not biological absences; full count differences include annotation scope.",
        umi_mode=args.umi_mode, scale_um=args.scale, feature_axis_audit=axes,
        inputs=dict(vendor={str(args.vendor_h5ad.resolve()): sha256(args.vendor_h5ad)}, methods=inputs),
        helper_source_sha256=sha256(Path(__file__).with_name("compare_hd_flex_h5ad_aggregates.py")),
        outputs={table.name: sha256(table)},
        invariants=dict(raw_mass_not_normalized=True, unmatched_feature_mass_not_discarded=True,
                        gene_scope_and_barcode_scope_reported_separately=True),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
