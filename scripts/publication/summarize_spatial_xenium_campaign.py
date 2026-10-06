#!/usr/bin/env python3
"""Summarize frozen Xenium endpoints without requiring a favorable outcome."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from summarize_sealed_spatial_primary import sha256

POLICIES = ("strict", "hard")
MODES = ("compatibility", "annotated")


def read_tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite endpoint")
    return result


def exact_binomial_two_sided(wins, trials):
    if not 0 <= wins <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    observed = math.comb(trials, wins)
    return sum(math.comb(trials, k) for k in range(trials + 1)
               if math.comb(trials, k) <= observed) / 2**trials


def summarize_top20(rows, method_rows):
    if len(rows) != 20 or [int(r["xenium_abundance_rank"]) for r in rows] != list(range(1, 21)):
        raise ValueError("expected the frozen 20-gene ranked endpoint")
    if len({r["gene_id"] for r in rows}) != 20:
        raise ValueError("duplicate top-20 gene")
    methods = {}
    for row in method_rows:
        if row["method"] in methods:
            raise ValueError("duplicate top-20 method")
        methods[row["method"]] = row
    reference = [number(r["space_ranger_spatial_pearson"]) for r in rows]
    summaries, comparisons = {}, {}
    for method in ("space_ranger", *POLICIES):
        observed = methods[method]
        values = [number(r[f"{method}_spatial_pearson"]) for r in rows]
        mass = sum(number(r[f"{method}_visium_counts"]) for r in rows)
        median = statistics.median(values)
        if int(observed["top_gene_count"]) != 20 or not math.isclose(number(observed["median_spatial_pearson"]), median, abs_tol=1e-12) or not math.isclose(number(observed["summed_visium_counts"]), mass, abs_tol=1e-8):
            raise ValueError("top-20 table and method summary disagree")
        summaries[method] = dict(top_gene_count=20, median_spatial_pearson=median, summed_visium_counts=mass)
        if method in POLICIES:
            delta = [v - ref for v, ref in zip(values, reference)]
            wins, ties = sum(d > 0 for d in delta), sum(d == 0 for d in delta)
            comparisons[method] = dict(wins_vs_space_ranger=wins, ties_vs_space_ranger=ties,
                losses_vs_space_ranger=20-wins-ties, trials=20,
                two_sided_exact_binomial_p=exact_binomial_two_sided(wins, 20),
                mean_spatial_pearson_delta_vs_space_ranger=statistics.mean(delta),
                historical_test_definition="strictly better genes out of all 20; exact fair-binomial probability ordering",
                tied_genes_require_interpretation_review=bool(ties))
    return dict(gene_count=20, gene_axis=[r["gene_id"] for r in rows], method_summary=summaries, comparisons=comparisons)


def summarize_cancer(payload, expected_vendor_sha256):
    if payload["status"] != "pass" or payload["parameters"]["patch_um"] != 128:
        raise ValueError("unexpected cancer endpoint status or patch size")
    if expected_vendor_sha256 not in payload["input_sha256"].values():
        raise ValueError("cancer endpoint used a different vendor matrix")
    rows = [r for r in payload["methods"] if r["method"] in POLICIES]
    expected = {(p, s) for p in POLICIES for s in ("full_reference_panel", "exclude_mecom", "label_leakage_controlled")}
    if len(rows) != 6 or {(r["method"], r["scope"]) for r in rows} != expected:
        raise ValueError("incomplete or duplicate cancer policy/scope grid")
    intervals = []
    for row in rows:
        low = number(row["mean_oriented_gene_auc_delta_vs_space_ranger_ci95_low"])
        high = number(row["mean_oriented_gene_auc_delta_vs_space_ranger_ci95_high"])
        if low > high:
            raise ValueError("reversed confidence interval")
        intervals.append(dict(scope=row["scope"], method=row["method"],
            mean_delta=number(row["mean_oriented_gene_auc_delta_vs_space_ranger"]),
            ci95_low=low, ci95_high=high, spans_zero=low <= 0 <= high))
    return dict(axes=payload["axes"], policy_scope_interval_count=6,
        intervals_spanning_zero=sum(r["spans_zero"] for r in intervals),
        all_policy_scope_bootstrap_intervals_span_zero=all(r["spans_zero"] for r in intervals), intervals=intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}-top20", type=Path, required=True)
        parser.add_argument(f"--{mode}-cancer", type=Path, required=True)
    parser.add_argument("--vendor-sha256", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing an existing summary")
    result = dict(schema="visium_hd_processing.xenium_campaign_summary.v1", release=args.release,
                  vendor_sha256=args.vendor_sha256, top20={}, cancer_panel={}, inputs={})
    for mode in MODES:
        top = getattr(args, f"{mode}_top20")
        cancer = getattr(args, f"{mode}_cancer") / "summary.json"
        files = [top / "top20_genes.tsv", top / "top20_summary.tsv", cancer]
        result["top20"][mode] = summarize_top20(read_tsv(files[0]), read_tsv(files[1]))
        result["cancer_panel"][mode] = summarize_cancer(json.loads(cancer.read_text()), args.vendor_sha256)
        result["inputs"].update({str(p): sha256(p) for p in files})
    if result["top20"]["compatibility"]["gene_axis"] != result["top20"]["annotated"]["gene_axis"]:
        raise ValueError("top-20 gene axis differs between modes")
    for key in ("xenium_cell_count", "xenium_cancer_cell_count"):
        if result["cancer_panel"]["compatibility"]["axes"][key] != result["cancer_panel"]["annotated"]["axes"][key]:
            raise ValueError("Xenium cell labels differ between modes")
    result["interval_count"] = sum(r["policy_scope_interval_count"] for r in result["cancer_panel"].values())
    result["intervals_spanning_zero"] = sum(r["intervals_spanning_zero"] for r in result["cancer_panel"].values())
    result["interpretation_review"] = dict(
        any_non_null_cancer_interval=result["intervals_spanning_zero"] != result["interval_count"],
        any_top20_ties=any(s["tied_genes_require_interpretation_review"] for r in result["top20"].values() for s in r["comparisons"].values()),
        vendor_scope_corrected_from_historical_4_0_1_exon_only=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
