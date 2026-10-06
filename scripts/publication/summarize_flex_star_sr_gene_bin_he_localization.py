#!/usr/bin/env python3
"""Summarize two-slide STAR/SR gene-bin residual localization on H&E."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_hd_flex_h5ad_aggregates import correlation, sha256  # noqa: E402


def parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=RESULT_DIR")
    return label, Path(raw_path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty table: {path}")
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def top_gene_ids(
    rows: list[dict[str, str]], method: str, field: str, count: int,
) -> set[str]:
    selected = [row for row in rows if row["method"] == method]
    selected.sort(key=lambda row: (-float(row[field]), row["gene_id"]))
    return {row["gene_id"] for row in selected[:count]}


def cross_slide_row(
    method: str,
    left_label: str,
    left_rows: list[dict[str, str]],
    right_label: str,
    right_rows: list[dict[str, str]],
) -> dict[str, object]:
    left = {
        row["gene_id"]: row for row in left_rows if row["method"] == method
    }
    right = {
        row["gene_id"]: row for row in right_rows if row["method"] == method
    }
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        raise ValueError("fewer than two shared genes across slides")

    def values(rows: dict[str, dict[str, str]], field: str) -> np.ndarray:
        return np.asarray([float(rows[gene][field]) for gene in common], dtype=np.float64)

    left_residual = values(left, "star_positive_gene_bin_residual_mass")
    right_residual = values(right, "star_positive_gene_bin_residual_mass")
    left_delta = values(left, "signed_star_minus_space_ranger_mass")
    right_delta = values(right, "signed_star_minus_space_ranger_mass")
    left_sr = values(left, "space_ranger_mass")
    right_sr = values(right, "space_ranger_mass")
    abundant = (left_sr >= 100) & (right_sr >= 100)
    left_rate = left_delta[abundant] / left_sr[abundant]
    right_rate = right_delta[abundant] / right_sr[abundant]
    if np.count_nonzero(abundant) < 2:
        rate_pearson = rate_spearman = None
    else:
        rate_pearson = correlation(left_rate, right_rate)
        rate_spearman = correlation(left_rate, right_rate, ranked=True)
    top10_left = top_gene_ids(
        left_rows, method, "star_positive_gene_bin_residual_mass", 10,
    )
    top10_right = top_gene_ids(
        right_rows, method, "star_positive_gene_bin_residual_mass", 10,
    )
    top100_left = top_gene_ids(
        left_rows, method, "star_positive_gene_bin_residual_mass", 100,
    )
    top100_right = top_gene_ids(
        right_rows, method, "star_positive_gene_bin_residual_mass", 100,
    )
    common_top100 = sorted(top100_left & top100_right)
    return {
        "method": method,
        "left_dataset": left_label,
        "right_dataset": right_label,
        "common_genes": len(common),
        "star_positive_residual_log1p_pearson": correlation(
            np.log1p(left_residual), np.log1p(right_residual),
        ),
        "star_positive_residual_spearman": correlation(
            left_residual, right_residual, ranked=True,
        ),
        "signed_gene_delta_pearson": correlation(left_delta, right_delta),
        "signed_gene_delta_spearman": correlation(
            left_delta, right_delta, ranked=True,
        ),
        "genes_with_sr_mass_ge_100_on_both_slides": int(np.count_nonzero(abundant)),
        "signed_delta_rate_pearson_sr_mass_ge_100": rate_pearson,
        "signed_delta_rate_spearman_sr_mass_ge_100": rate_spearman,
        "genes_with_positive_net_delta_on_both_slides": int(
            np.count_nonzero((left_delta > 0) & (right_delta > 0))
        ),
        "top10_positive_residual_gene_overlap": len(top10_left & top10_right),
        "top100_positive_residual_gene_overlap": len(common_top100),
        "common_top100_positive_residual_gene_ids": ",".join(common_top100),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", type=parse_dataset, required=True,
        metavar="LABEL=RESULT_DIR",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    if len(args.dataset) != 2:
        raise SystemExit("exactly two --dataset inputs are required")
    labels = [label for label, _ in args.dataset]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate dataset label")

    all_method_rows: list[dict[str, object]] = []
    all_top_rows: list[dict[str, object]] = []
    genes_by_dataset: dict[str, list[dict[str, str]]] = {}
    inputs: dict[str, object] = {}
    method_sets: list[set[str]] = []
    for label, root in args.dataset:
        summary_path = root / "summary.json"
        method_path = root / "method_summary.tsv"
        gene_path = root / "gene_residuals.tsv"
        top_path = root / "top_gene_residuals.tsv"
        for path in (summary_path, method_path, gene_path, top_path):
            if not path.is_file():
                raise SystemExit(f"missing dataset result: {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "visium_hd_processing.flex_star_sr_gene_bin_he_residuals.v1":
            raise ValueError(f"unexpected result schema: {summary_path}")
        for name, path in (
            ("method_summary.tsv", method_path),
            ("gene_residuals.tsv", gene_path),
            ("top_gene_residuals.tsv", top_path),
        ):
            if sha256(path) != summary["outputs"][name]["sha256"]:
                raise ValueError(f"dataset output differs from summary: {path}")
        method_rows = read_rows(method_path)
        genes = read_rows(gene_path)
        top_rows = read_rows(top_path)
        method_sets.append({row["method"] for row in method_rows})
        genes_by_dataset[label] = genes
        for row in method_rows:
            all_method_rows.append({"dataset": label, **row})
        for row in top_rows:
            all_top_rows.append({"dataset": label, **row})
        inputs[label] = {
            "root": str(root.resolve()),
            "summary_sha256": sha256(summary_path),
            "method_summary_sha256": sha256(method_path),
            "gene_residuals_sha256": sha256(gene_path),
            "top_gene_residuals_sha256": sha256(top_path),
        }
    if method_sets[0] != method_sets[1]:
        raise ValueError("slides have different STAR method sets")

    cross_rows = [
        cross_slide_row(
            method,
            labels[0],
            genes_by_dataset[labels[0]],
            labels[1],
            genes_by_dataset[labels[1]],
        )
        for method in sorted(method_sets[0])
    ]
    args.out_dir.mkdir(parents=True)
    method_out = args.out_dir / "two_slide_method_summary.tsv"
    cross_out = args.out_dir / "cross_slide_gene_consistency.tsv"
    top_out = args.out_dir / "two_slide_top_gene_residuals.tsv"
    write_rows(method_out, all_method_rows)
    write_rows(cross_out, cross_rows)
    write_rows(top_out, all_top_rows)
    summary = {
        "schema": "visium_hd_processing.flex_star_sr_gene_bin_he_two_slide.v1",
        "datasets": labels,
        "methods": sorted(method_sets[0]),
        "inputs": inputs,
        "outputs": {},
        "invariants": {
            "gene_bin_residuals_not_total_bin_residuals": True,
            "molecule_identity_compared": False,
            "image_or_segmentation_not_used_for_assignment": True,
            "raw_mass_not_normalized": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    for path in (method_out, cross_out, top_out):
        summary["outputs"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    checksum_paths = (method_out, cross_out, top_out, summary_path)
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
