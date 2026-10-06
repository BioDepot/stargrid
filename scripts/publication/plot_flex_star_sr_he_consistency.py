#!/usr/bin/env python3
"""Plot the matched CRC/SPATCH STAR-versus-Space-Ranger H&E result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("CRC", "SPATCH")
METHODS = ("hard", "soft_expected")
COLORS = {"CRC": "#16697A", "SPATCH": "#D95F02"}
METHOD_MARKERS = {"hard": "o", "soft_expected": "s"}
REQUIRED_COLUMNS = {
    "dataset",
    "method",
    "star_percent_excess_over_space_ranger",
    "star_minus_space_ranger_cell_roc_auc",
    "star_minus_space_ranger_nucleus_roc_auc",
    "star_minus_space_ranger_cell_fraction_spearman_16um",
    "star_minus_space_ranger_nucleus_fraction_spearman_16um",
    "star_only_minus_space_ranger_only_cell_roc_auc",
    "star_only_minus_space_ranger_only_nucleus_roc_auc",
    "star_only_minus_space_ranger_only_cell_fraction_spearman_16um",
    "star_only_minus_space_ranger_only_nucleus_fraction_spearman_16um",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("STAR/SR table lacks required figure columns")
        rows = list(reader)
    observed = {(row["dataset"], row["method"]) for row in rows}
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("unexpected dataset/method rows")
    return rows


def row_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset"], row["method"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    rows = load_rows(args.table)
    by_key = row_index(rows)
    args.out_dir.mkdir(parents=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.6), constrained_layout=False)
    fig.subplots_adjust(
        left=0.065, right=0.99, bottom=0.21, top=0.76, wspace=0.18,
    )

    ax = axes[0]
    for dataset_index, dataset in enumerate(DATASETS):
        for method_index, method in enumerate(METHODS):
            row = by_key[(dataset, method)]
            x = dataset_index + (-0.10 if method == "hard" else 0.10)
            ax.scatter(
                x, float(row["star_percent_excess_over_space_ranger"]),
                s=68, color=COLORS[dataset], marker=METHOD_MARKERS[method],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_ylabel("STAR excess over Space Ranger (%)")
    ax.set_title("A  Common-axis count mass")
    ax.grid(axis="y", alpha=0.2)

    whole_metrics = [
        ("Cell AUC", "star_minus_space_ranger_cell_roc_auc"),
        ("Nucleus AUC", "star_minus_space_ranger_nucleus_roc_auc"),
        ("Cell 16-µm ρ", "star_minus_space_ranger_cell_fraction_spearman_16um"),
        ("Nucleus 16-µm ρ", "star_minus_space_ranger_nucleus_fraction_spearman_16um"),
    ]
    residual_metrics = [
        ("Cell AUC", "star_only_minus_space_ranger_only_cell_roc_auc"),
        ("Nucleus AUC", "star_only_minus_space_ranger_only_nucleus_roc_auc"),
        ("Cell 16-µm ρ", "star_only_minus_space_ranger_only_cell_fraction_spearman_16um"),
        ("Nucleus 16-µm ρ", "star_only_minus_space_ranger_only_nucleus_fraction_spearman_16um"),
    ]
    for ax, metrics, title in (
        (axes[1], whole_metrics, "B  Whole-field H&E fit"),
        (axes[2], residual_metrics, "C  STAR-only minus SR-only H&E fit"),
    ):
        x_base = np.arange(len(metrics), dtype=np.float64)
        offsets = {
            ("CRC", "hard"): -0.18,
            ("CRC", "soft_expected"): -0.06,
            ("SPATCH", "hard"): 0.06,
            ("SPATCH", "soft_expected"): 0.18,
        }
        for dataset in DATASETS:
            for method in METHODS:
                row = by_key[(dataset, method)]
                values = [1000.0 * float(row[column]) for _, column in metrics]
                ax.scatter(
                    x_base + offsets[(dataset, method)], values,
                    s=45, color=COLORS[dataset], marker=METHOD_MARKERS[method],
                    edgecolor="white", linewidth=0.7, zorder=3,
                )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x_base, [name for name, _ in metrics], rotation=24, ha="right")
        ax.set_ylabel("Difference × 10³")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    handles = []
    labels = []
    for dataset in DATASETS:
        handles.append(plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=7,
            markerfacecolor=COLORS[dataset], markeredgecolor="white",
        ))
        labels.append(dataset)
    for method in METHODS:
        handles.append(plt.Line2D(
            [], [], linestyle="none", marker=METHOD_MARKERS[method], markersize=7,
            markerfacecolor="#777777", markeredgecolor="white",
        ))
        labels.append("Hard" if method == "hard" else "Soft expected")
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.895),
        ncol=4, frameon=False,
    )
    fig.suptitle(
        "STAR versus Space Ranger across matched independent H&E structures",
        fontsize=12, y=0.975,
    )
    outputs = {
        "png": args.out_dir / "flex_star_sr_he_consistency.png",
        "pdf": args.out_dir / "flex_star_sr_he_consistency.pdf",
        "svg": args.out_dir / "flex_star_sr_he_consistency.svg",
    }
    fig.savefig(outputs["png"], dpi=240, bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    summary = {
        "schema": "visium_hd_processing.flex_star_sr_he_consistency_figure.v1",
        "input": {"path": str(args.table.resolve()), "sha256": sha256(args.table)},
        "panels": {
            "A": "STAR count-mass excess over Space Ranger on the Space Ranger barcode axis",
            "B": "whole-field STAR minus Space Ranger H&E metric differences",
            "C": "positive spatial-bin residual STAR-only minus Space-Ranger-only H&E differences",
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
