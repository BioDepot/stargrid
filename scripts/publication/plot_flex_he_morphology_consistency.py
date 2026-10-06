#!/usr/bin/env python3
"""Plot the matched CRC/SPATCH Flex H&E morphology consistency result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "dataset",
    "policy",
    "percent_net_increment_over_genomic",
    "positive_minus_genomic_cell_fraction_pp",
    "positive_minus_genomic_nucleus_fraction_pp",
    "mapq_off_minus_genomic_cell_roc_auc",
    "mapq_off_minus_genomic_nucleus_roc_auc",
    "mapq_off_minus_genomic_cell_fraction_spearman_16um",
    "mapq_off_minus_genomic_nucleus_fraction_spearman_16um",
}
DATASET_COLORS = {"CRC": "#2864B4", "SPATCH": "#D97706"}
POLICY_MARKERS = {"hard": "o", "soft_expected": "D"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing consistency columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("empty consistency table")
    observed = {(row["dataset"], row["policy"]) for row in rows}
    expected = {
        (dataset, policy)
        for dataset in DATASET_COLORS
        for policy in POLICY_MARKERS
    }
    if observed != expected:
        raise ValueError(f"unexpected dataset/policy rows: {sorted(observed)}")
    return rows


def row_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset"], row["policy"]): row for row in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    rows = load_rows(args.table)
    indexed = row_index(rows)
    args.out_dir.mkdir(parents=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), constrained_layout=True)

    datasets = list(DATASET_COLORS)
    policies = list(POLICY_MARKERS)
    offsets = {"hard": -0.09, "soft_expected": 0.09}

    ax = axes[0]
    for dataset_index, dataset in enumerate(datasets):
        for policy in policies:
            row = indexed[(dataset, policy)]
            ax.scatter(
                dataset_index + offsets[policy],
                float(row["percent_net_increment_over_genomic"]),
                color=DATASET_COLORS[dataset],
                marker=POLICY_MARKERS[policy],
                facecolors=(
                    DATASET_COLORS[dataset] if policy == "hard" else "white"
                ),
                edgecolors=DATASET_COLORS[dataset],
                s=42,
                linewidths=1.2,
                zorder=3,
            )
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel("MAPQ-off net gain (%)")
    ax.set_title("A  Additional signal")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)

    ax = axes[1]
    categories = [
        ("CRC", "cell"),
        ("CRC", "nucleus"),
        ("SPATCH", "cell"),
        ("SPATCH", "nucleus"),
    ]
    for category_index, (dataset, compartment) in enumerate(categories):
        for policy in policies:
            row = indexed[(dataset, policy)]
            key = f"positive_minus_genomic_{compartment}_fraction_pp"
            ax.scatter(
                category_index + offsets[policy],
                float(row[key]),
                color=DATASET_COLORS[dataset],
                marker=POLICY_MARKERS[policy],
                facecolors=(
                    DATASET_COLORS[dataset] if policy == "hard" else "white"
                ),
                edgecolors=DATASET_COLORS[dataset],
                s=42,
                linewidths=1.2,
                zorder=3,
            )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(
        range(len(categories)),
        [f"{dataset}\n{compartment}" for dataset, compartment in categories],
    )
    ax.set_ylabel("Added − genomic mass fraction (pp)")
    ax.set_title("B  H&E localization")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)

    ax = axes[2]
    metrics = [
        ("Cell\nAUC", "mapq_off_minus_genomic_cell_roc_auc"),
        ("Nucleus\nAUC", "mapq_off_minus_genomic_nucleus_roc_auc"),
        (
            "Cell\nSpearman",
            "mapq_off_minus_genomic_cell_fraction_spearman_16um",
        ),
        (
            "Nucleus\nSpearman",
            "mapq_off_minus_genomic_nucleus_fraction_spearman_16um",
        ),
    ]
    dataset_offsets = {"CRC": -0.12, "SPATCH": 0.12}
    policy_offsets = {"hard": -0.025, "soft_expected": 0.025}
    for metric_index, (_, key) in enumerate(metrics):
        for dataset in datasets:
            for policy in policies:
                row = indexed[(dataset, policy)]
                ax.scatter(
                    metric_index + dataset_offsets[dataset] + policy_offsets[policy],
                    1000.0 * float(row[key]),
                    color=DATASET_COLORS[dataset],
                    marker=POLICY_MARKERS[policy],
                    facecolors=(
                        DATASET_COLORS[dataset] if policy == "hard" else "white"
                    ),
                    edgecolors=DATASET_COLORS[dataset],
                    s=34,
                    linewidths=1.1,
                    zorder=3,
                )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(len(metrics)), [label for label, _ in metrics])
    ax.set_ylabel("MAPQ-off − genomic (×10⁻³)")
    ax.set_title("C  Whole-field morphology fit")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)

    dataset_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", color=color,
            markerfacecolor=color, label=dataset,
        )
        for dataset, color in DATASET_COLORS.items()
    ]
    policy_handles = [
        plt.Line2D(
            [0], [0], marker=marker, linestyle="none", color="#333333",
            markerfacecolor=("#333333" if policy == "hard" else "white"),
            label=("Hard" if policy == "hard" else "Soft expected"),
        )
        for policy, marker in POLICY_MARKERS.items()
    ]
    fig.legend(
        handles=dataset_handles + policy_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Flex MAPQ-off signal is consistent across independent H&E structures",
        fontsize=11,
    )

    output_paths = {
        "svg": args.out_dir / "flex_he_morphology_consistency.svg",
        "pdf": args.out_dir / "flex_he_morphology_consistency.pdf",
        "png": args.out_dir / "flex_he_morphology_consistency.png",
    }
    fig.savefig(output_paths["svg"], bbox_inches="tight")
    fig.savefig(output_paths["pdf"], bbox_inches="tight")
    fig.savefig(output_paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "visium_hd_processing.flex_he_morphology_consistency_figure.v1",
        "input": {
            "path": str(args.table.resolve()),
            "sha256": sha256(args.table),
        },
        "panels": {
            "A": "net MAPQ-off gain over genomic mass",
            "B": "positive increment minus genomic H&E mass fraction",
            "C": "whole-field H&E AUC and 16-um Spearman changes",
        },
        "outputs": {
            key: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in output_paths.items()
        },
        "interpretation_guardrails": {
            "raw_mass_not_normalized": True,
            "small_metric_sign_does_not_select_policy": True,
            "hard_and_soft_expected_both_shown": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (*output_paths.values(), summary_path)
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
