#!/usr/bin/env python3
"""Render the five-panel paper morphology figure from an accepted result table."""

import sys
import argparse
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import spatial_paper_figstyle as fs

POLICY = "hard"           # primary policy; soft is reported in the caption
DATASETS = ["CRC", "SPATCH"]
LABELS = {"CRC": "Colorectal\n(10x)", "SPATCH": "Ovarian\n(SPATCH)"}


def load(source):
    df = pd.read_csv(source, sep="\t")
    df = df[df["method"] == POLICY].set_index("dataset")
    missing = [d for d in DATASETS if d not in df.index]
    if missing:
        raise SystemExit(f"missing datasets in {source}: {missing}")
    if df.index.duplicated().any():
        raise ValueError("duplicate hard-policy dataset rows")
    return df.loc[DATASETS]


def panel_excess(ax, df):
    """a --- mass excess over Space Ranger, measured on the vendor barcode axis."""
    x = np.arange(len(DATASETS))
    v = df["star_percent_excess_over_space_ranger"].to_numpy()
    ax.bar(x, v, width=0.5, color=fs.STAR, linewidth=0)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.18, f"{vi:+.2f}%", ha="center", va="bottom",
                fontsize=6, color=fs.INK)
    ax.set_xticks(x, [LABELS[d] for d in DATASETS])
    ax.set_ylabel("Mass excess over\nSpace Ranger (%)")
    span = max(1.0, float(np.max(np.abs(v))))
    ax.set_ylim(min(0.0, float(v.min())) - 0.15 * span, max(0.0, float(v.max())) + 0.28 * span)
    ax.axhline(0, color=fs.GRID, lw=0.6)
    ax.set_title("Mass difference", color=fs.INK)
    fs.hide_y_grid_behind(ax)
    fs.panel_label(ax, "a", dx=-0.27)


def panel_concordance(ax, df):
    """b --- Pearson concordance on the common vendor axis.

    Dots encode position; the axis expands to include every observed value.
    """
    x = np.arange(len(DATASETS))
    off = 0.16
    gene = df["gene_total_pearson"].to_numpy()
    spat = df["spatial_bin_total_pearson"].to_numpy()
    ax.axhline(1.0, color=fs.GRID, lw=0.6, zorder=1)
    for xi, g, s in zip(x, gene, spat):
        ax.plot(xi - off, g, "o", ms=5, color=fs.STAR, mec="white", mew=0.9, zorder=3)
        ax.plot(xi + off, s, "D", ms=4.2, color=fs.STAR_SOFT, mec="white", mew=0.9, zorder=3)
        ax.text(xi - off, g + 0.0011, f"{g:.4f}", ha="center", va="bottom",
                fontsize=5.6, color=fs.INK)
        ax.text(xi + off, s + 0.0011, f"{s:.4f}", ha="center", va="bottom",
                fontsize=5.6, color=fs.INK)
    ax.set_xticks(x, [LABELS[d] for d in DATASETS])
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    lower = min(0.9955, float(min(gene.min(), spat.min())) - 0.01)
    ax.set_ylim(max(-1.05, lower), 1.005)
    ax.set_ylabel("Pearson $r$ vs\nSpace Ranger")
    ax.set_title("Field concordance", color=fs.INK)
    handles = [
        plt.Line2D([], [], marker="o", ms=5, ls="none", color=fs.STAR,
                   mec="white", mew=0.9, label="Gene totals"),
        plt.Line2D([], [], marker="D", ms=4.2, ls="none", color=fs.STAR_SOFT,
                   mec="white", mew=0.9, label="8 µm spatial bins"),
    ]
    # Inside the axes: a legend floating between rows reads as if it belonged
    # to the row below it.
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.03, -0.04),
              ncol=1)
    fs.hide_y_grid_behind(ax)
    fs.panel_label(ax, "b", dx=-0.34)


def panel_residual_mass(ax, df):
    """c --- molecules each method has that the other does not."""
    x = np.arange(len(DATASETS))
    w = 0.34
    star = df["star_positive_spatial_bin_residual_mass"].to_numpy()
    sr = df["space_ranger_positive_spatial_bin_residual_mass"].to_numpy()
    ratio = df["star_to_space_ranger_positive_spatial_residual_mass_ratio"].to_numpy()
    ax.bar(x - w / 2, star / 1e6, width=w - 0.02, color=fs.STAR, linewidth=0)
    ax.bar(x + w / 2, sr / 1e6, width=w - 0.02, color=fs.SR, linewidth=0)
    for xi, a, b, r in zip(x, star, sr, ratio):
        ax.text(xi - w / 2, a / 1e6 + 0.22, f"{a/1e6:.1f}M", ha="center",
                va="bottom", fontsize=5.6, color=fs.INK)
        ax.text(xi + w / 2, b / 1e6 + 0.22, f"{b/1e6:.2f}M", ha="center",
                va="bottom", fontsize=5.6, color=fs.INK)
        ax.text(xi, max(a, b) / 1e6 + 1.55, f"{r:.1f}:1", ha="center",
                va="bottom", fontsize=6.5, color=fs.INK, fontweight="bold")
    ax.set_xticks(x, [LABELS[d] for d in DATASETS])
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylabel("Residual mass\n(10$^6$ molecules)")
    ax.set_ylim(0, max(star.max(), sr.max()) / 1e6 * 1.30)
    ax.set_title("Positive spatial residuals", color=fs.INK)
    fs.hide_y_grid_behind(ax)
    fs.panel_label(ax, "c", dx=-0.27)


def _auc_panel(ax, df, star_col, sr_col, title, letter, show_legend):
    """Lollipop from the chance line. Stem length IS distance from chance, so
    the axis can start at 0.45 without implying a magnitude from zero."""
    x = np.arange(len(DATASETS))
    off = 0.17
    star = df[star_col].to_numpy()
    sr = df[sr_col].to_numpy()

    ax.axhline(0.5, color=fs.CHANCE, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    ax.text(len(DATASETS) - 0.42, 0.503, "chance", fontsize=5.8,
            color=fs.CHANCE, va="bottom", ha="right")

    for xi, a, b in zip(x, star, sr):
        ax.plot([xi - off, xi - off], [0.5, a], color=fs.STAR, lw=1.4,
                solid_capstyle="round", zorder=2)
        ax.plot([xi + off, xi + off], [0.5, b], color=fs.SR, lw=1.4,
                solid_capstyle="round", zorder=2)
        # 2 px surface ring keeps markers legible where they meet the line
        ax.plot(xi - off, a, "o", ms=5, color=fs.STAR, mec="white", mew=0.9, zorder=3)
        ax.plot(xi + off, b, "o", ms=5, color=fs.SR, mec="white", mew=0.9, zorder=3)
        ax.text(xi - off, a + 0.012, f"{a:.3f}", ha="center", va="bottom",
                fontsize=6, color=fs.INK)
        ax.text(xi + off, b - 0.016, f"{b:.3f}", ha="center", va="top",
                fontsize=6, color=fs.INK)

    ax.set_xticks(x, [LABELS[d] for d in DATASETS])
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylim(max(-0.05, min(0.45, float(min(star.min(), sr.min())) - 0.06)),
                min(1.05, max(0.75, float(max(star.max(), sr.max())) + 0.06)))
    ax.set_yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    ax.set_ylabel("ROC AUC of residual field")
    ax.set_title(title, color=fs.INK)
    fs.hide_y_grid_behind(ax)
    fs.panel_label(ax, letter, dx=-0.20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit("refusing existing figure output")
    fs.apply()
    df = load(args.table)
    OUT = args.out_dir

    # 6 logical columns so the bottom row splits evenly (3+3) while the top
    # row splits into thirds (2+2+2); a 3-column grid cannot do both.
    fig = plt.figure(figsize=(fs.DOUBLE_COL, 3.75))
    gs = fig.add_gridspec(
        2, 6, height_ratios=[1.0, 1.1], hspace=0.66, wspace=1.60,
        left=0.085, right=0.985, top=0.855, bottom=0.085,
    )
    panel_excess(fig.add_subplot(gs[0, 0:2]), df)
    panel_concordance(fig.add_subplot(gs[0, 2:4]), df)
    panel_residual_mass(fig.add_subplot(gs[0, 4:6]), df)
    _auc_panel(fig.add_subplot(gs[1, 0:3]), df,
               "star_only_cell_roc_auc", "space_ranger_only_cell_roc_auc",
               "Cell ranking of the residual fields", "d", True)
    _auc_panel(fig.add_subplot(gs[1, 3:6]), df,
               "star_only_nucleus_roc_auc", "space_ranger_only_nucleus_roc_auc",
               "Nucleus ranking of the residual fields", "e", False)

    shared = [
        plt.Line2D([], [], marker="o", ms=5.5, ls="none", color=fs.STAR,
                   mec="white", mew=0.9, label="STAR Suite only"),
        plt.Line2D([], [], marker="o", ms=5.5, ls="none", color=fs.SR,
                   mec="white", mew=0.9, label="Space Ranger only"),
    ]
    fig.legend(handles=shared, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=2, fontsize=6.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_cellpose.{ext}")
    plt.close(fig)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"schema": "visium_hd_processing.paper_cellpose_figure.v1",
               "policy": POLICY, "spatial_bin_um": 8,
               "input": {"path": str(args.table.resolve()), "sha256": digest(args.table)},
               "style_sha256": digest(Path(fs.__file__)),
               "outputs": {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT/'fig4_cellpose.pdf'} and .png")

    # Echo the plotted values so the caption can be checked against the figure.
    cols = ["star_percent_excess_over_space_ranger", "star_only_cell_roc_auc",
            "space_ranger_only_cell_roc_auc", "star_only_nucleus_roc_auc",
            "space_ranger_only_nucleus_roc_auc",
            "star_to_space_ranger_positive_spatial_residual_mass_ratio"]
    print(df[cols].to_string())


if __name__ == "__main__":
    main()
