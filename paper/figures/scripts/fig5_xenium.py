#!/usr/bin/env python3
"""Fig. 5: adjacent-section Xenium on the ovarian 3' slide.

Inputs (accepted v1.9.5 tables): xenium_compatibility_cancer/method_summary.tsv,
xenium_compatibility_top20/top20_summary.tsv and top20_genes.tsv,
xenium_campaign_summary/summary.json (bootstrap intervals, both modes).
Usage: fig5_xenium.py [OUT_DIR]
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_paper_figstyle as S  # noqa: E402

R = HERE.parents[2] / "paper_results" / "spatial_v1_9_5_20260915"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
S.apply()
STRICT = S.STAR_SOFT
METHODS = [("Space\nRanger", "space_ranger", S.SR), ("strict", "strict", STRICT), ("hard", "hard", S.STAR)]


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


effect = {r["method"]: float(r["effect_pearson_vs_xenium"])
          for r in rows(R / "xenium_compatibility_cancer/method_summary.tsv")
          if r["scope"] == "label_leakage_controlled"}
top = {r["method"]: r for r in rows(R / "xenium_compatibility_top20/top20_summary.tsv")}
genes = rows(R / "xenium_compatibility_top20/top20_genes.tsv")
campaign = json.load(open(R / "xenium_campaign_summary/summary.json"))["cancer_panel"]

fig = plt.figure(figsize=(S.DOUBLE_COL, 118 * S.MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.55)
top_row = outer[0].subgridspec(1, 3, wspace=0.62)
bottom = outer[1].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.62)


def dots(ax, values, ylabel, fmt):
    for i, (name, key, colour) in enumerate(METHODS):
        ax.scatter([i], [values[key]], s=26, color=colour, zorder=3)
        ax.text(i, values[key], "  " + fmt.format(values[key]), va="center", ha="left", fontsize=5.6, color=S.INK)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m[0] for m in METHODS], fontsize=6)
    ax.set_xlim(-0.5, len(METHODS) - 0.2)
    ax.set_ylabel(ylabel)
    S.hide_y_grid_behind(ax)


# --- a: cancer-panel effect correlation ---------------------------------------------------
ax = fig.add_subplot(top_row[0, 0])
S.panel_label(ax, "a", dx=-0.42, dy=1.16)
ax.set_title("Cancer-panel effects", pad=6)
dots(ax, effect, "Pearson r with Xenium", "{:.3f}")
ax.set_ylim(0.85, 0.872)

# --- b: top-20 median spatial correlation ---------------------------------------------------
ax = fig.add_subplot(top_row[0, 1])
S.panel_label(ax, "b", dx=-0.42, dy=1.16)
ax.set_title("Top 20 genes, spatial r", pad=6)
dots(ax, {k: float(v["median_spatial_pearson"]) for k, v in top.items()}, "Median Pearson r with Xenium", "{:.3f}")
ax.set_ylim(0.23, 0.262)

# --- c: top-20 molecules relative to Space Ranger ----------------------------------------------
ax = fig.add_subplot(top_row[0, 2])
S.panel_label(ax, "c", dx=-0.42, dy=1.16)
ax.set_title("Top 20 genes, molecules", pad=6)
sr_counts = float(top["space_ranger"]["summed_visium_counts"])
for i, (name, key, colour) in enumerate(METHODS[1:]):
    v = 100 * (float(top[key]["summed_visium_counts"]) - sr_counts) / sr_counts
    ax.bar(i, v, width=0.55, color=colour)
    ax.text(i, v + (0.8 if v > 0 else -0.8), f"{v:+.1f}%", ha="center", va="bottom" if v > 0 else "top",
            fontsize=5.6, color=S.INK)
ax.axhline(0, color=S.CHANCE, lw=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["strict", "hard"], fontsize=6)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-17, 12)
ax.set_ylabel("% difference from Space Ranger")
S.hide_y_grid_behind(ax)

# --- d: per-gene differences -------------------------------------------------------------------
ax = fig.add_subplot(bottom[0, 0])
S.panel_label(ax, "d", dx=-0.20, dy=1.10)
wins = {k: 0 for k in ("strict", "hard")}
for g in genes:
    rank = int(g["xenium_abundance_rank"])
    sr = float(g["space_ranger_spatial_pearson"])
    for key, colour, off in (("strict", STRICT, 0.17), ("hard", S.STAR, -0.17)):
        d = float(g[f"{key}_spatial_pearson"]) - sr
        wins[key] += d > 0
        ax.scatter([d], [rank + off], s=10, color=colour, zorder=3)
ax.set_title(f"Per gene: hard closer on {wins['hard']}/20, strict on {wins['strict']}/20", pad=6)
ax.axvline(0, color=S.CHANCE, lw=0.6, ls="--")
ax.set_yticks([int(g["xenium_abundance_rank"]) for g in genes])
ax.set_yticklabels([g["gene_name"] for g in genes], fontsize=5.2)
ax.set_xlim(-0.037, 0.047)
ax.invert_yaxis()
ax.set_xlabel("Spatial r with Xenium, minus Space Ranger's")
ax.set_ylabel("Genes by Xenium abundance")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, ms=3) for c in (S.STAR, STRICT)],
          labels=["hard", "strict"], loc="lower right", fontsize=5.6, frameon=False)

# --- e: cancer-patch discrimination, bootstrap intervals -------------------------------------------
ax = fig.add_subplot(bottom[0, 1])
S.panel_label(ax, "e", dx=-0.52, dy=1.10)
ax.set_title("Cancer-patch discrimination", pad=6)
scope_name = {"full_reference_panel": "full panel", "exclude_mecom": "without MECOM",
              "label_leakage_controlled": "leakage-controlled"}
ticks, labels, y = [], [], 0
for mode in ("compatibility", "annotated"):
    ax.text(-0.0049, y - 0.2, f"{mode} mode", fontsize=5.4, color=S.INK, fontweight="bold",
            ha="left", va="bottom")
    y += 0.9
    for iv in campaign[mode]["intervals"]:
        colour = S.STAR if iv["method"] == "hard" else STRICT
        ax.plot([iv["ci95_low"], iv["ci95_high"]], [y, y], color=colour, lw=1.2)
        ax.scatter([iv["mean_delta"]], [y], s=9, color=colour, zorder=3)
        ticks.append(y)
        labels.append(f"{iv['method']}, {scope_name[iv['scope']]}")
        y += 1
    y += 0.6
ax.axvline(0, color=S.CHANCE, lw=0.6, ls="--")
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=5.2)
ax.set_ylim(y - 0.3, -0.6)
ax.set_xlim(-0.005, 0.005)
ax.set_xlabel("Mean AUC change per gene (95% CI)")
ax.xaxis.grid(True)
ax.set_axisbelow(True)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig5_xenium.{ext}")
print("wrote", OUT / "fig5_xenium.pdf", "hard wins", wins["hard"], "strict wins", wins["strict"])
