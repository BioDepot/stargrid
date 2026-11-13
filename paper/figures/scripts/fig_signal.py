#!/usr/bin/env python3
"""Fig. 6: the counts that STARgrid adds behave as signal (H&E of the same section; Xenium of an
adjacent section). Merges the H&E figure (fig4_cellpose.py) and the Xenium figure (fig5_xenium.py)
of the preprint into one.

a, share of the shared and the added counts that lies in cells, with the share of the area that
   cells occupy (what random placement would give), on the three slides.
b, cell ROC AUC of the shared counts, the added counts (STARgrid only) and the counts of Space Ranger only
   (the two Flex slides).
c, two-set diagrams of the counts in both and in one pipeline only (the two Flex slides), not to scale.
d-e, ovarian dataset against an adjacent-section Xenium Prime run: per-gene correlation, STARgrid
   (policy hard) minus Space Ranger, for the 20 cancer-panel genes (d); median correlation over them
   (e). The paper reports one policy, hard, and only this top-gene comparison (PI, 2026-10-27), so the
   other policies, the molecules panel, the cancer contrast and the patch separation of earlier
   versions are not drawn.

Inputs (accepted tables, policy hard): paper_results/spatch_rescore_ratio_20261109/
{three_slide_summary/he_shared_excess.tsv, flex_morphology_summary/flex_star_sr_he_consistency.tsv}
and the Xenium tables of paper_results/spatial_v1_9_5_20260915/ read by fig5_xenium.py.
The only numbers drawn are data values and axis scales; each is checked against the macros.
Usage: fig_signal.py [OUT_DIR]
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_macros  # noqa: E402
import spatial_paper_figstyle as S  # noqa: E402

ROOT = HERE.parents[2] / "paper_results"
HE = ROOT / "spatch_rescore_ratio_20261109"       # SPATCH rescored with the masks moved by the refitted displacement
XEN = ROOT / "spatial_v1_9_5_20260915"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
S.apply()
M = paper_macros.load()
SHARED = S.INK_MUTED
ADDED = S.STAR
SLIDES = [("CRC", "Colorectal"), ("SPATCH", "SPATCH"), ("OVARIAN_GEX", "Ovarian")]
FLEX = SLIDES[:2]
METHODS = [("Space Ranger", "space_ranger", S.SR), ("STARgrid", "hard", S.STAR)]


def require(condition, message):
    if not condition:
        sys.exit(f"fig_signal.py: {message}")


def rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def macro(name):
    return float(M[name].replace(",", ""))


# ------------------------------------------------------------------------------------------------
# data, checked against the manuscript's macros
# ------------------------------------------------------------------------------------------------
he = {(r["dataset"], r["component"]): r for r in rows(HE / "three_slide_summary/he_shared_excess.tsv")
      if r["method"] == "hard"}
flex = {r["dataset"]: r for r in rows(HE / "flex_morphology_summary/flex_star_sr_he_consistency.tsv")
        if r["method"] == "hard"}
in_cell, area, auc = {}, {}, {}
for key, _ in SLIDES:
    for part in ("shared", "star_excess"):
        r = he[(key, part)]
        in_cell[key, part] = 100 * float(r["cell_mass_fraction"])
        auc[key, part] = float(r["cell_roc_auc"])
        area[key, part] = in_cell[key, part] / float(r["cell_area_normalized_enrichment"])
    require(math.isclose(area[key, "shared"], area[key, "star_excess"], rel_tol=1e-9), f"{key}: two cell areas")
for key, name in (("CRC", "crcExcessDeficitPP"), ("SPATCH", "spatchExcessDeficitPP"), ("OVARIAN_GEX", "heExcessDeficitPP")):
    require(f"{in_cell[key, 'shared'] - in_cell[key, 'star_excess']:.2f}" == M[name], f"{name} differs from the table")
require(f"{auc['SPATCH', 'star_excess']:.3f}" == M["cpSPATCHStarOnlyCellAUC"], "SPATCH added-count AUC")
require(f"{auc['OVARIAN_GEX', 'star_excess']:.3f}" == M["heExcessAUC"], "ovarian added-count AUC")
for key, _ in FLEX:
    require(math.isclose(float(flex[key]["star_only_cell_roc_auc"]), auc[key, "star_excess"], abs_tol=1e-12),
            f"{key}: STARgrid-only counts are not the added counts")
sr_resid = {k: float(flex[k]["space_ranger_only_cell_roc_auc"]) for k, _ in FLEX}
resid_mass = {k: (float(flex[k]["star_positive_spatial_bin_residual_mass"]),
                  float(flex[k]["space_ranger_positive_spatial_bin_residual_mass"])) for k, _ in FLEX}
lo, hi = min(sr_resid.values()), max(sr_resid.values())
require(macro("cpSROnlyAUCLo") <= round(lo, 3) and round(hi, 3) <= macro("cpSROnlyAUCHi"),
        "Space Ranger-only AUCs outside the range the caption gives")

top = {r["method"]: r for r in rows(XEN / "xenium_compatibility_top20/top20_summary.tsv")}
genes = rows(XEN / "xenium_compatibility_top20/top20_genes.tsv")
for key, name in (("hard", "xenHardPearson"), ("space_ranger", "xenSRPearson")):
    require(f"{float(top[key]['median_spatial_pearson']):.3f}" == M[name], f"{name} differs from the table")
diffs = sorted(((float(g["hard_spatial_pearson"]) - float(g["space_ranger_spatial_pearson"]), g["gene_name"])
                for g in genes), reverse=True)
n = len(diffs)
wins = sum(d > 0 for d, _ in diffs)
require(all(d != 0 for d, _ in diffs), "a gene is tied; the sign test does not handle ties")
p = min(1.0, 2 * sum(math.comb(n, k) for k in range(max(wins, n - wins), n + 1)) / 2 ** n)
require(f"{p:.4f}" == M["xenHardSignP"] and str(wins) == M["xenHardWins"] and str(n) == M["xenTopGenes"],
        "the per-gene result does not match the manuscript's number macros")
sr_counts = float(top["space_ranger"]["summed_visium_counts"])
molecules = 100 * (float(top["hard"]["summed_visium_counts"]) - sr_counts) / sr_counts
require(f"{molecules:.1f}" == M["xenHardMassPct"], "xenHardMassPct differs from the table")

# ------------------------------------------------------------------------------------------------
# layout
# ------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(S.TEXT_WIDTH, 6.3))
top_row = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.05], wspace=0.45, left=0.10, right=0.985,
                           top=0.912, bottom=0.668)
low = fig.add_gridspec(1, 2, width_ratios=[1, 1.12], wspace=0.62, left=0.13, right=0.975, top=0.516,
                       bottom=0.087)
right = low[0, 1].subgridspec(2, 1, height_ratios=[1, 1.9])


def slide_axis(ax, keys):
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([name for _, name in keys])
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.tick_params(axis="x", length=0)
    S.hide_y_grid_behind(ax)


# --- a: share of counts in cells ----------------------------------------------------------------
ax_a = fig.add_subplot(top_row[0, 0])
for i, (key, _) in enumerate(SLIDES):
    ax_a.plot([i - 0.32, i + 0.32], [area[key, "shared"]] * 2, color=S.CHANCE, lw=0.9, ls=(0, (2.5, 1.5)), zorder=1)
    ax_a.plot([i - 0.11], [in_cell[key, "shared"]], "o", ms=4.6, color=SHARED, mec="white", mew=0.7, zorder=3)
    ax_a.plot([i + 0.11], [in_cell[key, "star_excess"]], "o", ms=4.6, color=ADDED, mec="white", mew=0.7, zorder=3)
slide_axis(ax_a, SLIDES)
A_LIM = (20, 95)
require(all(A_LIM[0] < v < A_LIM[1] for v in list(in_cell.values()) + list(area.values())), "panel a range")
ax_a.set_ylim(*A_LIM)
ax_a.set_ylabel("Counts in cells (%)")

# --- b: cell ROC AUC ------------------------------------------------------------------------------
ax_b = fig.add_subplot(top_row[0, 1])
ax_b.axhline(0.5, color=S.CHANCE, lw=0.6, ls=(0, (4, 3)), zorder=1)
for i, (key, _) in enumerate(SLIDES):
    ax_b.plot([i - 0.17], [auc[key, "shared"]], "o", ms=4.6, color=SHARED, mec="white", mew=0.7, zorder=3)
    ax_b.plot([i], [auc[key, "star_excess"]], "o", ms=4.6, color=ADDED, mec="white", mew=0.7, zorder=3)
    if key in sr_resid:
        ax_b.plot([i + 0.17], [sr_resid[key]], "o", ms=4.6, color=S.SR, mec="white", mew=0.7, zorder=3)
slide_axis(ax_b, SLIDES)
B_LIM = (0.45, 0.92)
require(all(B_LIM[0] < v < B_LIM[1] for v in list(auc.values()) + list(sr_resid.values())), "panel b range")
ax_b.set_ylim(*B_LIM)
ax_b.text(len(SLIDES) - 0.47, 0.505, "chance", fontsize=S.FS_NOTE, color=S.CHANCE, ha="right", va="bottom")
ax_b.set_ylabel("Cell ROC AUC")

# --- c: counts in both pipelines and in one only, as two-set diagrams (not to scale; the numbers give the sizes) -----
# In each 8 um bin, the counts in both are the smaller of the two pipelines' counts; the rest of each pipeline's count is that
# pipeline's only (STARgrid's are the added counts). Drawn to scale, the parts in one pipeline only would be slivers.
from matplotlib.patches import Circle  # noqa: E402

venn = {}
for key, _ in FLEX:
    star_only, sr_only = resid_mass[key]
    both = float(flex[key]["space_ranger_mass"]) - sr_only
    require(math.isclose(both, float(flex[key]["star_mass_on_common_axis"]) - star_only, abs_tol=0.5),
            f"{key}: the shared counts differ between the two pipelines' totals")
    venn[key] = (both, star_only, sr_only)

ax_c = fig.add_subplot(top_row[0, 2])
ax_c.set_aspect("equal")
ax_c.axis("off")
R, D, STEP = 1.0, 0.45, 2.75                    # circle radius, half the centre distance, vertical spacing of the diagrams


def millions(value):
    return f"{value / 1e6:.{1 if value >= 1e7 else 2}f}"


for i, (key, name) in enumerate(FLEX):
    both, star_only, sr_only = venn[key]
    y0 = -i * STEP
    star_disc = Circle((-D, y0), R, facecolor=ADDED, edgecolor="white", lw=0.8, zorder=1)
    sr_disc = Circle((D, y0), R, facecolor=S.SR, edgecolor="white", lw=0.8, zorder=2)
    ax_c.add_patch(star_disc)
    ax_c.add_patch(sr_disc)
    middle = Circle((-D, y0), R, facecolor=SHARED, edgecolor="white", lw=0.8, zorder=3)
    ax_c.add_patch(middle)
    middle.set_clip_path(sr_disc)
    ax_c.text(0, y0, millions(both), ha="center", va="center", fontsize=S.FS_NOTE, color="white", zorder=4)
    ax_c.text(-1.0, y0, millions(star_only), ha="center", va="center", fontsize=S.FS_NOTE, color="white", zorder=4)
    ax_c.text(1.0, y0, millions(sr_only), ha="center", va="center", fontsize=S.FS_NOTE, color=S.INK, zorder=4)
    ax_c.text(0, y0 + R + 0.18, name, ha="center", va="bottom", fontsize=S.FS_TICK, color=S.INK_MUTED)
ax_c.text(0, -(len(FLEX) - 1) * STEP - R - 0.3, "millions of molecules;\nnot to scale", ha="center", va="top",
          fontsize=S.FS_NOTE, color=S.INK_MUTED, linespacing=1.15)
ax_c.set_xlim(-D - R - 0.1, D + R + 0.1)
ax_c.set_ylim(-(len(FLEX) - 1) * STEP - R - 1.25, R + 0.75)

# --- d: per-gene correlation, hard minus Space Ranger --------------------------------------------
ax_d = fig.add_subplot(low[0, 0])
for y, (d, name) in enumerate(diffs):
    ax_d.barh(y, d, height=0.62, color=S.STAR if d > 0 else S.SR)
ax_d.axvline(0, color=S.CHANCE, lw=0.6)
ax_d.set_yticks(range(n))
ax_d.set_yticklabels([name for _, name in diffs], fontstyle="italic")
ax_d.set_ylim(n - 0.4, -0.6)
ax_d.tick_params(axis="y", length=0)
ax_d.set_xlabel("Correlation with Xenium,\nSTARgrid minus Space Ranger")
ax_d.xaxis.grid(True)
ax_d.set_axisbelow(True)
ax_d.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (S.STAR, S.SR)],
            labels=[f"STARgrid closer ({wins})", f"Space Ranger closer ({n - wins})"],
            loc="lower right", frameon=False, borderaxespad=0.2, fontsize=S.FS_NOTE)


def dots(ax, values, xlabel, fmt):
    for i, (name, key, colour) in enumerate(METHODS):
        ax.scatter([values[key]], [i], s=20, color=colour, zorder=3)
        ax.text(values[key], i, "   " + fmt.format(values[key]), va="center", ha="left", fontsize=S.FS_NOTE, color=S.INK)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([m[0] for m in METHODS])
    ax.set_ylim(len(METHODS) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


# --- e: median correlation ------------------------------------------------------------------------
ax_e = fig.add_subplot(right[0, 0])
dots(ax_e, {k: float(v["median_spatial_pearson"]) for k, v in top.items()}, "Median Pearson r with Xenium", "{:.3f}")
ax_e.set_xlim(0.23, 0.265)
ax_e.set_xticks([0.23, 0.24, 0.25, 0.26])


# ------------------------------------------------------------------------------------------------
# panel heads: letter and title on one baseline, as in Figs 4 and 5
# ------------------------------------------------------------------------------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()


def left_edge(ax):
    return inv.transform((ax.get_tightbbox(renderer).x0, 0))[0]


def top_edge(ax):
    box = ax.get_legend().get_window_extent(renderer) if ax.get_legend() else ax.get_window_extent(renderer)
    return max(ax.get_position().y1, inv.transform((0, box.y1))[1])


def head(x, y, letter, title):
    fig.text(x, y, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
    fig.text(x + 0.024, y, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)


top_y = max(ax.get_position().y1 for ax in (ax_a, ax_b, ax_c))
fig.text(0.0, top_y + 0.048, "H&E of the same section", fontsize=S.FS_LABEL, fontweight="bold", color=S.INK,
         va="baseline")
fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.6, color=SHARED),
                    Line2D([], [], ls="none", marker="o", ms=4.6, color=ADDED),
                    Line2D([], [], ls="none", marker="o", ms=4.6, color=S.SR),
                    Line2D([], [], color=S.CHANCE, lw=0.9, ls=(0, (2.5, 1.5)))],
           labels=["shared counts", "added counts (STARgrid only)", "Space Ranger only",
                   "fraction of area in cells"],
           loc="lower right", bbox_to_anchor=(0.985, top_y + 0.036), ncol=4, frameon=False, borderaxespad=0.0,
           fontsize=S.FS_NOTE, handlelength=1.2, columnspacing=0.7, handletextpad=0.4)
y1 = max(ax.get_position().y1 for ax in (ax_a, ax_b, ax_c)) + 0.014
for ax, letter, title in ((ax_a, "a", "Counts in cells"), (ax_b, "b", "Separating cells"),
                          (ax_c, "c", "Overlap of counts")):
    head(left_edge(ax), y1, letter, title)
fig.text(0.0, ax_d.get_position().y1 + 0.052, "Xenium of an adjacent section, ovarian dataset",
         fontsize=S.FS_LABEL, fontweight="bold", color=S.INK, va="baseline")
head(left_edge(ax_d), ax_d.get_position().y1 + 0.014, "d",
     f"Cancer genes: {wins} of {n} closer, P = {M['xenHardSignP']}")
head(left_edge(ax_e), top_edge(ax_e) + 0.012, "e", "Median r, same genes")

S.save(fig, OUT, "fig_signal")
print(f"STARgrid closer on {wins} of {n}, P {p:.4f}; Space Ranger-only AUC {lo:.3f}-{hi:.3f}")
