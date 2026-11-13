#!/usr/bin/env python3
"""Supplementary Fig. S1: averages over all nuclei can hide the dip, so the check moves the masks.

a, mean count per capture location against distance from the edge of the nearest nucleus, in the
   nucleus maps deposited with the ovarian dataset and with the colorectal slide (the R03 panel
   "Counts around nuclei", extended to the colorectal slide). Post hoc, exploratory.
b, mean count under the deposited nucleus map of the colorectal slide at every shift: a narrow dip
   at zero inside a broad fall towards large shifts, which is why the dip is an offset lower than its
   neighbouring shifts and not the lowest value of the search.

Inputs: paper_results/registration_check_20261018/edge_profile_{ovarian,colorectal}.tsv and
paper_results/registration_local_dip_20261106/placements/crc_published (surface and dip of the deposited map). Shifts are (dy, dx)
in capture locations of 2 um, rows downward. The only numbers drawn are axis scales.
Usage: fig_supp_dip.py [OUT_DIR]
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_paper_figstyle as S  # noqa: E402

ROOT = HERE.parents[2]
R = ROOT / "paper_results" / "registration_check_20261018"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
UM = 2.0
NEWR = ROOT / "paper_results" / "registration_local_dip_20261106" / "placements"
S.apply()
SR4 = S.SR
COUNT_MAP = LinearSegmentedColormap.from_list("counts", ["#f7f7f8", "#c9ccd2", "#7c818b", "#2f343c"])
NUCLEUS_FILL = "#ece8f3"      # light tint of the nucleus purple


def require(condition, message):
    if not condition:
        sys.exit(f"fig_supp_dip.py: {message}")


def tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def bin_centre(label):
    """Centre (um) of a distance bin; negative inside the nucleus. Open-ended bins return None.

    Same centres as the R03 figure: 'at least 4 um inside' at -5 and the edge location at -1.
    """
    if label == "nucleus, at least 4 um inside":
        return -5.0
    if label == "nucleus, edge square":
        return -1.0
    m = re.fullmatch(r"(nucleus|outside nucleus), (\d+)-(\d+) um( inside)?", label)
    if not m:
        return None                                   # 'beyond 50 um': no finite centre
    lo, hi = float(m.group(2)), float(m.group(3))
    centre = (lo + hi) / 2
    return -centre if m.group(1) == "nucleus" else centre


profiles = []
for name, label, colour, style, marker in (("edge_profile_ovarian.tsv", "Ovarian", S.INK, "-", "o"),
                                           ("edge_profile_colorectal.tsv", "Colorectal", S.INK_MUTED, "--", "s")):
    rows = tsv(R / name)
    points = sorted((bin_centre(r["bin"]), float(r["mean_count"])) for r in rows if bin_centre(r["bin"]) is not None)
    require(len(points) == len(rows) - 1, f"{name}: expected exactly one open-ended bin")
    profiles.append((label, colour, style, marker, points))

import json
pub_rows = json.loads((NEWR / "crc_published.rows.json").read_text())
pub_whole = [r for r in pub_rows if r["surface"] == "whole"][0]
require(pub_whole["dip"] and pub_whole["usable"] and (pub_whole["dy"], pub_whole["dx"]) == (0, 0),
        "colorectal deposited map: dip not usable at zero shift")
full = np.load(NEWR / "crc_published.surfaces.npz")["whole"]
half = 8                                        # drawn window, each way (the search is 12)
c0 = full.shape[0] // 2
surface = full[c0 - half:c0 + half + 1, c0 - half:c0 + half + 1]
low = np.unravel_index(np.argmin(surface), surface.shape)
low_at = (int(low[0]) - half, int(low[1]) - half)

fig = plt.figure(figsize=(S.TEXT_WIDTH, 2.35))
grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.05], wspace=0.34, left=0.085, right=0.70, top=0.86,
                        bottom=0.19)

# --- a: count against distance from the nucleus edge, both slides ------------------------------
ax = fig.add_subplot(grid[0, 0])
inside = min(x for *_, pts in profiles for x, _ in pts) - 1.5
ax.axvspan(inside, 0, color=NUCLEUS_FILL, lw=0, zorder=0)
ax.text(inside / 2, 1.0, "nucleus", ha="center", va="bottom", rotation=90, fontsize=S.FS_NOTE, color=S.INK_MUTED)
ax.set_ylim(0, 1.1 * max(y for *_, pts in profiles for _, y in pts))
for label, colour, style, marker, points in profiles:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=colour, ls=style, marker=marker, ms=2.6, lw=1.0, zorder=3)
    if ys[-1] > 0.1 * ax.get_ylim()[1] or label == "Ovarian":
        ax.text(xs[-1] + 1.2, ys[-1], label, fontsize=S.FS_NOTE, color=colour, va="center", ha="left")
    else:                                         # the line ends near zero: label it above its tail
        ax.text(xs[-2], ys[-2] + 1.2, label, fontsize=S.FS_NOTE, color=colour, va="bottom", ha="left")
far = max(x for *_, pts in profiles for x, _ in pts)
ax.set_xlim(inside, far + 13)
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel("Distance from the nucleus edge (µm)")
ax.set_ylabel("Mean count per location")
S.hide_y_grid_behind(ax)
ax_a = ax

# --- b: the colorectal shift surface under the deposited map ------------------------------------
ax = fig.add_subplot(grid[0, 1])
norm = Normalize(np.nanmin(surface), np.nanmax(surface))
edge = UM * (half + 0.5)
ax.imshow(surface, cmap=COUNT_MAP, norm=norm, extent=(-edge, edge, edge, -edge), origin="upper",
          interpolation="nearest")
ax.plot([0], [0], marker="x", ms=4.5, mew=1.1, color=SR4)
ax.plot([UM * low_at[1]], [UM * low_at[0]], marker="o", ms=4.5, mfc="none", mec=S.INK, mew=1.0)
ax.set_xticks([-16, -8, 0, 8, 16])
ax.set_yticks([-16, -8, 0, 8, 16])
ax.tick_params(length=2)
ax.set_xlabel("Column shift (µm)")
ax.set_ylabel("Row shift (µm)")
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
ax_b = ax
cax = fig.add_subplot(grid[0, 2])
fig.canvas.draw()
pos, cpos = ax_b.get_position(), cax.get_position()
cax.set_position([pos.x1 + 0.015, pos.y0, cpos.width, pos.height])
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=COUNT_MAP), cax=cax)
bar.ax.tick_params(labelsize=S.FS_NOTE, length=2)
bar.outline.set_linewidth(0.4)
bar.set_label("Mean count under the masks", fontsize=S.FS_NOTE, labelpad=3)
ax_b.legend(handles=[plt.Line2D([], [], ls="none", marker="x", ms=4.5, mew=1.1, color=SR4),
                     plt.Line2D([], [], ls="none", marker="o", ms=4.5, mfc="none", mec=S.INK, mew=1.0)],
            labels=["deepest dip, at zero shift", "lowest value in the window"], loc="upper left",
            bbox_to_anchor=(1.42, 1.02), frameon=False, borderaxespad=0.0, fontsize=S.FS_NOTE, handlelength=1.4,
            labelspacing=0.5)

# --- panel heads ---------------------------------------------------------------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
y = max(ax_a.get_position().y1, ax_b.get_position().y1) + 0.05
for ax, letter, title in ((ax_a, "a", "Counts around nuclei, by slide"),
                          (ax_b, "b", "Colorectal, deposited nucleus map")):
    x = inv.transform((ax.get_tightbbox(renderer).x0, 0))[0]
    fig.text(x, y, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
    fig.text(x + 0.024, y, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)

S.save(fig, OUT, "fig_supp_dip")
print(f"panel b: lowest value in the window at {low_at} (locations); the deepest dip is at zero shift")
