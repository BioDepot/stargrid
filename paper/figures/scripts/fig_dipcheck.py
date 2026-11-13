#!/usr/bin/env python3
"""Fig. 4: how the counts around nuclei register the image, and Supplemental Fig. S2.

a, schematic in four steps (expanded from Fig. 1a, step 3, and from the R03 strategy figure):
   nucleus masks placed on the grid; counts lower under each nucleus than just around it
   (a local hollow, not fewer counts in nuclei overall); the masks moved over the grid; the
   mean count under the masks against shift traces a dip, an offset lower than both shifts beside it,
   whose shift is the offset of the registration; a mask of the ring of higher counts around the nuclei
   traces a peak at the same shift (grey).
b, mean count under our Cellpose masks of the ovarian dataset at every shift, with the masks
   placed by Space Ranger's registration and by our registration from the images alone; the cross
   is the deepest dip.
c, the diagonal of b (the panel that was Fig. 4a of the preprint draft, unchanged).
d, counts across the nucleus edge: at the shift the correlation ratio finds, the mean corrected count (each count
   divided by the mean of the supported locations within 2 locations on each axis) in each band of distance from
   the nucleus edge (I3, I2, I1 inside; O1-O6 outside; far), for five whole slides.
e, the correlation ratio eta^2 (the share of the variance of the corrected counts explained by the bands) at every
   shift, cropped to +-8 locations: the post-Xenium lung section with H&E nuclei, and the kidney.
Supplemental Fig. S2 (fig_supp_pairdepth), drawn here from the same data, was panel e: nucleus by
   nucleus, the median paired depth of nuclei in place and of the same nuclei moved far away
   (background), and their difference.

Inputs, all under paper_results/: registration_correlation_ratio_20261108/ (panels d and e, read by
registration_ratio.py), registration_local_dip_20261106/ (surfaces and dips of panels a-c),
registration_check_20261018/ (lung mask displacements) and paper_tests_20261021/ (far-shift control and
Amendment 6 summaries). Shifts are (dy, dx) in
capture locations of 2 um, rows downward. The only numbers drawn are axis scales.
Usage: fig_dipcheck.py [OUT_DIR]
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_paper_figstyle as S  # noqa: E402
import registration_ratio as CR  # noqa: E402

ROOT = HERE.parents[2]
R = ROOT / "paper_results" / "registration_check_20261018"
T = ROOT / "paper_results" / "paper_tests_20261021"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
UM = 2.0                      # one capture location
HALF = 8                      # half-width of the drawn shift surfaces, in locations
NEWR = ROOT / "paper_results" / "registration_local_dip_20261106" / "placements"
SEARCH = 12                   # the dip is the deepest within 12 locations
S.apply()
OURS = S.STAR
SR4 = S.SR
NUCLEUS = "#6d4c9f"
COUNT = "#4b5563"             # grey of the counts, as in Fig. 1a
GRID_EDGE = "#c3c7ce"
COUNT_MAP = LinearSegmentedColormap.from_list("counts", ["#f7f7f8", "#c9ccd2", "#7c818b", "#2f343c"])
MOVED = "#9ca3af"


def require(condition, message):
    if not condition:
        sys.exit(f"fig_dipcheck.py: {message}")


def load(name, base=R):
    return json.loads((base / name).read_text())


def tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def whole_dip(name):
    base = CR.placements_dir(name)          # the Cellpose-SAM placements and spatch_after_ratio have their own records
    rows = json.loads((base / f"{name}.rows.json").read_text())
    meta = json.loads((base / f"{name}.meta.json").read_text())
    require(meta["all_reproduced"] and meta["search"] == SEARCH, f"{name}: not a reproduced run of the final rule")
    row = [r for r in rows if r["surface"] == "whole"][0]
    require(row["dip"] and row["interior"] and row["usable"], f"{name}: whole-slide dip not usable")
    return row, np.load(base / f"{name}.surfaces.npz")["whole"]


m0_ov, surface_sr = whole_dip("ov_cellpose_dep")
centre = surface_sr.shape[0] // 2
require((m0_ov["dy"], m0_ov["dx"]) == (0, 0), "ovarian Cellpose dip under Space Ranger's placement not at zero")
sr_crop = surface_sr[centre - HALF:centre + HALF + 1, centre - HALF:centre + HALF + 1]

ours_row, ours_full = whole_dip("ov_cellpose_img")
ours = ours_full[centre - HALF:centre + HALF + 1, centre - HALF:centre + HALF + 1]
require(not np.isnan(ours).any(), "our-placement surface incomplete")
ours_min = (ours_row["dy"], ours_row["dx"])
recorded = load("ovarian_dip_our_placement.json")["maps"]["native_cellpose"]["minimum"]
require(ours_min == (recorded["dy"], recorded["dx"]) == (-3, -3), "our-placement dip differs from the first analysis")
sr_min = (m0_ov["dy"], m0_ov["dx"])

PROFILES = [  # placement, label, colour, marker, line style, filled
    ("ov_cellpose_dep", "Ovarian, Cellpose (Space Ranger 4)", SR4, "o", "-", True),
    ("crc_cellpose_dep", "Colorectal, Cellpose", SR4, "D", "-", True),
    ("postxen_he", "Post-Xenium lung, H&E nuclei", NUCLEUS, "^", "-", True),
    ("public_kidney", "Kidney", S.INK, "s", "-", True),
    ("public_mouse_brain", "Mouse brain", S.INK_MUTED, "v", "--", False),
]
BAND_LABELS = [">4", "2–4", "0–2", "0–2", "2–4", "4–6", "6–8", "8–10", "10–12", "far"]
EDGE_AT = 2.5                 # between I1 and O1
FAR_AT = 9.9                 # "far", set apart after a gap
BAND_X = list(range(9)) + [FAR_AT]

far = {(r["case"], int(r["radius"])): r for r in tsv(T / "far_shift_all_summaries.tsv")
       if r["offset"] == "pooled" and r["group"] == "all"}
real = {(r["case"], int(r["radius"])): r for r in tsv(T / "amendment6_all_summaries.tsv") if r["group"] == "all"}
lung_cores = list(load("lung_same_section.json")["criteria"])
CASES = [("C1", "Ovarian, deposited map"), ("C2", "Colorectal"), ("July", "Colorectal, H1-VM2JXXK/A1"),
         ("T1", "Ovarian, our masks")] + [(core, f"Lung, {core}") for core in lung_cores]
require({c for c, _ in CASES} == {c for c, _ in far}, "far-shift cases differ from the cases drawn")
for key, row in far.items():
    require(np.isclose(float(row["real_median"]), float(real[key]["median"])),
            f"{key}: far-shift real median differs from Amendment 6")


def pct(value):
    return 100 * float(value)


def pct_interval(text):
    lo, hi = json.loads(text)
    return 100 * lo, 100 * hi


# ------------------------------------------------------------------------------------------------
# layout
# ------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(S.TEXT_WIDTH, 7.0))
steps = fig.add_gridspec(1, 4, wspace=0.10, left=0.02, right=0.985, top=0.962, bottom=0.8157)
lower = fig.add_gridspec(2, 1, height_ratios=[1.16, 1.0], hspace=0.40, left=0.105, right=0.935, top=0.761,
                         bottom=0.103)
middle = lower[0].subgridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.36)
bottom = lower[1].subgridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.22)

# ------------------------------------------------------------------------------------------------
# a: the check in four steps
# ------------------------------------------------------------------------------------------------
titles = ["1  Nucleus masks", "2  Counts", "3  Move the masks", "4  Find the dip"]
subs = ["placed on the grid by the\nregistration under test",
        "lower under the nuclei\nthan in a ring around them",
        "mean count under the\nmasks at each shift",
        "lower than the shifts beside\nit; its shift is the offset"]
step_axes = []
for i in range(4):
    ax = fig.add_subplot(steps[0, i])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.55, 7.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.1, 7.5, titles[i], fontsize=S.FS_LABEL, fontweight="bold", va="top", color=S.INK)
    ax.text(0.1, 6.62, subs[i], fontsize=S.FS_NOTE, va="top", color=S.INK, linespacing=1.2)
    step_axes.append(ax)
fig.canvas.draw()
for i in range(3):
    left, right = step_axes[i].get_position(), step_axes[i + 1].get_position()
    y = left.y0 + 0.30 * left.height
    fig.patches.append(FancyArrowPatch((left.x1 - 0.012, y), (right.x0 + 0.002, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7, lw=0.8, color=S.INK_MUTED))

NCOL, NROW, SQ = 8, 7, 0.68
GX, GY = 4.5, 0.1
IMAGE_NUCLEI = [(1.6, 4.3), (4.9, 2.0)]          # (column, row) of each nucleus in the image, grid units
TRUE_SHIFT = (1.0, -1.0)                          # where the nuclei lie in the counts: one location off
RX, RY = 1.5, 1.1                                 # nucleus half-widths, in locations
RING_OUT = 3.4                                    # outer edge of the ring (squared, in nucleus half-widths): about one location wide


def at(col, row):
    return GX + (col + 0.5) * SQ, GY + (row + 0.5) * SQ


def count_level(col, row):
    """Grey level of one location: denser tissue to the left and a hollow under each nucleus, so the hollow on
    the left holds more counts than the tissue on the right."""
    base = 0.95 - 0.085 * col
    d = min(((col - c - TRUE_SHIFT[0]) / RX) ** 2 + ((row - r - TRUE_SHIFT[1]) / RY) ** 2 for c, r in IMAGE_NUCLEI)
    if d < 1:
        return base - 0.30
    return min(1.0, base + 0.30) if d < RING_OUT else base     # a ring of higher counts around each nucleus


def counts(ax):
    for col in range(NCOL):
        for row in range(NROW):
            x, y = at(col, row)
            ax.add_patch(Rectangle((x - SQ / 2, y - SQ / 2), SQ, SQ, fc=COUNT, alpha=count_level(col, row),
                                   ec="white", lw=0.3))


def masks(ax, shift=(0.0, 0.0), **style):
    for c, r in IMAGE_NUCLEI:
        x, y = at(c + shift[0], r + shift[1])
        ax.add_patch(Ellipse((x, y), 2 * RX * SQ, 2 * RY * SQ, fc="none", **style))


def key_square(ax, y, alpha, label, edge="none"):
    ax.add_patch(Rectangle((0.15, y - 0.27), 0.54, 0.54, fc=COUNT if alpha else "white", alpha=alpha or 1,
                           ec=edge, lw=0.35))
    ax.text(0.9, y, label, fontsize=S.FS_NOTE, color=S.INK, va="center")


# 1: masks on the grid
ax = step_axes[0]
for col in range(NCOL):
    for row in range(NROW):
        x, y = at(col, row)
        ax.add_patch(Rectangle((x - SQ / 2, y - SQ / 2), SQ, SQ, fc="white", ec=GRID_EDGE, lw=0.35))
masks(ax, ec=NUCLEUS, lw=1.1)
ax.add_patch(Ellipse((0.42, 4.55), 0.62, 0.46, fc="none", ec=NUCLEUS, lw=1.1))
ax.text(0.9, 4.55, "nucleus", fontsize=S.FS_NOTE, color=NUCLEUS, va="center")
key_square(ax, 3.65, 0, "capture\nlocation", edge=GRID_EDGE)

# 2: counts, a local hollow under each nucleus
ax = step_axes[1]
counts(ax)
ax.text(0.15, 4.55, "counts", fontsize=S.FS_NOTE, color=S.INK, va="center")
key_square(ax, 3.75, 0.86, "more")
key_square(ax, 2.95, 0.12, "fewer")

# 3: the masks moved over the grid
ax = step_axes[2]
counts(ax)
masks(ax, ec=NUCLEUS, lw=0.7, ls=(0, (2, 1.5)), alpha=0.75)
masks(ax, shift=TRUE_SHIFT, ec=NUCLEUS, lw=1.3)
c, r = IMAGE_NUCLEI[0]
ax.add_patch(FancyArrowPatch(at(c, r), at(c + TRUE_SHIFT[0], r + TRUE_SHIFT[1]), arrowstyle="-|>",
                             mutation_scale=6, lw=0.9, color=S.INK, shrinkA=0, shrinkB=0, zorder=6))
ax.plot([0.15, 0.75], [4.55, 4.55], color=NUCLEUS, lw=0.8, ls=(0, (2, 1.5)))
ax.text(0.9, 4.55, "as placed", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.plot([0.15, 0.75], [3.75, 3.75], color=NUCLEUS, lw=1.2)
ax.text(0.9, 3.75, "moved", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.add_patch(FancyArrowPatch((0.15, 2.95), (0.75, 2.95), arrowstyle="-|>", mutation_scale=6, lw=0.9, color=S.INK))
ax.text(0.9, 2.95, "shift", fontsize=S.FS_NOTE, color=S.INK, va="center")

# 4: mean count against shift, a cartoon (no data)
ax = step_axes[3]
X0, X1, Y0, Y1 = 1.0, 9.8, 0.75, 5.3
step = 0.52                                        # one location along the cartoon's axis
dip_x, zero_x = 4.6, 4.6 + 2 * step
xs = np.linspace(X0 + 0.1, X1 - 0.1, 300)
rel = (xs - dip_x) / step
ys = 4.35 - 2.5 * np.exp(-(rel / 1.7) ** 2) + 0.28 * np.exp(-((abs(rel) - 4.0) / 1.4) ** 2) - 0.012 * rel ** 2
ax.plot([X0, X0], [Y0, Y1], color=S.INK_MUTED, lw=0.6)
ax.plot([X0, X1], [Y0, Y0], color=S.INK_MUTED, lw=0.6)
ax.plot([zero_x, zero_x], [Y0, Y1], color=S.CHANCE, lw=0.5, ls="--")
ys_ring = 2.5 + 1.5 * np.exp(-(rel / 2.0) ** 2) - 0.010 * rel ** 2          # the mask of the ring: a peak at the same shift
ax.plot(xs, ys_ring, color=MOVED, lw=1.1, zorder=2)
ax.text(dip_x, float(ys_ring.max()) + 0.1, "ring", fontsize=S.FS_NOTE, color=S.INK_MUTED, ha="center", va="bottom")
ax.plot(xs, ys, color=S.INK, lw=1.1, zorder=3)
ax.plot([dip_x], [ys.min()], "o", ms=2.6, color=S.INK, zorder=4)
ax.text(zero_x, Y0 - 0.12, "0", fontsize=S.FS_NOTE, color=S.INK_MUTED, ha="center", va="top")
ax.text(dip_x - 0.3, ys.min(), "dip", fontsize=S.FS_NOTE, color=S.INK, ha="right", va="center")
# the dip against the shifts beside it: both neighbours lie higher
for side in (-1, 1):
    nx_ = dip_x + side * step
    ny_ = float(np.interp(nx_, xs, ys))
    ax.plot([nx_], [ny_], "o", ms=2.6, mfc="white", mec=S.INK, mew=0.7, zorder=4)
nb_y = float(np.interp(dip_x + step, xs, ys))
ax.plot([dip_x - 1.5 * step, dip_x + 1.5 * step], [nb_y, nb_y], color=S.INK_MUTED, lw=0.5, ls=(0, (2, 1.5)), zorder=2)
ax.add_patch(FancyArrowPatch((dip_x, Y0 + 0.3), (zero_x, Y0 + 0.3), arrowstyle="<|-|>", mutation_scale=4.5,
                             lw=0.6, color=S.INK, shrinkA=0, shrinkB=0))
ax.text(zero_x + 0.15, Y0 + 0.3, "offset", fontsize=S.FS_NOTE, color=S.INK, ha="left", va="center")
ax.text(X0 - 0.2, (Y0 + Y1) / 2, "mean count", fontsize=S.FS_NOTE, color=S.INK, rotation=90,
        ha="right", va="center")
ax.text(X1, Y0 - 0.12, "shift", fontsize=S.FS_NOTE, color=S.INK, ha="right", va="top")

row1_y = step_axes[0].get_position().y1 + 0.008
fig.text(0.0, row1_y, "A", fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
fig.text(0.024, row1_y, "The dip in four steps", fontsize=S.FS_TITLE, va="baseline", color=S.INK)

# ------------------------------------------------------------------------------------------------
# b: the shift surfaces, same masks under two registrations
# ------------------------------------------------------------------------------------------------
maps = middle[0, 0].subgridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.14)
norm = Normalize(min(np.nanmin(sr_crop), np.nanmin(ours)), max(np.nanmax(sr_crop), np.nanmax(ours)))
extent = (-UM * (HALF + 0.5), UM * (HALF + 0.5), UM * (HALF + 0.5), -UM * (HALF + 0.5))
map_axes = []
for j, (surface, low_at, colour, title) in enumerate(((sr_crop, sr_min, SR4, "Space Ranger's registration"),
                                                      (ours, ours_min, OURS, "our image-only stage"))):
    ax = fig.add_subplot(maps[0, j])
    ax.imshow(surface, cmap=COUNT_MAP, norm=norm, extent=extent, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=S.FS_TICK, loc="center", pad=3, color=S.INK)
    cy, cx = UM * low_at[0], UM * low_at[1]
    ax.plot([cx], [cy], marker="x", ms=4.5, mew=1.1, color=colour)
    ax.axhline(0, color="white", lw=0.4, alpha=0.8)
    ax.axvline(0, color="white", lw=0.4, alpha=0.8)
    ax.set_xticks([-16, -8, 0, 8, 16])
    ax.set_yticks([-16, -8, 0, 8, 16])
    ax.tick_params(length=2)
    ax.set_xlabel("Column shift (µm)")
    if j == 0:
        ax.set_ylabel("Row shift (µm)")
    else:
        ax.set_yticklabels([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.4)
    map_axes.append(ax)
cax = fig.add_subplot(maps[0, 2])
fig.canvas.draw()
first, cpos = map_axes[0].get_position(), cax.get_position()
cax.set_position([cpos.x0, first.y0, cpos.width, first.height])
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=COUNT_MAP), cax=cax)
bar.ax.tick_params(labelsize=S.FS_NOTE, length=2)
bar.outline.set_linewidth(0.4)
bar.set_label("Mean count under the masks", fontsize=S.FS_NOTE, labelpad=3)

# ------------------------------------------------------------------------------------------------
# c: the diagonal
# ------------------------------------------------------------------------------------------------
ax_c = fig.add_subplot(middle[0, 1])
diag_ours = [(k, float(ours[k + HALF, k + HALF])) for k in range(-HALF, HALF + 1)]
diag_sr = [(k, float(surface_sr[centre + k, centre + k])) for k in range(-HALF, HALF + 1)]
for series, colour, label in ((diag_sr, SR4, "Space Ranger's registration"), (diag_ours, OURS, "our image-only stage")):
    ax_c.plot([UM * k for k, _ in series], [v for _, v in series], color=colour, lw=1.2, marker="o", ms=2.4,
              zorder=3, label=label)
ax_c.axvline(0, color=S.CHANCE, lw=0.5, ls="--", zorder=1)
ax_c.set_xlim(-17, 17)
ax_c.set_xticks([-16, -8, 0, 8, 16])
C_LIM = (25.0, 33.0)
require(all(C_LIM[0] <= v <= C_LIM[1] for _, v in diag_ours + diag_sr), "panel c: data outside the fixed y range")
ax_c.set_ylim(*C_LIM)
ax_c.set_xlabel("Shift of the masks, each axis (µm)")
ax_c.set_ylabel("Mean count under the masks")
ax_c.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
            borderaxespad=0.1, fontsize=S.FS_NOTE, handlelength=1.4)
S.hide_y_grid_behind(ax_c)

# ------------------------------------------------------------------------------------------------
# d: counts across the nucleus edge
# ------------------------------------------------------------------------------------------------
ax_d = fig.add_subplot(bottom[0, 0])
D_LIM = (0.85, 1.36)
profile_values = {}
for name, label, colour, marker, style, filled in PROFILES:
    found, usable, shift, row = CR.ratio(CR.placement(name)[0]["whole"])
    require(usable, f"{name}: whole-slide correlation ratio not usable")
    ys = [row["profile"][b] for b in CR.BANDS]
    require(all(D_LIM[0] <= y <= D_LIM[1] for y in ys), f"panel d, {name}: data outside the fixed y range")
    profile_values[name] = (shift, ys)
    ax_d.plot(BAND_X[:9], ys[:9], color=colour, lw=1.0, ls=style, marker=marker, ms=3.0, zorder=3,
              mfc=colour if filled else "white", mec=colour, mew=0.8, label=label)
    ax_d.plot(BAND_X[9:], ys[9:], color=colour, marker=marker, ms=3.0, ls="none", zorder=3,
              mfc=colour if filled else "white", mec=colour, mew=0.8)
ax_d.axhline(1.0, color=S.INK_MUTED, lw=0.5, zorder=1)
ax_d.axvline(EDGE_AT, color=S.CHANCE, lw=0.6, ls="--", zorder=1)
ax_d.axvspan(-0.6, EDGE_AT, color=S.GRID, alpha=0.5, lw=0, zorder=0)
ax_d.axvline((8 + FAR_AT) / 2, color=S.INK_MUTED, lw=0.5, ls=":", zorder=1)
ax_d.set_xlim(-0.6, FAR_AT + 0.6)
ax_d.set_xticks(BAND_X)
ax_d.set_xticklabels(BAND_LABELS)
ax_d.tick_params(axis="x", length=2, labelsize=S.FS_NOTE)
ax_d.set_ylim(*D_LIM)
ax_d.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
ax_d.text(EDGE_AT - 0.15, D_LIM[0] + 0.01, "inside", fontsize=S.FS_NOTE, color=S.INK_MUTED, ha="right", va="bottom")
ax_d.text(EDGE_AT + 0.15, D_LIM[0] + 0.01, "outside", fontsize=S.FS_NOTE, color=S.INK_MUTED, ha="left", va="bottom")
ax_d.set_xlabel("Distance from the nucleus edge (µm)\ninside ← nucleus edge → outside")
ax_d.set_ylabel("Count relative to the local mean")
ax_d.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, borderaxespad=0.1,
            fontsize=S.FS_NOTE, handlelength=2.0, labelspacing=0.3)
S.hide_y_grid_behind(ax_d)

# ------------------------------------------------------------------------------------------------
# e: the correlation ratio surfaces
# ------------------------------------------------------------------------------------------------
GREY_UP = COUNT_MAP.reversed()                  # lighter = higher eta^2, so the shift found is light, as the dip in b
heat = bottom[0, 1].subgridspec(1, 5, width_ratios=[1, 0.06, 0.55, 1, 0.06], wspace=0.12)
eta_axes = []
eta_cax = []
eta_marks = {}
for j, (name, title, reference) in enumerate((("postxen_he", "lung, H&E nuclei", "truth"),
                                              ("public_kidney", "kidney", "zero"))):
    g = CR.placement(name)[0]["whole"]
    found, usable, shift, _ = CR.ratio(g)
    require(usable, f"{name}: whole-slide correlation ratio not usable")
    if reference == "truth":
        require(shift == tuple(g["truth"]), f"{name}: shift found {shift} differs from the truth {g['truth']}")
    else:
        require(max(abs(shift[0]), abs(shift[1])) <= 1, f"{name}: shift found {shift} is more than one location from zero")
    eta = CR.surface(name, "whole")
    c0 = eta.shape[0] // 2
    crop = eta[c0 - HALF:c0 + HALF + 1, c0 - HALF:c0 + HALF + 1]
    require(not np.isnan(crop).any() and abs(shift[0]) <= HALF and abs(shift[1]) <= HALF, f"{name}: surface crop incomplete")
    ax = fig.add_subplot(heat[0, 3 * j])
    cax_e = fig.add_subplot(heat[0, 3 * j + 1])
    eta_cax.append(cax_e)
    enorm = Normalize(float(crop.min()), float(crop.max()))
    ax.imshow(crop, cmap=GREY_UP, norm=enorm, extent=extent, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=S.FS_TICK, loc="center", pad=3, color=S.INK)
    ax.plot([UM * shift[1]], [UM * shift[0]], marker="x", ms=4.5, mew=1.1, color=OURS)
    if reference == "truth":
        ty, tx = g["truth"]
        ax.plot([UM * tx], [UM * ty], marker="s", ms=6.5, mfc="none", mec=NUCLEUS, mew=0.9)
    else:
        ax.axhline(0, color="white", lw=0.4, alpha=0.8)
        ax.axvline(0, color="white", lw=0.4, alpha=0.8)
    ax.set_xticks([-16, 0, 16])
    ax.set_yticks([-16, 0, 16])
    ax.tick_params(length=2)
    ax.set_xlabel("Column shift (µm)")
    if j == 0:
        ax.set_ylabel("Row shift (µm)")
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.4)
    bar_e = fig.colorbar(plt.cm.ScalarMappable(norm=enorm, cmap=GREY_UP), cax=cax_e)
    bar_e.formatter = FuncFormatter(lambda v, _: f"{v * 1e3:g}")        # eta^2 in units of 10^-3
    bar_e.update_ticks()
    bar_e.ax.tick_params(labelsize=S.FS_NOTE, length=2)
    bar_e.outline.set_linewidth(0.4)
    bar_e.set_label("η² (×10⁻³)", fontsize=S.FS_NOTE, labelpad=2)
    eta_axes.append(ax)
    eta_marks[name] = shift

fig.canvas.draw()
for ax, cax_e in zip(eta_axes, eta_cax):          # colour bars as tall as the (square) maps
    box, cpos = ax.get_position(), cax_e.get_position()
    cax_e.set_position([cpos.x0, box.y0, cpos.width, box.height])

# ------------------------------------------------------------------------------------------------
# panel heads
# ------------------------------------------------------------------------------------------------
fig.canvas.draw()
inv = fig.transFigure.inverted()


def left_edge(ax):
    """Left edge of an axes including its tick labels and y label, in figure coordinates."""
    box = ax.get_tightbbox(fig.canvas.get_renderer())
    return inv.transform((box.x0, box.y0))[0]


row2_y = max(a.get_position().y1 for a in map_axes + [ax_c])
title_box = map_axes[0].title.get_window_extent(fig.canvas.get_renderer())
row2_y = max(row2_y, inv.transform((0, title_box.y1))[1])
for ax, letter, title in ((map_axes[0], "b", "Same masks, two registrations"), (ax_c, "c", "Along the diagonal")):
    x = left_edge(ax)
    fig.text(x, row2_y + 0.014, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
    fig.text(x + 0.024, row2_y + 0.014, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)
row3_y = ax_d.get_position().y1
for ax, letter, title in ((ax_d, "d", "Counts across the nucleus edge"), (eta_axes[0], "e", "The correlation ratio")):
    x = left_edge(ax)
    fig.text(x, row3_y + 0.012, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
    fig.text(x + 0.024, row3_y + 0.012, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)

S.save(fig, OUT, "fig_dipcheck")

# ------------------------------------------------------------------------------------------------
# Supplemental Fig. S2 (was Fig. 4e): nucleus by nucleus, against the far-shift background
# ------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(S.TEXT_WIDTH, 2.9))
pair = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10, left=0.27, right=0.97, top=0.82, bottom=0.2)
ax_e = fig.add_subplot(pair[0, 0])
ax_f = fig.add_subplot(pair[0, 1], sharey=ax_e)
ys_rows = np.arange(len(CASES))
for i, (case, _) in enumerate(CASES):
    inside = real[(case, 1)]
    moved = far[(case, 1)]
    lo, hi = pct_interval(inside["interval"])
    ax_e.plot([lo, hi], [i - 0.13] * 2, color=NUCLEUS, lw=0.9, solid_capstyle="butt", zorder=2)
    ax_e.plot([pct(inside["median"])], [i - 0.13], "o", ms=3.6, color=NUCLEUS, zorder=3)
    lo, hi = pct_interval(moved["interval"])
    ax_e.plot([lo, hi], [i + 0.13] * 2, color=MOVED, lw=0.9, solid_capstyle="butt", zorder=2)
    ax_e.plot([pct(moved["median"])], [i + 0.13], "o", ms=3.6, mfc="white", mec=S.INK_MUTED, mew=0.8, zorder=3)
    for radius, offset, marker, face in ((1, -0.13, "o", S.INK), (2, 0.13, "D", "white")):
        row = far[(case, radius)]
        lo, hi = pct_interval(row["difference_interval"])
        ax_f.plot([lo, hi], [i + offset] * 2, color=S.INK, lw=0.9, solid_capstyle="butt", zorder=2)
        ax_f.plot([pct(row["real_minus_control"])], [i + offset], marker, ms=3.4 if marker == "o" else 3.0,
                  mfc=face, mec=S.INK, mew=0.8, zorder=3)
split = sum(1 for case, _ in CASES if case not in lung_cores) - 0.5
for ax in (ax_e, ax_f):
    ax.axhline(split, color=S.GRID, lw=0.6)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
ax_f.axvline(0, color=S.CHANCE, lw=0.6, zorder=1)
ax_e.set_yticks(ys_rows)
ax_e.set_yticklabels([label for _, label in CASES])
ax_e.set_ylim(len(CASES) - 0.5, -0.5)
plt.setp(ax_f.get_yticklabels(), visible=False)
E_LIM, F_LIM = (0, 6.4), (-5.6, 8.6)
for case, _ in CASES:
    values_e = [pct(real[(case, 1)]["median"]), pct(far[(case, 1)]["median"]),
                *pct_interval(real[(case, 1)]["interval"]), *pct_interval(far[(case, 1)]["interval"])]
    values_f = [v for radius in (1, 2) for v in (pct(far[(case, radius)]["real_minus_control"]),
                                                 *pct_interval(far[(case, radius)]["difference_interval"]))]
    require(all(E_LIM[0] <= v <= E_LIM[1] for v in values_e), f"Supplemental Fig. S2, {case}: data outside the fixed x range")
    require(all(F_LIM[0] <= v <= F_LIM[1] for v in values_f), f"Supplemental Fig. S2, {case}: difference outside the x range")
ax_e.set_xlim(*E_LIM)
ax_e.set_xticks([0, 2, 4, 6])
ax_f.set_xlim(*F_LIM)
ax_f.set_xticks([-4, 0, 4, 8])
ax_e.set_xlabel("Median paired depth\n(%)")
ax_f.set_xlabel("In place − moved\n(percentage points)")
ax_e.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, color=NUCLEUS),
                     Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=S.INK_MUTED, mew=0.8)],
            labels=["in place", "moved"], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False,
            borderaxespad=0.1, fontsize=S.FS_NOTE, handletextpad=0.2, columnspacing=0.8)
ax_f.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.4, color=S.INK),
                     Line2D([], [], ls="none", marker="D", ms=3.0, mfc="white", mec=S.INK, mew=0.8)],
            labels=["neighbouring shifts", "next shifts out"], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1,
            frameon=False,
            borderaxespad=0.1, fontsize=S.FS_NOTE, handletextpad=0.2, columnspacing=0.8)

S.save(fig, OUT, "fig_supp_pairdepth")

# For the caption writer: the case behind each row of Supplemental Fig. S2.
print("Supplemental Fig. S2 rows:", "; ".join(f"{label} = {case}" for case, label in CASES))
print("panel d profiles (shift; I3 I2 I1 O1..O6 far):", "; ".join(
    f"{n} {sh} " + " ".join(f"{y:.3f}" for y in ys) for n, (sh, ys) in profile_values.items()))
print("panel e shifts found:", eta_marks)
print(f"panel b minima: Space Ranger's registration {sr_min}, our registration {ours_min} (locations)")
