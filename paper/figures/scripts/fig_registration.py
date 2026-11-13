#!/usr/bin/env python3
"""Fig. 5: what registration with the correlation ratio found, and Supplemental Fig. S3.

a, whole-slide offset found by the correlation ratio under each registration tested (Fig. 4d of the preprint draft).
b, offsets in the regions of the SPATCH slide: our registration before and after the fitted
   correction, and the registrations of Space Ranger 3.0.0 and 4.1.0 (Fig. 4c plus the 3.0.0 map).
Supplemental Fig. S3 (fig_supp_lung, was panel c): the lung same-section test, four cores and the whole post-Xenium sections, H&E nuclei
   segmented with Cellpose-SAM: each starts where the dataset's registration places it (minus the truth) and is moved by the shift found from
   the counts; filled where it is usable and lands on the Xenium nuclei, small open one location away, large open where no shift meets the rule.
Supplemental Fig. S4 (fig_supp_regsens), drawn here, was panel d: what swapping one step of the
   registration moves on the colorectal slide.
How the check works is Fig. 4 (fig_dipcheck.py).

Inputs: paper_results/registration_correlation_ratio_20261108/ (the correlation ratio on every placement, read by
registration_ratio.py), registration_local_dip_20261106/ (the dip, checked against the ratio for panel a only),
registration_check_20261018/ (lung mask displacements) and the colorectal registration-sensitivity
record of 2026-09-15. Usable regions and shifts are those of the rule: highest local maximum of the correlation ratio
within 12 locations, interior, within one location of the full-data shift in at least 95% of 200 block draws. Displacements are (dy, dx) in capture locations of 2 um, mask to counts.
Usage: fig_registration.py [OUT_DIR]
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_macros  # noqa: E402
import registration_ratio as CR  # noqa: E402
import spatial_paper_figstyle as S  # noqa: E402

ROOT = HERE.parents[2]
R = ROOT / "paper_results" / "registration_check_20261018"
NEWR = ROOT / "paper_results" / "registration_local_dip_20261106" / "placements"
SENSITIVITY = ROOT / "paper_results" / "spatial_v1_9_5_20260915" / "crc_frozen_registration_sensitivity"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
UM = 2.0
S.apply()
OURS = S.STAR
SR4 = S.SR
SR3 = S.SR_SOFT
NUCLEUS = "#6d4c9f"
RAMP = LinearSegmentedColormap.from_list("offset", ["#eef8f6", "#9fd6cb", "#2a9d8f", "#134e48"])
UNRESOLVED = "#e5e7eb"


def require(condition, message):
    if not condition:
        sys.exit(f"fig_registration.py: {message}")


def load(name, base=R):
    return json.loads((base / name).read_text())


def dist(dy, dx):
    return UM * math.hypot(dy, dx)


def rows_of(name):
    base = CR.placements_dir(name)
    meta = json.loads((base / f"{name}.meta.json").read_text())
    require(meta["all_reproduced"] and meta["search"] == 12 and meta["draws"] == 200, f"{name}: not a reproduced run of the final rule")
    return json.loads((base / f"{name}.rows.json").read_text())


def whole_row(name):
    w = [r for r in rows_of(name) if r["surface"] == "whole"][0]
    require(w["dip"] and w["usable"], f"{name}: whole-slide dip not usable")
    return w


def m0_dist(name):
    """Offset found by the correlation ratio on the whole slide (the dip's whole-slide shift is the same)."""
    found, usable, shift, _ = CR.ratio(CR.placement(name)[0]["whole"])
    require(usable, f"{name}: whole-slide correlation ratio not usable")
    w = whole_row(name)
    require(shift == (w["dy"], w["dx"]), f"{name}: correlation-ratio shift {shift} differs from the dip's {(w['dy'], w['dx'])}")
    return dist(*shift)


def regional(name):
    rows, _ = CR.placement(name)
    out = []
    for key, g in rows.items():
        if key == "whole":
            continue
        found, usable, shift, _ = CR.ratio(g) if g.get("tested") else (False, False, None, None)    # not tested: grey
        out.append({"row": int(key[1]), "col": int(key[3]), "usable": usable,
                    "dy": shift[0] if shift else None, "dx": shift[1] if shift else None})
    require(len(out) == 25, f"{name}: not a 5 x 5 division")
    return out


fig = plt.figure(figsize=(S.TEXT_WIDTH, 2.55))
top = fig.add_gridspec(1, 2, width_ratios=[0.80, 2.45], wspace=0.10, left=0.125, right=0.93, top=0.83, bottom=0.255)

# --- a: whole-slide offset by registration -------------------------------------------------------
ax_a = fig.add_subplot(top[0, 0])
rows = [
    ("Colorectal", [("ours", m0_dist("crc_cellpose_img")), ("sr4", m0_dist("crc_cellpose_dep"))]),
    ("Ovarian", [("ours", m0_dist("ov_cellpose_img")), ("sr4", m0_dist("ov_cellpose_dep"))]),
    ("SPATCH", [("ours", m0_dist("spatch_before")), ("sr3", m0_dist("spatch_sr3")), ("sr4", m0_dist("spatch_sr410"))]),
    ("Colorectal\nH1-VM2JXXK/A1", [("sr3", m0_dist("a1_cellpose_sr3")), ("sr4", m0_dist("a1_cellpose_sr410"))]),
]
style = {"ours": (OURS, "o", "our image-only stage"), "sr3": (SR3, "s", "Space Ranger 3.0.0"),
         "sr4": (SR4, "D", "Space Ranger 4")}
offset = {"ours": -0.2, "sr3": 0.0, "sr4": 0.2}
largest = max(v for _, points in rows for _, v in points)
for i, (name, points) in enumerate(rows):
    for key, value in points:
        colour, marker, _ = style[key]
        ax_a.scatter([value], [i + offset[key]], s=18, marker=marker, color=colour, edgecolor="white", lw=0.5, zorder=3)
ax_a.set_yticks(range(len(rows)))
ax_a.set_yticklabels([r[0] for r in rows])
ax_a.set_ylim(len(rows) - 0.5, -2.1)
ax_a.tick_params(axis="y", length=0)
ax_a.set_xlim(-0.8, 4 * math.ceil(largest / 4) + 1.0)
ax_a.set_xticks(range(0, 4 * math.ceil(largest / 4) + 1, 4))
ax_a.set_xlabel("Offset of the nuclei from\nSTARgrid's fit to the counts (µm)")
ax_a.xaxis.grid(True)
ax_a.set_axisbelow(True)
ax_a.legend(handles=[Line2D([], [], ls="none", marker=m, ms=4.0, color=c) for c, m, _ in style.values()],
            labels=[lab for _, _, lab in style.values()], loc="upper right", ncol=1, frameon=False,
            borderaxespad=0.1, fontsize=S.FS_NOTE, handletextpad=0.3, labelspacing=0.25)

# --- b: regional offsets on the SPATCH slide ----------------------------------------------------
# the registrations made from the images alone, then Space Ranger 4.1.0, which also uses the counts; the offsets are measured from
# STARgrid's fit to the counts, so our second stage (fitted to these offsets) is not drawn
maps = top[0, 1].subgridspec(1, 5, width_ratios=[1, 1, 0.22, 1, 0.07], wspace=0.10)
sets = [("Ours,\nfirst stage", regional("spatch_before")),
        ("Space Ranger\n3.0.0", regional("spatch_sr3")),
        ("Space Ranger\n4.1.0", regional("spatch_sr410"))]
GROUPS = (("Images only", (0, 1)), ("Images and counts", (2, 2)))
SLOTS = (0, 1, 3)
side = max(max(r["row"], r["col"]) for _, regions in sets for r in regions) + 1
require(all(len(regions) == side * side for _, regions in sets), "regional maps are not complete square divisions")
top_um = max(dist(r["dy"], r["dx"]) for _, regions in sets for r in regions if r.get("usable"))
vmax = 4 * math.ceil(top_um / 4)
norm = Normalize(0, vmax)
ARROW = 0.62                            # arrow length in regions: direction only, the colour gives the size
map_axes = []
for j, (title, regions) in enumerate(sets):
    ax = fig.add_subplot(maps[0, SLOTS[j]])
    ax.set_title(title, pad=3, fontsize=S.FS_TICK, loc="center")
    for r in regions:
        x, y = r["col"], r["row"]
        if not r.get("usable"):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=UNRESOLVED, edgecolor="white", lw=0.9))
            continue
        d = dist(r["dy"], r["dx"])
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=RAMP(norm(d)), edgecolor="white", lw=0.9))
        if d == 0:
            ax.plot(x + 0.5, y + 0.5, "o", ms=1.5, color=S.INK)
        else:
            ink = "white" if d > 0.55 * vmax else S.INK
            hx, hy = ARROW * UM * r["dx"] / (2 * d), ARROW * UM * r["dy"] / (2 * d)
            ax.add_patch(FancyArrowPatch((x + 0.5 - hx, y + 0.5 - hy), (x + 0.5 + hx, y + 0.5 + hy),
                                         arrowstyle="-|>", mutation_scale=4.0, lw=0.65, color=ink,
                                         shrinkA=0, shrinkB=0))
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    map_axes.append(ax)
cax = fig.add_subplot(maps[0, 4])
fig.canvas.draw()
first = map_axes[0].get_position()
_inv = fig.transFigure.inverted()
_title_top = max(_inv.transform((0, a.title.get_window_extent(fig.canvas.get_renderer()).y1))[1] for a in map_axes)
group_heads = []
for label, (i, k) in GROUPS:                     # a header and a rule over each pair of maps
    x0, x1 = map_axes[i].get_position().x0, map_axes[k].get_position().x1
    fig.add_artist(Line2D([x0, x1], [_title_top + 0.012] * 2, transform=fig.transFigure, color=S.INK_MUTED, lw=0.6))
    group_heads.append(fig.text((x0 + x1) / 2, _title_top + 0.020, label, ha="center", va="bottom", fontsize=S.FS_TICK,
                                color=S.INK))
cpos = cax.get_position()
cax.set_position([cpos.x0, first.y0, cpos.width, first.height])
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax)
bar.set_ticks(range(0, vmax + 1, 4))
bar.ax.tick_params(labelsize=S.FS_NOTE, length=2)
bar.outline.set_linewidth(0.4)
bar.set_label("Offset (µm)", fontsize=S.FS_NOTE, labelpad=2)
fig.text(first.x0, first.y0 - 0.025, "Offset: how far the nuclei must move from where the registration places them to\n"
         "where STARgrid fits them to the counts (correlation ratio). Colour, its size; arrow, its direction;\n"
         "dot, none; grey, no shift meeting our criteria. Rows run down the slide.", fontsize=S.FS_NOTE, color=S.INK_MUTED, va="top",
         linespacing=1.25)

# --- panel heads ---------------------------------------------------------------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()


def left_edge(ax):
    return inv.transform((ax.get_tightbbox(renderer).x0, 0))[0]


def top_edge(*artists):
    return max(inv.transform((0, a.get_window_extent(renderer).y1))[1] for a in artists)


def head(x, y, letter, title):
    fig.text(x, y, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
    fig.text(x + 0.024, y, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)


y_top = max(top_edge(*[a.title for a in map_axes], *group_heads), ax_a.get_position().y1) + 0.016
head(left_edge(ax_a), y_top, "a", "Whole-slide offset")
head(first.x0 - 0.01, y_top, "b", "Offsets across the SPATCH slide")

S.save(fig, OUT, "fig_registration")

# --- Supplemental Fig. S3 (was Fig. 5c): the lung, core by core and section by section, against the nuclei imaged by Xenium
fig = plt.figure(figsize=(0.82 * S.TEXT_WIDTH, 2.55))
low = fig.add_gridspec(1, 2, width_ratios=[0.62, 1.0], wspace=0.04, left=0.12, right=0.99, top=0.97, bottom=0.16)
import registration_ring as RR  # noqa: E402

ax_c = fig.add_subplot(low[0, 0])
lung = load("lung_same_section.json")["criteria"]


def lung_ratio(name):
    """(shift, truth, marker style) of the correlation ratio on a whole lung core or section (the Cellpose-SAM H&E nuclei): a filled circle where
    usable and at the truth, a small open one where usable and not at it, a large open one where not usable (the 2026-11-10 lung record reports
    the cases that are not usable or not at the truth)."""
    g = CR.placement(name)[0]["whole"]
    found, usable, shift, _ = CR.ratio(g)
    require(found, f"{name}: no correlation-ratio maximum")
    truth = tuple(g["truth"])
    return shift, truth, DOT if usable and truth == shift else OFF if usable else OPEN


DOT = dict(s=24, facecolor=SR3, edgecolor=SR4, lw=0.6, zorder=4)            # usable, on the Xenium nuclei
OFF = dict(s=24, facecolor="white", edgecolor=SR4, lw=0.9, zorder=4)        # usable, one location from them
OPEN = dict(s=80, facecolor="white", edgecolor=SR4, lw=0.9, zorder=4)       # not usable
cases = [(core, f"lung_cpsam_{core}") for core in lung] + [(sec["id"], Path(sec["he_rows"]).name[: -len(".rows.json")])
                                                            for sec in RR.sections(cpsam=True)]
results = [(label, *lung_ratio(name)) for label, name in cases]       # Cellpose-SAM H&E nuclei: truth is their own mask displacement

# One frame centred on the Xenium nuclei. Each H&E nucleus map starts where the dataset's registration puts it (minus the truth,
# the move that brings it onto the Xenium nuclei); the arrow is the shift found from the counts, so it ends at found - truth.
starts = {}
for label, (dy, dx), (ty, tx), style in results:
    sx, sy = -UM * tx, -UM * ty
    ex, ey = UM * (dx - tx), UM * (dy - ty)
    ax_c.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=S.INK_MUTED,
                                   shrinkA=2, shrinkB=3.5, zorder=2))
    ax_c.scatter([ex], [ey], **style)
    starts.setdefault((sx, sy), []).append(label)
for (sx, sy), labels in starts.items():
    ax_c.scatter([sx], [sy], s=8, color=S.INK_MUTED, zorder=3)
    if len(labels) == 1:
        left = sx < 3                              # the start nearest the vertical axis is labelled on its left
        ax_c.text(sx + (-0.8 if left else 0.8), sy, labels[0], fontsize=S.FS_NOTE, color=S.INK,
                  ha="right" if left else "left", va="center")
    else:                                          # a shared start: the names stacked below and to the right of it
        ax_c.text(sx + 0.8, sy + 0.5, "\n".join(labels), fontsize=S.FS_NOTE, color=S.INK, ha="left", va="top",
                  linespacing=1.1)
ax_c.scatter([0], [0], s=26, marker="s", facecolor="none", edgecolor=S.INK, lw=0.9, zorder=5)
ax_c.set_xlim(-5.0, 14.5)
ax_c.set_ylim(12.5, -4.0)                      # rows run down the slide
ax_c.set_anchor("W")
ax_c.set_aspect("equal", adjustable="box")
ax_c.set_xticks([-4, 0, 4, 8, 12])
ax_c.set_yticks([-4, 0, 4, 8, 12])
ax_c.set_xlabel("Column, from the Xenium nuclei (µm)")
ax_c.set_ylabel("Row, from the\nXenium nuclei (µm)")
ax_c.grid(True, color=S.GRID, lw=0.5)
ax_c.set_axisbelow(True)

ax_l = fig.add_subplot(low[0, 1])
ax_l.axis("off")
ax_l.legend(handles=[Line2D([], [], ls="none", marker="s", ms=4.6, mfc="none", mec=S.INK, mew=0.9),
                     Line2D([], [], ls="none", marker="o", ms=2.6, color=S.INK_MUTED),
                     Line2D([], [], color=S.INK_MUTED, lw=0.8, marker=">", ms=3),
                     Line2D([], [], ls="none", marker="o", ms=4.2, mfc=SR3, mec=SR4, mew=0.6),
                     Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=SR4, mew=0.9),
                     Line2D([], [], ls="none", marker="o", ms=8.9, mfc="white", mec=SR4, mew=0.9)],
            labels=["nuclei imaged by Xenium",
                    "H&E nuclei (Cellpose-SAM), where the\ndataset's registration places them",
                    "moved by the shift found from the counts\n(correlation ratio)",
                    "lands on the Xenium nuclei",
                    "lands one capture location away",
                    "no shift meeting our\nstability criterion"],
            loc="center left", frameon=False, borderaxespad=0.0, handlelength=1.4,
            fontsize=S.FS_NOTE, labelspacing=0.9)

S.save(fig, OUT, "fig_supp_lung")


# --- Supplemental Fig. S4 (was Fig. 5d): what a small change of registration moves ----------------------------------------------
fig = plt.figure(figsize=(0.5 * S.TEXT_WIDTH, 2.3))
ax_d = fig.add_axes([0.24, 0.16, 0.72, 0.76])
moved = json.loads((SENSITIVITY / "summary.json").read_text())["mask_reprojection"]
require(math.isclose(float(paper_macros.load()["crcRegMedianShiftUm"]), moved["median_displacement_um"], abs_tol=0.005),
        "median shift differs from the macro the caption uses")
parts = [("Nucleus mask", 100 * moved["nucleus_mask_xor_fraction"]), ("Cell mask", 100 * moved["cell_mask_xor_fraction"])]
for i, (name, v) in enumerate(parts):
    ax_d.bar(i, v, width=0.6, color=NUCLEUS if name == "Nucleus mask" else S.INK_MUTED)
    ax_d.text(i, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=S.FS_NOTE, color=S.INK)
ax_d.set_xticks([0, 1])
ax_d.set_xticklabels([p[0] for p in parts])
ax_d.set_xlim(-0.6, 1.6)
ax_d.set_ylim(0, 1.2 * max(v for _, v in parts))
ax_d.yaxis.set_major_locator(MultipleLocator(5))
ax_d.set_ylabel("Capture locations\nchanging mask (%)")
S.hide_y_grid_behind(ax_d)

S.save(fig, OUT, "fig_supp_regsens")
