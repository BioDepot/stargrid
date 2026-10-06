#!/usr/bin/env python3
"""Fig. 1: the processing path, the spatial barcode, molecule resolution and
the decoder check. Schematic; the only numbers come from the macro files.
Usage: fig1_system.py [OUT_DIR]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_macros  # noqa: E402
import spatial_paper_figstyle as S  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
M = paper_macros.load()
S.apply()
PALE = "#eef3ff"      # light tint of the STAR blue for process boxes
EDGE = "#9ca3af"
MONO = "DejaVu Sans Mono"


def box(ax, x, y, w, h, title, body="", face="white", edge=EDGE, title_colour=S.INK, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, lw=lw))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.3,
            fontweight="bold", color=title_colour)
    if body:
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=5.6,
                color=S.INK, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, colour=S.INK_MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.7, color=colour, shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(S.DOUBLE_COL, 118 * S.MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.28)
bottom = outer[1].subgridspec(1, 3, width_ratios=[1.1, 1.35, 0.85], wspace=0.30)

# --- a: processing path --------------------------------------------------------
ax = fig.add_subplot(outer[0])
ax.set_xlim(0, 200)
ax.set_ylim(0, 60)
ax.axis("off")
S.panel_label(ax, "a", dx=-0.02, dy=1.06)

box(ax, 0, 14, 28, 34, "Paired FASTQ", "")
ax.text(14, 37, "Read 1\nUMI + spatial\nbarcode", ha="center", va="center", fontsize=5.6)
ax.text(14, 22, "Read 2\ntranscript\nor probe", ha="center", va="center", fontsize=5.6)
ax.plot([3, 25], [29.5, 29.5], color=EDGE, lw=0.5)

ax.add_patch(FancyBboxPatch((33, 2), 132, 56, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor="none", edgecolor=S.STAR, lw=0.9, ls=(0, (3, 2))))
ax.text(35, 55.2, "One STAR process", fontsize=6.3, color=S.STAR, fontweight="bold", va="top")

box(ax, 38, 31, 38, 19, "Spatial decoding",
    "every square at the\nminimum edit distance\n(≤ 2 edits per half)", face=PALE)
box(ax, 38, 8, 38, 21, "Feature assignment",
    "3′: genome alignment,\nintrons included\nProbe: half-probe match,\nno genome", face=PALE)
box(ax, 84, 18, 34, 21, "Read cliques",
    "same gene,\nsame raw UMI,\na shared candidate\nsquare", face=PALE)
box(ax, 125, 18, 36, 21, "Molecule resolution",
    "Phred likelihood +\noligo-frequency prior;\nUMI correction per\ncandidate square", face=PALE)
box(ax, 170, 14, 30, 31, "Count matrices",
    "strict\nhard (reported)\nsoft-expected\ngated-hard\n\nat 2, 8 and 16 µm", face="white")

arrow(ax, 28, 37, 38, 40.5)     # Read 1 -> spatial decoding
arrow(ax, 28, 22, 38, 18.5)     # Read 2 -> feature assignment
arrow(ax, 76, 40.5, 84, 32)
arrow(ax, 76, 18.5, 84, 25)
arrow(ax, 118, 28.5, 125, 28.5)
arrow(ax, 161, 28.5, 170, 28.5)
ax.text(140, 7.0, "No Space Ranger output\nenters any count", ha="center", fontsize=5.6,
        color=S.INK_MUTED, style="italic")

# --- b: barcode layout -----------------------------------------------------------
ax = fig.add_subplot(bottom[0, 0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
S.panel_label(ax, "b", dx=-0.10, dy=1.10)
ax.set_title("The spatial barcode", pad=4)
ax.text(0, 96, f"BC1 names the column, BC2 the row:\n{M['gridSide']} × {M['gridSide']} squares", fontsize=5.6,
        va="top", color=S.INK, linespacing=1.3)
g0x, g0y, cell = 34, 22, 11
bc1 = [15, 16, 15, 16]
bc2 = [14, 15, 14, 15]
for i, l1 in enumerate(bc1):
    ax.text(g0x + (i + 0.5) * cell, g0y + 4 * cell + 3, str(l1), ha="center", fontsize=5.6, color=S.STAR)
for j, l2 in enumerate(bc2):
    ax.text(g0x - 3, g0y + (3 - j + 0.5) * cell, str(l2), ha="right", va="center", fontsize=5.6, color=S.STAR)
    for i in range(4):
        ax.add_patch(Rectangle((g0x + i * cell, g0y + (3 - j) * cell), cell, cell,
                               facecolor="white", edgecolor=EDGE, lw=0.6))
ax.text(g0x + 2 * cell, g0y + 4 * cell + 9, "BC1 length (bases)", ha="center", fontsize=5.6, color=S.INK_MUTED)
ax.text(g0x - 22, g0y + 2 * cell, "BC2\nlength", ha="center", va="center", fontsize=5.6, color=S.INK_MUTED)
ax.text(0, 12, "Lengths alternate between neighbours, so a read\none base short or long in a half has the\nlength of its neighbour's barcode.",
        va="top", fontsize=5.4, color=S.INK_MUTED, linespacing=1.3)

# --- c: molecule resolution --------------------------------------------------------
ax = fig.add_subplot(bottom[0, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
S.panel_label(ax, "c", dx=-0.07, dy=1.10)
ax.set_title("Keeping every candidate until the molecule", pad=4)
reads = [("read 1", ["A", "B"]), ("read 2", ["A", "B"]), ("read 3", ["A"])]
for k, (name, cands) in enumerate(reads):
    y = 80 - k * 14
    ax.text(0, y, name, fontsize=5.6, va="center", color=S.INK)
    for m, c in enumerate(cands):
        ax.add_patch(Rectangle((16 + m * 9, y - 4), 7, 8, facecolor=PALE, edgecolor=S.STAR, lw=0.6))
        ax.text(16 + m * 9 + 3.5, y, c, ha="center", va="center", fontsize=5.6, color=S.STAR)
ax.text(0, 30, "one read clique:\nsame gene,\nsame raw UMI", fontsize=5.6, va="center", color=S.INK_MUTED, linespacing=1.3)
arrow(ax, 36, 66, 44, 66)
ax.text(46, 84, "Candidates A, B", fontsize=5.8, color=S.INK, fontweight="bold")
ax.text(46, 76, "posterior: A ≫ B", fontsize=5.6, color=S.INK)
policies = [("strict", "dropped: two candidates"), ("hard", "A"), ("soft-expected", "A and B, by posterior"),
            ("gated-hard", "A, if confident")]
for k, (p, what) in enumerate(policies):
    y = 64 - k * 13
    ax.text(46, y, p, fontsize=5.6, color=S.STAR if p == "hard" else S.INK,
            fontweight="bold" if p == "hard" else "normal", va="center")
    ax.text(46, y - 5, what, fontsize=5.3, color=S.INK_MUTED, va="center")

# --- d: decoder check -----------------------------------------------------------------
ax = fig.add_subplot(bottom[0, 2])
ax.axis("off")
S.panel_label(ax, "d", dx=-0.12, dy=1.10)
ax.set_title("Decoder check", pad=4)
ax.text(0.0, 0.84, f"{M['oracleErrors']} errors", transform=ax.transAxes, fontsize=11,
        fontweight="bold", color=S.STAR, va="top")
ax.text(0.0, 0.60, f"in {M['oracleDistanceChecks']} distances\n"
        f"over {M['oracleQueryPairs']} query pairs\n"
        f"and the full codebook of {M['oracleOligos']}\noligos, against an independent\n"
        "Wagner–Fischer calculation", transform=ax.transAxes, fontsize=5.6, va="top", color=S.INK,
        linespacing=1.3)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_system.{ext}")
print("wrote", OUT / "fig1_system.pdf")
