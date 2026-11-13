#!/usr/bin/env python3
"""Fig. 1: the four steps of STARgrid (decoding, gene assignment, count assignment, registration), the
processing path, the spatial barcode and molecule resolution. Schematic; the only numbers come from the macro files.
Usage: fig1_system.py [OUT_DIR]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

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
NUCLEUS = "#6d4c9f"   # nuclei, as in the registration figure
MOLECULE = "#d1495b"
GRID_EDGE = "#c3c7ce"


def box(ax, x, y, w, h, title, body="", face="white", edge=EDGE, title_colour=S.INK, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, lw=lw))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=S.FS_LABEL,
            fontweight="bold", color=title_colour)
    if body:
        ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=S.FS_NOTE,
                color=S.INK, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, colour=S.INK_MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.7, color=colour, shrinkA=0, shrinkB=0))


def small_grid(ax, x0, y0, n, m, s):
    for a in range(n):
        for b in range(m):
            ax.add_patch(Rectangle((x0 + a * s, y0 + b * s), s, s, fc="white", ec=GRID_EDGE, lw=0.35))


fig = plt.figure(figsize=(S.TEXT_WIDTH, 7.2))
outer = fig.add_gridspec(3, 1, height_ratios=[0.62, 1.42, 1], hspace=0.28, left=0.03, right=0.995, top=0.965, bottom=0.03)
steps = outer[0].subgridspec(1, 4, wspace=0.16)
bottom = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.30)

# --- a: the four steps of STARgrid ---------------------------------------------------
titles = ["1  Decoding", "2  Gene assignment", "3  Count assignment", "4  Registration"]
subs = ["barcode to best-matching locations", "Read 2 to a gene", "one location per molecule",
        "image to the capture grid"]
tags = [("STAR Suite process", S.STAR), ("STAR Suite process", S.STAR), ("STAR Suite process", S.STAR),
        ("images and counts", S.STAR)]
axes = []
for i in range(4):
    ax = fig.add_subplot(steps[0, i])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.1, 7.3, titles[i], fontsize=S.FS_LABEL, fontweight="bold", va="top", color=S.INK)
    ax.text(0.1, 6.35, subs[i], fontsize=S.FS_NOTE, va="top", color=S.INK)
    ax.text(0.1, -0.75, tags[i][0], fontsize=S.FS_NOTE, va="bottom", color=tags[i][1], style="italic")
    axes.append(ax)
for i in range(3):
    left, right = axes[i].get_position(), axes[i + 1].get_position()
    y = left.y0 + 0.42 * left.height
    fig.patches.append(FancyArrowPatch((left.x1 - 0.004, y), (right.x0 + 0.004, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7, lw=0.8, color=S.INK_MUTED))

# 1 decoding, top view: a read barcode matches one or more capture locations
ax = axes[0]
ax.add_patch(Rectangle((0.2, 4.3), 3.0, 0.45, fc="#dddddd", ec="#555555", lw=0.45))
ax.add_patch(Rectangle((0.2, 4.3), 1.3, 0.45, fc=S.STAR, ec="#555555", lw=0.45))
ax.text(0.2, 3.85, "read barcode", fontsize=S.FS_NOTE, color=S.STAR, va="top")
ax.add_patch(FancyArrowPatch((3.4, 4.5), (5.35, 3.2), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=S.INK_MUTED,
                             connectionstyle="arc3,rad=-0.3"))
small_grid(ax, 5.45, 0.2, 5, 6, 0.75)
ax.add_patch(Rectangle((5.45 + 1 * 0.75, 0.2 + 3 * 0.75), 0.75, 0.75, fc=S.STAR, ec="#555555", lw=0.5))
ax.add_patch(Rectangle((5.45 + 1 * 0.75, 0.2 + 2 * 0.75), 0.75, 0.75, fc=S.STAR, alpha=0.3, ec="#555555", lw=0.5))
ax.text(0.2, 1.6, "all best-matching\nlocations kept", fontsize=S.FS_NOTE, va="center",
        color=S.INK_MUTED, linespacing=1.25)

# 2 gene assignment: a read aligned to a gene (RNA-seq) and a probe pair matched by its halves (Flex)
ax = axes[1]
ax.text(0.2, 5.05, "RNA-seq: alignment", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.plot([0.6, 9.4], [4.05, 4.05], color=S.INK_MUTED, lw=0.7)                       # intron line
for x0e, w in ((0.6, 2.0), (4.2, 1.6), (7.4, 2.0)):
    ax.add_patch(Rectangle((x0e, 3.7), w, 0.7, fc="#e5e7eb", ec="#555555", lw=0.45))   # exons
ax.add_patch(Rectangle((4.5, 3.0), 2.6, 0.32, fc=S.STAR, ec="none"))              # a read, partly intronic
ax.text(9.4, 3.16, "gene", fontsize=S.FS_NOTE, color=S.INK_MUTED, ha="right", va="center")
ax.text(0.2, 2.05, "Flex: probe halves", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.add_patch(Rectangle((1.6, 0.9), 2.8, 0.42, fc=S.STAR, ec="#555555", lw=0.45))
ax.add_patch(Rectangle((4.4, 0.9), 2.8, 0.42, fc=S.STAR_SOFT, ec="#555555", lw=0.45))
ax.plot([0.6, 8.2], [0.55, 0.55], color=S.INK_MUTED, lw=0.7)
ax.text(8.4, 0.55, "RNA", fontsize=S.FS_NOTE, color=S.INK_MUTED, va="center")

# 3 count assignment: the reads of one molecule share candidate locations; one location is chosen
ax = axes[2]
gx0, gy0, cs = 5.45, 0.2, 0.75
small_grid(ax, gx0, gy0, 5, 6, cs)
cand_a, cand_b = (gx0 + 2 * cs, gy0 + 3 * cs), (gx0 + 3 * cs, gy0 + 3 * cs)
ax.add_patch(Rectangle(cand_a, cs, cs, fc=S.STAR, ec="#555555", lw=0.5, zorder=3))
ax.add_patch(Rectangle(cand_b, cs, cs, fc="white", ec=S.STAR, lw=0.9, ls=(0, (2, 1)), zorder=3))
ax.text(cand_a[0] + cs / 2, cand_a[1] + cs / 2, "A", fontsize=S.FS_NOTE, color="white", ha="center", va="center", zorder=4)
ax.text(cand_b[0] + cs / 2, cand_b[1] + cs / 2, "B", fontsize=S.FS_NOTE, color=S.STAR, ha="center", va="center", zorder=4)
for k, cands in enumerate(("A B", "A B", "A")):
    y = 4.6 - k * 0.85
    ax.add_patch(Rectangle((0.2, y - 0.16), 1.6, 0.32, fc="#dddddd", ec="#555555", lw=0.4))
    ax.text(2.1, y, cands, fontsize=S.FS_NOTE, color=S.STAR, va="center")
ax.text(0.2, 1.15, "reads of one\nmolecule: one\nlocation chosen", fontsize=S.FS_NOTE, color=S.INK_MUTED, va="center",
        linespacing=1.2)
ax.add_patch(FancyArrowPatch((3.5, 3.8), (cand_a[0] - 0.1, cand_a[1] + cs / 2), arrowstyle="-|>", mutation_scale=6,
                             lw=0.7, color=S.INK_MUTED, connectionstyle="arc3,rad=-0.2"))

# 4 registration, top view: counts are lower under each nucleus than just around it (a local hollow, as in
# Fig. 4a, step 2), and the hollows lie one location off the nuclei in the image
ax = axes[3]
s = 0.68
x0, y0 = 4.55, 0.2
nuclei = [(6.1, 3.0), (8.6, 1.3)]
shift = (s, -s)                                   # the counts lie one location off the image
for a in range(8):
    for b in range(7):
        x, y = x0 + (a + 0.5) * s, y0 + (b + 0.5) * s
        d = min(((x - cx - shift[0]) / 1.3) ** 2 + ((y - cy - shift[1]) / 1.05) ** 2 for cx, cy in nuclei)
        base = 0.95 - 0.085 * a                                       # denser tissue to the left
        level = base - 0.30 if d < 1.0 else base
        ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc="#4b5563", alpha=level, ec="white", lw=0.3))
for cx, cy in nuclei:
    ax.add_patch(Ellipse((cx, cy), 1.8, 1.35, fc="none", ec=NUCLEUS, lw=1.1))
cx, cy = nuclei[0]
ax.plot([cx], [cy], "o", ms=2.2, color=NUCLEUS, zorder=7)
ax.add_patch(FancyArrowPatch((cx, cy), (cx + shift[0], cy + shift[1]), arrowstyle="-|>", mutation_scale=6,
                             lw=0.9, color=S.INK, zorder=6))
ax.plot([0.15, 0.75], [4.75, 4.75], color=NUCLEUS, lw=1.1)
ax.text(0.95, 4.75, "nuclei", fontsize=S.FS_NOTE, color=NUCLEUS, va="center")
ax.text(0.15, 3.95, "counts", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.add_patch(Rectangle((0.15, 2.95), 0.6, 0.6, fc="#4b5563", alpha=0.80, ec="none"))
ax.text(0.95, 3.25, "more", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.add_patch(Rectangle((0.15, 2.15), 0.6, 0.6, fc="#4b5563", alpha=0.20, ec="none"))
ax.text(0.95, 2.45, "fewer", fontsize=S.FS_NOTE, color=S.INK, va="center")
ax.annotate("", xy=(0.95, 1.35), xytext=(0.15, 1.35), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=S.INK, mutation_scale=6))
ax.text(1.15, 1.35, "offset", fontsize=S.FS_NOTE, color=S.INK, va="center")

# --- b: processing path --------------------------------------------------------
ax = fig.add_subplot(outer[1])
ax.set_xlim(0, 200)
ax.set_ylim(0, 96)
ax.axis("off")
ax_b = ax

# inputs
box(ax, 0, 50, 28, 38, "Paired FASTQ", "")
ax.text(14, 72, "Read 1\nUMI + spatial\nbarcode", ha="center", va="center", fontsize=S.FS_NOTE)
ax.text(14, 57.5, "Read 2\ntranscript\nor probe", ha="center", va="center", fontsize=S.FS_NOTE)
ax.plot([3, 25], [64.5, 64.5], color=EDGE, lw=0.5)
box(ax, 0, 4, 28, 28, "Images", "microscope (H&E)\nCytAssist", face="white")

ax.add_patch(FancyBboxPatch((32, 1), 134, 94, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor="none", edgecolor=S.STAR, lw=0.9, ls=(0, (3, 2))))
ax.text(34, 92.6, "STARgrid", fontsize=S.FS_LABEL, color=S.STAR, fontweight="bold", va="top")
ax.text(52, 92.6, "steps 1–3: one STAR Suite process", fontsize=S.FS_NOTE, color=S.STAR, va="top")
ax.text(34, 37.0, "step 4: registration", fontsize=S.FS_NOTE, color=S.STAR, va="top")

box(ax, 37, 67, 38, 19, "1 Barcode decoding",
    f"every location at the\nminimum edit distance\n(≤ {M['perHalfED']} edits per barcode)", face=PALE)
box(ax, 37, 43, 38, 21, "2 Gene assignment",
    "RNA-seq: alignment,\nintrons included\nFlex: half-probe match,\nno genome", face=PALE)
box(ax, 81, 54, 32, 21, "3 Read cliques",
    "same gene,\nsame raw UMI,\na shared candidate\nlocation", face=PALE)
box(ax, 119, 54, 43, 21, "3 Molecule resolution",
    "Phred likelihood +\nbarcode-frequency prior;\nUMI correction per\nlocation", face=PALE)
box(ax, 170, 50, 30, 38, "Count matrices",
    "STRICT\nHARD (reported)\nSOFT\nGATED\n\nat 2, 8 and 16 µm", face="white")

box(ax, 37, 4, 46, 28, "4a From the images",
    "microscope to CytAssist\nby image features;\nCytAssist to grid by\nfiducial marks; no counts", face=PALE)
box(ax, 89, 4, 73, 28, "4b From the counts",
    "nucleus masks moved over the counts to\nthe offset of highest correlation ratio\nacross the nucleus edge, for the slide\nor region by region", face=PALE)
box(ax, 170, 4, 30, 28, "Registered", "masks of nuclei\nand cells on the\ncapture grid", face="white")

arrow(ax, 28, 72, 37, 76.5)     # Read 1 -> decoding
arrow(ax, 28, 57.5, 37, 53.5)   # Read 2 -> gene assignment
arrow(ax, 75, 76.5, 81, 68)
arrow(ax, 75, 53.5, 81, 61)
arrow(ax, 113, 64.5, 119, 64.5)
arrow(ax, 162, 64.5, 170, 64.5)
arrow(ax, 28, 18, 37, 18)       # images -> registration
arrow(ax, 83, 18, 89, 18)
arrow(ax, 162, 18, 170, 18)
ax.add_patch(FancyArrowPatch((185, 50), (150, 32.5), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                             color=S.INK_MUTED, connectionstyle="arc3,rad=-0.25"))   # counts -> second stage
ax.text(187, 41, "counts", fontsize=S.FS_NOTE, color=S.INK_MUTED, ha="left", va="center")

# --- c: barcode layout -----------------------------------------------------------
ax = fig.add_subplot(bottom[0, 0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax_c = ax
ax.text(0, 96, f"BC1 names the column, BC2 the row:\n{M['gridSide']} × {M['gridSide']} capture locations", fontsize=S.FS_NOTE,
        va="top", color=S.INK, linespacing=1.3)
extent = ax.get_window_extent()
g0x, g0y, cell = 30, 24, 8
cell_h = cell * extent.width / extent.height      # the same length on the page as cell
bc1 = [M["designBCOneMinLength"], M["designBCOneMaxLength"]] * 2      # lengths alternate from column to column
bc2 = [M["designBCTwoMinLength"], M["designBCTwoMaxLength"]] * 2      # and from row to row
for i, l1 in enumerate(bc1):
    ax.text(g0x + (i + 0.5) * cell, g0y + 4 * cell_h + 2.5, str(l1), ha="center", fontsize=S.FS_NOTE, color=S.STAR)
for j, l2 in enumerate(bc2):
    ax.text(g0x - 3, g0y + (3 - j + 0.5) * cell_h, str(l2), ha="right", va="center", fontsize=S.FS_NOTE, color=S.STAR)
    for i in range(4):
        ax.add_patch(Rectangle((g0x + i * cell, g0y + (3 - j) * cell_h), cell, cell_h,
                               facecolor="white", edgecolor=EDGE, lw=0.6))
ax.text(g0x + 2 * cell, g0y + 4 * cell_h + 9, "BC1 length (bases)", ha="center", fontsize=S.FS_NOTE, color=S.INK_MUTED)
ax.text(g0x - 22, g0y + 2 * cell_h, "BC2\nlength", ha="center", va="center", fontsize=S.FS_NOTE, color=S.INK_MUTED)
ax.text(0, 15, "Lengths alternate between neighbours,\nso a barcode read one base short or\nlong has the length of its neighbour.",
        va="top", fontsize=S.FS_NOTE, color=S.INK_MUTED, linespacing=1.3)

# --- d: molecule resolution --------------------------------------------------------
ax = fig.add_subplot(bottom[0, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax_d = ax
reads = [("read 1", ["A", "B"]), ("read 2", ["A", "B"]), ("read 3", ["A"])]
for k, (name, cands) in enumerate(reads):
    y = 80 - k * 14
    ax.text(0, y, name, fontsize=S.FS_NOTE, va="center", color=S.INK)
    for m, c in enumerate(cands):
        ax.add_patch(Rectangle((16 + m * 9, y - 4), 7, 8, facecolor=PALE, edgecolor=S.STAR, lw=0.6))
        ax.text(16 + m * 9 + 3.5, y, c, ha="center", va="center", fontsize=S.FS_NOTE, color=S.STAR)
ax.text(0, 30, "one read clique:\nsame gene,\nsame raw UMI", fontsize=S.FS_NOTE, va="center", color=S.INK_MUTED, linespacing=1.3)
arrow(ax, 36, 66, 44, 66)
ax.text(46, 84, "Candidates A, B", fontsize=S.FS_NOTE, color=S.INK, fontweight="bold")
ax.text(46, 76, "posterior: A ≫ B", fontsize=S.FS_NOTE, color=S.INK)
policies = [("STRICT", "dropped: two candidates"), ("HARD", "A"), ("SOFT", "A and B, by posterior"),
            ("GATED", "A, if confident")]
for k, (p, what) in enumerate(policies):
    y = 64 - k * 14
    ax.text(46, y, p, fontsize=S.FS_NOTE, color=S.STAR if p == "HARD" else S.INK,
            fontweight="bold" if p == "HARD" else "normal", va="center")
    ax.text(46, y - 5.6, what, fontsize=S.FS_NOTE, color=S.INK_MUTED, va="center")

# --- panel heads: letter and title on one baseline, as in the other figures ------------------
fig.canvas.draw()


def head(ax, letter, title, dy=0.012):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.01, pos.y1 + dy, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline", color=S.INK)
    fig.text(pos.x0 + 0.014, pos.y1 + dy, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)


head(axes[0], "a", "The four steps of STARgrid")
head(ax_b, "b", "Inputs, steps and outputs", dy=0.0)
y_cd = max(ax_c.get_position().y1, ax_d.get_position().y1)
for ax, letter, title in ((ax_c, "c", "The spatial barcode"), (ax_d, "d", "Resolving a read clique")):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.01, y_cd + 0.012, letter.upper(), fontsize=S.FS_PANEL, fontweight="bold", va="baseline",
             color=S.INK)
    fig.text(pos.x0 + 0.014, y_cd + 0.012, title, fontsize=S.FS_TITLE, va="baseline", color=S.INK)

S.save(fig, OUT, "fig1_system")
