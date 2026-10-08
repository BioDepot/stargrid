#!/usr/bin/env python3
"""Fig. 3: when a read matches two squares, the two pipelines choose differently.

Inputs: figures/data/fig3_edit_joint.tsv and fig3_summary.json (written by
fig3_policy_data.py from the July 2026 audit of Space Ranger's BAM),
figures/scripts/sr_audit_worked_cases.tsv, and the number macros.
Usage: fig3_policy.py [OUT_DIR]
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_macros  # noqa: E402
import spatial_paper_figstyle as S  # noqa: E402

DATA = HERE.parent / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
M = paper_macros.load()
S.apply()
MONO = "DejaVu Sans Mono"
GREY = "#9ca3af"
LIGHT = "#d1d5db"


def num(key):
    return float(M[key].replace(",", ""))


def align(read, oligo):
    """Unit-cost global alignment; returns (read_row, oligo_row) with '-' gaps."""
    n, m = len(read), len(oligo)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (read[i - 1] != oligo[j - 1]))
    a, b, i, j = [], [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (read[i - 1] != oligo[j - 1]):
            a.append(read[i - 1]); b.append(oligo[j - 1]); i -= 1; j -= 1
        elif j and d[i][j] == d[i][j - 1] + 1:
            a.append("-"); b.append(oligo[j - 1]); j -= 1
        else:
            a.append(read[i - 1]); b.append("-"); i -= 1
    return "".join(reversed(a)), "".join(reversed(b))


fig = plt.figure(figsize=(S.DOUBLE_COL, 112 * S.MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.0], hspace=0.62)
top = outer[0].subgridspec(1, 3, width_ratios=[0.95, 0.85, 1.05], wspace=0.95)
bottom = outer[1].subgridspec(1, 2, width_ratios=[2.05, 1], wspace=0.62)

# --- a: where Space Ranger's square falls -------------------------------------
ax = fig.add_subplot(top[0, 0])
ax.axis("off")
S.panel_label(ax, "a", dx=-0.10, dy=1.13)
ax.set_title("Reads with a location in range", pad=6)
rows = [("A closest location", M["taxSRMinimumPct"], M["taxSRMinimum"], S.INK),
        ("A different location", M["taxSRNonminimumPct"], M["taxSRNonminimum"], S.SR),
        ("No location", M["taxSRUnassignedPct"], M["taxSRUnassigned"], S.INK_MUTED)]
ax.text(0.0, 0.93, f"{M['taxCandidateReads']} reads; Space Ranger reports:", transform=ax.transAxes,
        fontsize=6, color=S.INK_MUTED, va="top")
for k, (label, pct, count, colour) in enumerate(rows):
    y = 0.70 - k * 0.27
    ax.text(0.0, y, f"{float(pct):.2f}%" if float(pct) > 1 else f"{float(pct):.3f}%", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=colour, va="center")
    ax.text(0.58, y, label, transform=ax.transAxes, fontsize=6.5, color=S.INK, va="center")
    if count:
        ax.text(0.58, y - 0.10, f"{count} reads", transform=ax.transAxes, fontsize=5.5,
                color=S.INK_MUTED, va="center")

# --- b: edit distances, closest vs Space Ranger --------------------------------
ax = fig.add_subplot(top[0, 1])
S.panel_label(ax, "b", dx=-0.42, dy=1.13)
ax.set_title("Edit counts", pad=6)
joint = [(int(r["closest_edits"]), int(r["space_ranger_edits"]), int(r["reads"]))
         for r in csv.DictReader(open(DATA / "fig3_edit_joint.tsv"), delimiter="\t")]
total = sum(c for _, _, c in joint)
ax.plot([0.5, 4.5], [0.5, 4.5], color=S.CHANCE, lw=0.6, ls="--", zorder=1)
ax.text(2.2, 1.78, "equal", fontsize=5.3, color=S.INK_MUTED, ha="left", va="bottom", rotation=42)
peak = max(c for _, _, c in joint)
for x, y, c in joint:
    size = 12 + 380 * (c / peak) ** 0.5
    ax.scatter([x], [y], s=size, color=S.SR if y > x else S.INK_MUTED, edgecolor="white", lw=0.6, zorder=3)
    share = 100 * c / total
    label = f"{share:.1f}%" if share >= 0.1 else f"{c:,}"
    dx, dy = (0.0, 0.40) if c == peak else (0.16, -0.34)
    ax.text(x + dx, y + dy, label, fontsize=5.5, color=S.INK, ha="center" if c == peak else "left", va="center")
ax.set_xlim(0.5, 3.75)
ax.set_ylim(1.35, 4.6)
ax.set_xticks([1, 2, 3])
ax.set_yticks([2, 3, 4])
ax.set_xlabel("Edits to the closest location")
ax.set_ylabel("Edits to Space Ranger's location")
S.hide_y_grid_behind(ax)
ax.xaxis.grid(True)

# --- c: geometry ---------------------------------------------------------------
ax = fig.add_subplot(top[0, 2])
S.panel_label(ax, "c", dx=-0.56, dy=1.13)
ax.set_title("Fit of Space Ranger's location", pad=6)
summ = json.load(open(DATA / "fig3_summary.json"))
cls = summ["sr_edit_class"]
n_all = sum(cls.values())
geo = [("BC1 one short,\nBC2 one long", cls["deletion_frame+insertion_frame"]),
       ("BC1 edited,\nBC2 one long", cls["same_length_edit+insertion_frame"])]
geo.append(("Other", n_all - sum(c for _, c in geo)))
ys = range(len(geo))[::-1]
for y, (label, c) in zip(ys, geo):
    ax.barh(y, 100 * c / n_all, color=S.SR if label != "Other" else LIGHT, height=0.55)
    ax.text(100 * c / n_all + 1.5, y, f"{100 * c / n_all:.1f}%", va="center", fontsize=5.8, color=S.INK)
ax.set_yticks(list(ys))
ax.set_yticklabels([g[0] for g in geo], fontsize=5.8)
ax.set_xlim(0, 100)
ax.set_xlabel(f"% of the {M['taxSRNonminimum']} reads")
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True)
ax.set_axisbelow(True)

# --- d: worked example -----------------------------------------------------------
ax = fig.add_subplot(bottom[0, 0])
ax.axis("off")
S.panel_label(ax, "d", dx=-0.045, dy=1.10)
ax.set_title("One read, two locations (Table 5, read 1)", pad=6)
cases = [r for r in csv.DictReader(open(HERE / "sr_audit_worked_cases.tsv"), delimiter="\t")
         if r["case"] == "same 8 um bin"]
closest = next(r for r in cases if r["square"] == "closest")
sr = next(r for r in cases if r["square"] == "Space Ranger")
read1, read2 = closest["split"].split("|")
x0, cw = 0.255, 0.0182         # text origin and character width in axes units


def draw_seq(y, seq, marks=(), colour=S.INK, weight="normal"):
    for k, ch in enumerate(seq):
        x = x0 + k * cw
        if k in marks:
            ax.add_patch(Rectangle((x - 0.001, y - 0.07), cw * 0.95, 0.14, transform=ax.transAxes,
                                   color=colour, alpha=0.22, lw=0))
        ax.text(x + cw * 0.45, y, ch, transform=ax.transAxes, family=MONO, fontsize=6.3,
                ha="center", va="center", color=S.INK, fontweight=weight)


def aligned(read_part, oligo):
    r, o = align(read_part, oligo)
    return r, o, [k for k, (p, q) in enumerate(zip(r, o)) if p != q]


r1c, o1c, m1c = aligned(read1, closest["bc1_oligo"])
r2c, o2c, m2c = aligned(read2, closest["bc2_oligo"])
r2s, o2s, m2s = aligned(read2, sr["bc2_oligo"])
width1 = max(len(r1c), len(o1c))
gap = width1 + 1
read_row = r1c + " " + (r2s if len(r2s) >= len(r2c) else r2c)
labels = [(0.80, "Read barcode", read_row, [], S.INK),
          (0.52, f"Closest\nrow {closest['row']}", o1c + " " + o2c.rjust(len(r2s)), m1c + [gap + k + (len(r2s) - len(o2c)) for k in m2c], S.STAR),
          (0.18, f"Space Ranger\nrow {sr['row']}", o1c + " " + o2s, m1c + [gap + k for k in m2s], S.SR)]
for y, name, seq, marks, colour in labels:
    ax.text(0.0, y, name, transform=ax.transAxes, fontsize=6, color=colour if colour != S.INK else S.INK,
            va="center", fontweight="bold" if colour != S.INK else "normal")
    draw_seq(y, seq, marks, colour)
ax.text(x0 + (width1 / 2) * cw, 0.98, "BC1", transform=ax.transAxes, ha="center", fontsize=6, color=S.INK_MUTED)
ax.text(x0 + (gap + len(r2s) / 2) * cw, 0.98, "BC2", transform=ax.transAxes, ha="center", fontsize=6, color=S.INK_MUTED)
xe = x0 + (gap + len(r2s) + 1.2) * cw
ax.text(xe, 0.52, f"{closest['total']} edit", transform=ax.transAxes, fontsize=6.5, color=S.STAR,
        fontweight="bold", va="center")
ax.text(xe, 0.18, f"{sr['total']} edits", transform=ax.transAxes, fontsize=6.5, color=S.SR,
        fontweight="bold", va="center")
ax.text(0.0, -0.06, "Shaded: edits relative to the read. The two BC2 oligos differ only by the first base;\n"
        "the locations are in adjacent rows, 2 µm apart.", transform=ax.transAxes, fontsize=5.5,
        color=S.INK_MUTED, va="top")

# --- e: spatial consequence --------------------------------------------------------
ax = fig.add_subplot(bottom[0, 1])
S.panel_label(ax, "e", dx=-0.66, dy=1.10)
ax.set_title("Space Ranger's location vs closest", pad=6)
pr = summ["parent_relation"]
n_pr = sum(pr.values())
parts = [("Same 8 µm bin", pr["same_8um"]), ("Same 16 µm bin", pr["same_16um_only"]),
         ("Other 16 µm bin", pr["cross_16um"])]
ys = range(len(parts))[::-1]
for y, (label, c) in zip(ys, parts):
    ax.barh(y, 100 * c / n_pr, color=S.SR, height=0.55)
    ax.text(100 * c / n_pr + 1.5, y, f"{100 * c / n_pr:.1f}%", va="center", fontsize=5.8, color=S.INK)
ax.set_yticks(list(ys))
ax.set_yticklabels([p[0] for p in parts], fontsize=5.8)
ax.set_xlim(0, 100)
ax.set_xlabel("% of reads")
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True)
ax.set_axisbelow(True)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_policy.{ext}")
print("wrote", OUT / "fig3_policy.pdf")
