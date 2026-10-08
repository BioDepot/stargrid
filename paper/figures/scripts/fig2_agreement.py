#!/usr/bin/env python3
"""Fig. 2: the output agrees with Space Ranger's on three slides and both chemistries.

Inputs (accepted v1.9.5 tables under paper_results/spatial_v1_9_5_20260915):
primary summaries (read accounting, policy masses), crc/ovarian
matrix_concordance.tsv, spatch aggregate_concordance.tsv and count_gain.tsv.
Usage: fig2_agreement.py [OUT_DIR]
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
STRICT = S.STAR_SOFT          # the conservative field: same hue as hard, lighter
GREY = "#d1d5db"
SLIDES = [("Colorectal\n(Flex)", "crc"), ("SPATCH\n(Flex)", "spatch"), ("Ovarian\n(RNA-seq)", "ovarian")]
SUMMARY = {"crc": R / "supporting/crc_primary_summary.json",
           "spatch": R / "summary_spatch_flex_2020a/summary.json",
           "ovarian": R / "summary_ovarian_gex_2024a_compat/summary.json"}


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


acc = {k: json.load(open(p))["accounting"] for k, p in SUMMARY.items()}
masses = {k: json.load(open(p))["policy_masses_full_capture_axis"] for k, p in SUMMARY.items()}
sr_mass = {r["dataset"].lower().replace("ovarian_gex", "ovarian"): float(r["space_ranger_biological_mass"])
           for r in rows(R / "three_slide_summary/count_gain.tsv") if r["method"] == "hard"}
conc = {"crc": [r for r in rows(R / "crc_concordance/matrix_concordance.tsv")
                if r["policy"] == "postcollapse_hard" and r["umi_mode"] == "1mm_cr"],
        "ovarian": [r for r in rows(R / "ovarian_concordance/matrix_concordance.tsv")
                    if r["policy"] == "postcollapse_hard" and r["umi_mode"] == "1mm_cr"]}
spatch = next(r for r in rows(R / "spatch_concordance/aggregate_concordance.tsv")
              if r["method"] == "hard" and r["umi_mode"] == "1mm_cr")
gene_r = {"crc": float(conc["crc"][0]["gene_total_pearson"]),
          "spatch": float(spatch["gene_total_pearson"]),
          "ovarian": float(conc["ovarian"][0]["gene_total_pearson"])}

fig = plt.figure(figsize=(S.DOUBLE_COL, 112 * S.MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.85, wspace=0.42)

# --- a: read accounting ------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
S.panel_label(ax, "a", dx=-0.16, dy=1.14)
ax.set_title("Read accounting", pad=6)
stages = [("with a candidate location", "reads_with_candidates", GREY),
          ("with a gene and a candidate", "joined_reads", S.STAR)]
bar_h = 0.34
for i, (label, key) in enumerate(SLIDES):
    y0 = len(SLIDES) - 1 - i
    pairs = acc[key]["reads_decoded"]
    for j, (stage, field, colour) in enumerate(stages):
        v = 100 * acc[key][field] / pairs
        y = y0 + (0.19 if j == 0 else -0.19)
        ax.barh(y, v, height=bar_h, color=colour)
        ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=5.6, color=S.INK)
    ax.text(101, y0 + 0.47, f"{pairs / 1e6:.0f}M read pairs; {acc[key]['read_cliques'] / 1e6:.0f}M read cliques",
            fontsize=5.3, color=S.INK_MUTED, ha="right", va="bottom")
ax.set_yticks(range(len(SLIDES))[::-1])
ax.set_yticklabels([s[0] for s in SLIDES], fontsize=6)
ax.set_xlim(0, 112)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("% of read pairs")
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in stages],
          labels=[s for s, _, _ in stages], loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=2,
          fontsize=5.6, frameon=False, borderaxespad=0)

# --- b: gene totals -----------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
S.panel_label(ax, "b", dx=-0.30, dy=1.16)
ax.set_title("Gene totals", pad=6)
for i, (label, key) in enumerate(SLIDES):
    ax.scatter([i], [gene_r[key]], s=22, color=S.STAR, zorder=3)
    ax.text(i, gene_r[key] + 0.00012, f"{gene_r[key]:.4f}" if gene_r[key] < 0.9999 else f"{gene_r[key]:.6f}",
            ha="center", va="bottom", fontsize=5.6, color=S.INK)
ax.set_xticks(range(len(SLIDES)))
ax.set_xticklabels([s[0] for s in SLIDES], fontsize=6)
ax.set_xlim(-0.5, len(SLIDES) - 0.5)
ax.set_ylim(0.998, 1.0006)
ax.set_ylabel("Pearson r with Space Ranger")
S.hide_y_grid_behind(ax)

# --- c: bin totals by scale ------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
S.panel_label(ax, "c", dx=-0.16, dy=1.19)
ax.set_title("Spatial bin totals", pad=6)
markers = {"crc": "o", "ovarian": "s", "spatch": "^"}
names = {"crc": "Colorectal", "ovarian": "Ovarian", "spatch": "SPATCH"}
for key in ("crc", "ovarian"):
    pts = sorted((int(r["scale_um"]), float(r["bin_total_pearson"])) for r in conc[key])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=S.STAR, lw=1.0, marker=markers[key], ms=3.5, zorder=3)
    ax.text(xs[0] - 0.6, ys[0], f"{names[key]} {ys[0]:.4f}", ha="right", va="center", fontsize=5.4, color=S.INK)
sp = float(spatch["spatial_bin_total_pearson_on_space_ranger_axis"])
ax.scatter([8], [sp], marker=markers["spatch"], s=16, color=S.STAR, zorder=3)
ax.text(8.6, sp - 0.0004, f"SPATCH {sp:.4f}", ha="left", va="top", fontsize=5.4, color=S.INK)
ax.text(16.4, 0.9998, "Colorectal and\novarian ≥ 0.9998", ha="left", va="center", fontsize=5.4, color=S.INK_MUTED)
ax.set_xticks([2, 8, 16])
ax.set_xticklabels(["2 µm", "8 µm", "16 µm"])
ax.set_xlim(-6, 22)
ax.set_ylim(0.980, 1.001)
ax.set_ylabel("Pearson r with Space Ranger")
S.hide_y_grid_behind(ax)

# --- d: molecules relative to Space Ranger -------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
S.panel_label(ax, "d", dx=-0.30, dy=1.19)
ax.set_title("Molecules, % of Space Ranger", pad=6)
w = 0.26
for i, (label, key) in enumerate(SLIDES):
    sr = sr_mass[key]
    vals = [("strict", 100 * masses[key]["strict"] / sr, STRICT),
            ("Space Ranger", 100.0, S.SR),
            ("hard", 100 * masses[key]["hard"] / sr, S.STAR)]
    for j, (name, v, colour) in enumerate(vals):
        x = i + (j - 1) * w
        ax.bar(x, v, width=w * 0.92, color=colour)
        if name != "Space Ranger":
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=5.2, color=S.INK)
ax.set_xticks(range(len(SLIDES)))
ax.set_xticklabels([s[0] for s in SLIDES], fontsize=6)
ax.set_ylim(70, 121)
ax.axhline(100, color=S.CHANCE, lw=0.5, ls="--", zorder=0)
ax.set_ylabel("% of Space Ranger molecules")
S.hide_y_grid_behind(ax)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (STRICT, S.SR, S.STAR)],
          labels=["strict", "Space Ranger", "hard"], loc="upper left", ncol=3, fontsize=5.6, frameon=False)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_agreement.{ext}")
print("wrote", OUT / "fig2_agreement.pdf")
