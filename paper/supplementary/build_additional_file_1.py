#!/usr/bin/env python3
"""Additional file 1: Supplementary Tables S1-S5, generated from the accepted v1.9.5 tables.

Reads paper_results/spatial_v1_9_5_20260915 and writes supplementary_tables.tex,
which additional_file_1.tex inputs. Every value is copied from an accepted table.
Usage: build_additional_file_1.py [OUT_TEX]
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[1] / "paper_results" / "spatial_v1_9_5_20260915"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "supplementary_tables.tex"
POLICY = {"strict": "strict", "postcollapse_soft": "soft-expected", "soft_expected": "soft-expected",
          "postcollapse_hard": "hard", "hard": "hard", "gated_hard": "gated-hard"}
ORDER = ["strict", "soft-expected", "hard", "gated-hard"]


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def f(x, places):
    return f"{float(x):.{places}f}"


def sci(x, places=2):
    mantissa, exponent = format(float(x), f".{places}e").split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def n(x):
    return f"{int(round(float(x))):,}"


def table(caption, label, header, body, spec, note=None):
    out = [r"\begin{table}[H]", r"\centering\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(rf"\par\smallskip\parbox{{0.92\linewidth}}{{\scriptsize {note}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


blocks = []

# --- S1: policy-by-scale concordance ---------------------------------------------------
body = []
for slide, path in (("Colorectal", R / "crc_concordance/matrix_concordance.tsv"),
                    ("Ovarian 3$'$", R / "ovarian_concordance/matrix_concordance.tsv")):
    data = [r for r in rows(path) if r["umi_mode"] == "1mm_cr"]
    data.sort(key=lambda r: (int(r["scale_um"]), ORDER.index(POLICY[r["policy"]])))
    for r in data:
        body.append([slide, r["scale_um"], POLICY[r["policy"]], f(r["mass_ratio"], 4), f(r["exact_entry_fraction"], 4),
                     f(r["gene_total_pearson"], 6), f(r["bin_total_pearson"], 5)])
for r in sorted((r for r in rows(R / "spatch_concordance/aggregate_concordance.tsv") if r["umi_mode"] == "1mm_cr"),
                key=lambda r: ORDER.index(POLICY[r["method"]])):
    body.append(["SPATCH", r["scale_um"], POLICY[r["method"]], f(r["mass_ratio"], 4), "---",
                 f(r["gene_total_pearson"], 6), f(r["spatial_bin_total_pearson_on_space_ranger_axis"], 5)])
blocks.append(table(
    r"\textbf{Table S1. Concordance with Space Ranger for every field, scale and slide.} Mass ratio is STAR Suite "
    r"molecule total divided by the Space Ranger total. Exact entries are the fraction of entries nonzero in either matrix that "
    r"have identical counts. Correlations are Pearson $r$. SPATCH uses the bins in Space Ranger's published filtered matrix at 8\,\textmu m.",
    "tab:s1", ["Slide", "Bin (\\textmu m)", "Field", "Mass ratio", "Exact entries", "Gene-total $r$", "Bin-total $r$"],
    body, "llllrrr"))

# --- S2: SPATCH with the 2024-A reference ------------------------------------------------
body = []
for r in sorted((r for r in rows(R / "spatch_2024a_concordance/aggregate_concordance.tsv") if r["umi_mode"] == "1mm_cr"),
                key=lambda r: ORDER.index(POLICY[r["method"]])):
    body.append([POLICY[r["method"]], n(r["star_features"]), n(r["shared_features"]),
                 n(r["star_mass_on_features_absent_from_space_ranger"]), n(r["absent_feature_space_ranger_mass"]),
                 f(r["mass_ratio"], 4), f(r["gene_total_pearson_on_shared_features"], 6),
                 f(r["spatial_bin_total_pearson_on_space_ranger_axis"], 5)])
first = rows(R / "spatch_2024a_concordance/aggregate_concordance.tsv")[0]
blocks.append(table(
    r"\textbf{Table S2. SPATCH slide processed with the 2024-A reference.} The published Space Ranger matrix uses "
    rf"the 2020-A probe set ({n(first['space_ranger_features'])} features); the union of the gene sets has "
    rf"{n(first['union_features'])} features. Unmatched mass counts molecules on genes present in only one output. "
    r"Gene-total Pearson $r$ uses shared genes; bin-total Pearson $r$ uses the 8\,\textmu m bins in Space Ranger's published matrix.",
    "tab:s2", ["Field", "STAR features", "Shared", "STAR mass, unmatched", "SR mass, unmatched", "Mass ratio",
               "Gene-total $r$", "Bin-total $r$"], body, "lrrrrrrr"))

# --- S3: CODEX, all fields and grids ------------------------------------------------------
data = rows(R / "codex_both_arms/method_summary.tsv")
grids = sorted({int(r["bin_size_um"]) for r in data})
methods = list(dict.fromkeys(r["method"] for r in data))
label = {"space_ranger_2020a": "Space Ranger (2020-A)"}
for m in methods:
    if m.startswith("star_"):
        ref, pol = m.split("_", 2)[1], m.split("_", 2)[2]
        label[m] = f"STAR {POLICY.get(pol, pol)} ({ref[:4]}-{ref[4:].upper()})"
body = [[label[m]] + [f(next(r for r in data if r["method"] == m and int(r["bin_size_um"]) == g)["macro_roc_auc"], 4)
                      for g in grids] for m in methods]
markers = data[0]["markers_scored"]
blocks.append(table(
    rf"\textbf{{Table S3. CODEX comparison on the SPATCH slide.}} Mean ROC AUC across {markers} matched "
    r"RNA--protein markers. For each marker, protein-positive squares are those at or above the 80th percentile of CODEX intensity, on "
    rf"grids from {min(grids)} to {max(grids)}\,\textmu m.",
    "tab:s3", ["Field"] + [f"{g}\\,\\textmu m" for g in grids], body, "l" + "r" * len(grids)))

recon = rows(R / "codex_shared_excess_2020a/component_reconciliation.tsv")
body = [[r["bin_size_um"], n(r["space_ranger_direct_marker_mass"]), n(r["star_direct_marker_mass"]),
         n(r["shared_direct_marker_mass"]), n(r["star_excess_direct_marker_mass"]),
         n(r["space_ranger_only_direct_marker_mass"])] for r in recon if r["method"].endswith("hard")]
if body:
    blocks.append(table(
        r"\textbf{Table S3 (continued). Marker mass shared and added (2020-A, hard).} RNA counts for the matched markers in "
        r"the scored squares. Shared counts are the smaller count from the two pipelines in each square; each pipeline's remainder is its residual. These are differences in counts, not matched molecules.",
        "tab:s3b", ["Grid (\\textmu m)", "Space Ranger", "STAR", "Shared", "STAR only", "Space Ranger only"],
        body, "rrrrrr"))

# --- S4: registration sensitivity (colorectal) ----------------------------------------------
reg = json.load(open(R / "crc_frozen_registration_sensitivity/summary.json"))
mask = reg["mask_reprojection"]
sens = rows(R / "crc_frozen_registration_sensitivity/registration_sensitivity.tsv")
fields = [("space_ranger", "Space Ranger"), ("strict", "STAR strict"), ("postcollapse_hard", "STAR hard"),
          ("postcollapse_hard_star_only", "hard, STAR only"), ("postcollapse_hard_space_ranger_only", "hard, Space Ranger only")]
metrics = [("in_cell_mass_fraction", "In-cell fraction"), ("in_cell_roc_auc", "Cell ROC AUC"),
           ("in_nucleus_roc_auc", "Nucleus ROC AUC")]
body = []
for key, name in fields:
    row = [name]
    for metric, _ in metrics:
        hit = [r for r in sens if r["field"] == key and r["metric"] == metric]
        if hit:
            row.append(f"{f(hit[0]['space_ranger_registration_value'], 4)} / {f(hit[0]['native_registration_value'], 4)}")
        else:
            row.append("---")
    body.append(row)
blocks.append(table(
    r"\textbf{Table S4. Registration sensitivity on the colorectal slide.} Each entry gives the score with Space "
    r"Ranger's microscope registration, followed by the score with our registration, using the same image masks "
    rf"reprojected through each. Reprojection changes {100 * mask['cell_mask_xor_fraction']:.2f}\% of grid squares' "
    rf"cell-mask state and {100 * mask['nucleus_mask_xor_fraction']:.2f}\% of their nucleus-mask state "
    rf"(median displacement {mask['median_displacement_um']:.2f}\,\textmu m). The largest change in any STAR minus "
    rf"Space Ranger score difference is ${sci(reg['maximum_absolute_star_minus_vendor_contrast_change'])}$.",
    "tab:s4", ["Field"] + [m[1] for m in metrics], body, "lrrr"))

# --- S5: ovarian slide, annotated multimapper setting ---------------------------------------
annot = json.load(open(R / "summary_ovarian_gex_2024a_annotated/summary.json"))["policy_masses_full_capture_axis"]
compat = json.load(open(R / "summary_ovarian_gex_2024a_compat/summary.json"))["policy_masses_full_capture_axis"]
sr = float(next(r for r in rows(R / "three_slide_summary/count_gain.tsv")
                if r["dataset"] == "OVARIAN_GEX" and r["method"] == "hard")["space_ranger_biological_mass"])
body = [[POLICY[p], n(compat[p]), f(100 * (compat[p] / sr - 1), 2), n(annot[p]), f(100 * (annot[p] / sr - 1), 2)]
        for p in ("strict", "soft_expected", "hard", "gated_hard")]
blocks.append(table(
    rf"\textbf{{Table S5. Ovarian 3$'$ slide under the two multimapper settings.}} Molecule totals over the full capture "
    rf"grid and percentage differences from Space Ranger 4.1.0 with introns included ({n(sr)} molecules).",
    "tab:s5", ["Field", "Compatibility", "vs SR (\\%)", "Annotated", "vs SR (\\%)"], body, "lrrrr"))

xa = [r for r in rows(R / "xenium_annotated_cancer/method_summary.tsv") if r["scope"] == "label_leakage_controlled"]
xc = [r for r in rows(R / "xenium_compatibility_cancer/method_summary.tsv") if r["scope"] == "label_leakage_controlled"]
ta = {r["method"]: r for r in rows(R / "xenium_annotated_top20/top20_summary.tsv")}
tc = {r["method"]: r for r in rows(R / "xenium_compatibility_top20/top20_summary.tsv")}
body = []
for m in ("space_ranger", "strict", "hard"):
    c = next(r for r in xc if r["method"] == m)
    a = next(r for r in xa if r["method"] == m)
    body.append([{"space_ranger": "Space Ranger", **POLICY}[m], f(c["effect_pearson_vs_xenium"], 3),
                 f(a["effect_pearson_vs_xenium"], 3), f(tc[m]["median_spatial_pearson"], 3),
                 f(ta[m]["median_spatial_pearson"], 3)])
blocks.append(table(
    r"\textbf{Table S5 (continued). Xenium endpoints under the two settings.} Pearson correlation with Xenium of cancer-rich versus cancer-poor effects "
    r"across the cancer panel, excluding the label genes and MECOM, and median spatial Pearson correlation for the most abundant Xenium panel genes. The latter gene set is fixed by Xenium abundance before comparing Visium fields; its size is given in the column headings.",
    "tab:s5b", ["Field", "Effect $r$, compat.", "Effect $r$, annot.", "Top-20 $r$, compat.", "Top-20 $r$, annot."],
    body, "lrrrr"))

OUT.write_text("% Generated by supplementary/build_additional_file_1.py from the accepted v1.9.5 tables.\n"
               + "\n\n".join(blocks) + "\n")
print("wrote", OUT)
