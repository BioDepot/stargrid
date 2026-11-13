#!/usr/bin/env python3
"""Additional file 1: Supplementary Table S2-S7, generated from the accepted tables.

Reads paper_results/spatial_v1_9_5_20260915 (S1, S2), paper_results/paper_tests_20261021 (S3, S4),
paper_results/registration_local_dip_20261106 (dip rule: S5, S6, S12) and paper_results/registration_check_20261018 (lung mask displacements), the
source split of the additional molecules (S7), and
the settings of the check from paper/registration_numbers.tex (generated from the same records). Writes
supplementary_tables.tex, which additional_file_1.tex inputs. Every value is copied from an accepted table.
Usage: build_additional_file_1.py [OUT_TEX]
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[1] / "paper_results" / "spatial_v1_9_5_20260915"
T = HERE.parents[1] / "paper_results" / "paper_tests_20261021"
C = HERE.parents[1] / "paper_results" / "registration_check_20261018"
REGISTRATION_NUMBERS = HERE.parent / "registration_numbers.tex"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "supplementary_tables.tex"
# Short policy names, as defined in the Methods (PI, 2026-10-27): HARD, STRICT, SOFT (expected counts), GATED.
POLICY = {"strict": r"\policyStrict{}", "postcollapse_soft": r"\policySoft{}", "soft_expected": r"\policySoft{}",
          "postcollapse_hard": r"\policyHard{}", "hard": r"\policyHard{}", "gated_hard": r"\policyGated{}"}
ORDER = [r"\policyStrict{}", r"\policySoft{}", r"\policyHard{}", r"\policyGated{}"]


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def f(x, places):
    return f"{float(x):.{places}f}"


def sci(x, places=2):
    mantissa, exponent = format(float(x), f".{places}e").split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def n(x):
    return f"{int(round(float(x))):,}"


def table(caption, label, header, body, spec, note=None, size=r"\footnotesize", above=(), colsep=None):
    out = [r"\begin{table}[H]", rf"\centering{size}", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if colsep:
        out.append(rf"\setlength{{\tabcolsep}}{{{colsep}}}")
    out += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *above, " & ".join(header) + r" \\", r"\midrule"]
    out += [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(rf"\par\smallskip\parbox{{0.92\linewidth}}{{\scriptsize {note}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def need(condition, message):
    if not condition:
        sys.exit(f"build_additional_file_1.py: {message}")


blocks = []

# --- S1: policy-by-scale concordance ---------------------------------------------------
# Slide H1-VM2JXXK/A1: the v1.12.2 matrices of every policy against Space Ranger on the probe set's genes (2026-11-10)
A1 = HERE.parents[1] / "paper_results" / "examined_agreement_20261027" / "target_all_policies_v1_12_2" / "summary.json"
a1 = json.loads(A1.read_text())
need(a1["status"] == "complete" and len(a1["rows"]) == 12, "S1 A1: comparison incomplete")
body = []
for slide, path in (("Colorectal", R / "crc_concordance/matrix_concordance.tsv"),
                    ("Ovarian", R / "ovarian_concordance/matrix_concordance.tsv")):
    data = [r for r in rows(path) if r["umi_mode"] == "1mm_cr"]
    data.sort(key=lambda r: (int(r["scale_um"]), ORDER.index(POLICY[r["policy"]])))
    for r in data:
        body.append([slide, r["scale_um"], POLICY[r["policy"]], f(r["mass_ratio"], 4), f(r["exact_entry_fraction"], 4),
                     f(r["gene_total_pearson"], 6), f(r["bin_total_pearson"], 5)])
    if slide == "Colorectal":
        for r in sorted((r for r in a1["rows"] if r["umi_mode"] == "1mm_cr"),
                        key=lambda r: (r["scale_um"], ORDER.index(POLICY[r["policy"]]))):
            # Seven decimals for the gene totals, as in Table 1: at six, most of them would round to 1
            body.append(["H1-VM2JXXK/A1", str(r["scale_um"]), POLICY[r["policy"]], f(r["mass_ratio"], 4),
                         f(r["exact_entry_fraction"], 4), f(r["gene_total_pearson"], 7), f(r["bin_total_pearson"], 5)])
for r in sorted((r for r in rows(R / "spatch_concordance/aggregate_concordance.tsv") if r["umi_mode"] == "1mm_cr"),
                key=lambda r: ORDER.index(POLICY[r["method"]])):
    body.append(["SPATCH", r["scale_um"], POLICY[r["method"]], f(r["mass_ratio"], 4), "---",
                 f(r["gene_total_pearson"], 6), f(r["spatial_bin_total_pearson_on_space_ranger_axis"], 5)])
blocks.append(table(
    r"\textbf{Supplemental Table S2. Concordance with Space Ranger for every resolution policy, scale and slide.} Mass ratio is the STARgrid "
    r"molecule total divided by the Space Ranger total. Exact entries are the fraction of entries nonzero in either matrix that "
    r"have identical counts. Correlations are Pearson $r$. SPATCH uses the bins in Space Ranger's published filtered matrix at 8\,\textmu m.",
    "tab:s1", ["Slide", "Bin (\\textmu m)", "Policy", "Mass ratio", "Exact entries", "Gene-total $r$", "Bin-total $r$"],
    body, "llllrrr"))

# --- S2: registration sensitivity (colorectal) ----------------------------------------------
reg = json.load(open(R / "crc_frozen_registration_sensitivity/summary.json"))
mask = reg["mask_reprojection"]
sens = rows(R / "crc_frozen_registration_sensitivity/registration_sensitivity.tsv")
fields = [("space_ranger", "Space Ranger"), ("postcollapse_hard", r"STARgrid \policyHard{}"),
          ("postcollapse_hard_star_only", r"\policyHard{}, STARgrid only"),
          ("postcollapse_hard_space_ranger_only", r"\policyHard{}, Space Ranger only")]
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
    r"\textbf{Supplemental Table S14. Registration sensitivity on the colorectal slide.} Each entry gives the score with Space "
    r"Ranger's registration of the microscope image to the CytAssist image, followed by the score with ours, using the same image masks "
    r"placed through each (Supplemental Fig.~S4). The largest change in any STARgrid minus "
    rf"Space Ranger score difference is ${sci(reg['maximum_absolute_star_minus_vendor_contrast_change'])}$.",
    "tab:s3", ["Counts"] + [m[1] for m in metrics], body, "lrrr"))

# --- Helpers for S3-S6 ----------------------------------------------------------------------
UM = 2.0                                    # one capture location


def jload(path):
    return json.loads(Path(path).read_text())


need(REGISTRATION_NUMBERS.exists(), "paper/registration_numbers.tex missing: run registration_numbers.py first")
SET = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", REGISTRATION_NUMBERS.read_text())}


def signed(text):
    """A leading hyphen-minus set as a minus sign."""
    return "$-$" + text[1:] if text.startswith("-") else text


def pp(fraction, places=2):
    text = f"{100 * float(fraction):.{places}f}"
    return signed(text.lstrip("-") if float(text) == 0 else text)


def interval(text):
    low, high = json.loads(text)
    return float(low), float(high)


def with_interval(value, bounds, places=2):
    return f"{pp(value, places)} [{pp(bounds[0], places)}, {pp(bounds[1], places)}]"


def signed_n(x):
    """Signed count with a true minus sign."""
    return (f"{x:+,}").replace("-", "$-$")


def shift(dy, dx):
    return f"({signed(str(int(dy)))}, {signed(str(int(dx)))})"


def distance_um(dy, dx):
    return UM * math.hypot(dy, dx)


def sig(p, figures=3):
    return f"{float(p):.{figures}g}"


# --- S3: cancer-gene sensitivity --------------------------------------------------------------
sens = rows(T / "cancer_sensitivity.tsv")
gate128 = jload(T / "cancer_reproduction_128um_summary.json")
need(gate128["top20_byte_identical"] is True and gate128["largest_absolute_pearson_difference"] == 0.0,
     "S3: the 128 um generator did not reproduce the accepted top-20 table")
VARIANT = {"primary": r"\textbf{Primary}", "N10": "Fewer genes", "N30": "More genes", "N50": "More genes",
           "all": "All eligible genes", "patch64": "Smaller patches", "patch256": "Larger patches",
           "spearman": "Rank correlation"}
need([r["variant"] for r in sens][0] == "primary" and {r["variant"] for r in sens} == set(VARIANT), "S3: unexpected variants")
body = []
for r in sens:
    need(r["ties"] == "0" and r["nonfinite"] == "0" and int(r["sign_test_N"]) == int(r["hard_wins"]) + int(r["hard_losses"]),
         f"S3 {r['variant']}: ties, undefined correlations or sign-test N")
    asked = re.fullmatch(r"N(\d+)", r["variant"])
    genes = r["N"] if not asked or int(asked.group(1)) == int(r["N"]) else f"{r['N']} of {asked.group(1)}"
    if asked and int(asked.group(1)) != int(r["N"]):
        continue                      # runs short of genes: the same genes as the all-gene variant
    body.append([VARIANT[r["variant"]], genes, r["patch_um"], r["correlation"].capitalize(), r["hard_wins"], r["hard_losses"],
                 sig(r["sign_test_P"]), f(r["median_space_ranger"], 3), f(r["median_hard"], 3)])
primary = sens[0]
eligible = next(r for r in sens if r["variant"] == "all")["N"]
short = [r for r in sens if (a := re.fullmatch(r"N(\d+)", r["variant"])) and int(a.group(1)) != int(r["N"])]
need(len(short) == 1 and short[0]["N"] == eligible, "S3: only the largest gene-number variant should run short of genes")
requested = re.fullmatch(r"N(\d+)", short[0]["variant"]).group(1)
blocks.append(table(
    r"\textbf{Supplemental Table S13. Sensitivity of the top-gene comparison with Xenium.} Ovarian dataset against the adjacent-section "
    rf"Xenium run. The first row is the primary comparison reported in the main text: the {primary['N']} cancer-panel genes with "
    rf"the most Xenium counts, on {primary['patch_um']}\,\textmu m patches, with Pearson correlation. Each other row changes one "
    r"of these choices; the variants are reported without adjustment for multiple "
    rf"testing. Genes: the number of genes compared; only {eligible} panel genes are measured in all count matrices, so "
    rf"the {requested}-gene variant is the all-gene variant. "
    r"\policyHard{} wins and losses: genes whose correlation with Xenium across patches is higher or lower "
    r"under \policyHard{} than under Space Ranger; no gene was tied and no correlation was undefined. $P$: exact two-sided sign test on "
    r"the wins and losses. Median correlation: the median over the genes of the correlation of each set of counts with Xenium "
    r"across patches.",
    "tab:s4", ["Variant", "Genes", "Patch (\\textmu m)", "Correlation", r"\policyHard{} wins", r"\policyHard{} losses", "$P$",
               "Space Ranger", r"\policyHard{}"],
    body, "lrrlrrrrr",
    above=(r" & & & & & & & \multicolumn{2}{c}{Median correlation} \\", r"\cmidrule(l){8-9}")))

# --- S4: nucleus by nucleus, in place and with the far-shift background ------------------------
far_rows = rows(T / "far_shift_all_summaries.tsv")
far = {(r["case"], r["group"], int(r["radius"])): r for r in far_rows if r["offset"] == "pooled"}
real = {(r["case"], r["group"], int(r["radius"])): r for r in rows(T / "amendment6_all_summaries.tsv")}
CASES = (("C1", "Ovarian", "deposited map"), ("T1", "Ovarian", "our Cellpose"), ("C2", "Colorectal", "deposited map"),
         ("July", "Colorectal, H1-VM2JXXK/A1", "deposited map"), ("VUILD49LA", "Lung core VUILD49LA", "Xenium"),
         ("TILD113LA", "Lung core TILD113LA", "Xenium"), ("TILD111LA", "Lung core TILD111LA", "Xenium"),
         ("VUHD049", "Lung core VUHD049", "Xenium"))
QUARTERS = tuple(f"density_Q{q}" for q in range(1, 5))
need({c for c, _, _ in CASES} == {r["case"] for r in far_rows} == {k[0] for k in real}, "S4: cases differ")
need(set(jload(C / "lung_same_section.json")["criteria"]) == {c for c, s, _ in CASES if s.startswith("Lung")}, "S4: lung cores differ")
offsets = sorted({r["offset"] for r in far_rows} - {"pooled"})
steps = {abs(int(v)) for o in offsets for v in o.split("_")}
need(len(offsets) == 4 and len(steps) == 1, "S4: far-shift offsets are not the four diagonals at one distance")
step = steps.pop()
smallest = {json.loads(real[(c, "all", 1)]["size_range"])[0] for c, _, _ in CASES}
draws = {r["valid_replicates"] for r in far_rows} | {r["valid_replicates"] for r in real.values()}
need(len(smallest) == 1 and len(draws) == 1, "S4: nucleus size floor or number of draws differs between cases")
ours = jload(C / "ovarian_dip_our_placement.json")["maps"]["native_cellpose"]["minimum"]
body, quarters = [], []
AGAINST = {1: "neighbours", 2: "next out"}
for case, slide, source in CASES:
    for radius in (1, 2):
        key = (case, "all", radius)
        need(float(far[key]["real_median"]) == float(real[key]["median"]), f"S4 {key}: real medians differ")
        first = radius == 1
        body.append([slide if first else "", source if first else "", n(real[key]["n"]) if first else "", AGAINST[radius],
                     with_interval(real[key]["median"], interval(real[key]["interval"])),
                     with_interval(far[key]["median"], interval(far[key]["interval"])),
                     with_interval(far[key]["real_minus_control"], interval(far[key]["difference_interval"]))])
        quarters.append([slide if first else "", source if first else "", AGAINST[radius]] +
                        [with_interval(far[(case, q, radius)]["real_minus_control"], interval(far[(case, q, radius)]["difference_interval"]))
                         for q in QUARTERS])
    if case != CASES[-1][0]:
        body.append(r"\addlinespace[2pt]")
        quarters.append(r"\addlinespace[2pt]")
ring = {radius: sum(max(abs(dy), abs(dx)) == radius for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1))
        for radius in (1, 2)}
blocks.append(table(
    r"\textbf{Supplemental Table S7. Nucleus by nucleus: paired depth in place and against a far-shift background.} For each nucleus with "
    rf"at least {smallest.pop()} capture locations in the tissue, the paired depth is one minus the mean count on its locations "
    rf"divided by the mean on the same locations moved to each of the {ring[1]} neighbouring shifts, one location away "
    rf"(neighbours), or to each of the {ring[2]} next shifts out, two locations away (next out). In place and Background are "
    rf"medians in \%, and In place $-$ background is their difference in percentage points, each with a "
    rf"percentile 95\% interval from {draws.pop()} draws of spatial blocks of the slide. In place: the nuclei where the "
    r"registration puts them, median over nuclei. Background: the same statistic with every nucleus first moved "
    rf"{step} capture locations ({UM * step:.0f}\,\textmu m) on each axis, along each of the four diagonals, median over the four placements "
    r"of every nucleus; "
    r"a moved nucleus can overlap other nuclei, so this is a background placement, not a nucleus-free one. Nuclei: nuclei in "
    r"place after leaving out those that would cross the edge of the capture area at one of these shifts and those whose mean "
    r"over these shifts is zero. Deposited map: the nucleus map deposited with the dataset. Our Cellpose: our masks of the ovarian dataset "
    rf"moved by {shift(ours['dy'], ours['dx'])} locations, the dip under our registration. Xenium: the nuclei imaged by Xenium "
    r"on the same section as the Visium HD run, in four cores of one slide.",
    "tab:s5", ["Case", "Nuclei from", "Nuclei", "Against", "In place", "Background", "In place $-$ background"],
    body, "llrrrrr", size=r"\scriptsize", colsep="4pt"))
# --- S5: whole-slide and regional offsets under each registration: the dip and the correlation ratio ----------------
NEW = HERE.parents[1] / "paper_results" / "registration_local_dip_20261106"
SEARCH, DRAWS, STABLE, DIVISION, MIN_NUCLEUS = 12, 200, 0.95, 5, 30000
sys.path.insert(0, str(HERE.parent / "figures" / "scripts"))
import registration_ring as RR  # noqa: E402
import registration_ratio as CR  # noqa: E402
need(SET["regcrBaselineLoc"] == str(CR.BASELINE), "baseline differs from registration_numbers.tex")
CR_DEF = (r"Correlation ratio: the capture locations are grouped by their distance from the edge of the nearest nucleus, in bands one location wide "
          rf"({SET['regcrBandsIn']} inside the nuclei, {SET['regcrBandsOut']} outside them, and the rest of the tissue); each count is divided by the mean count of the capture "
          rf"locations in tissue in the {SET['regcrBaselineSide']} $\times$ {SET['regcrBaselineSide']} square around it; the shift is the highest local maximum of the "
          rf"correlation ratio $\eta^2$ of the divided counts on the groups within {SEARCH} locations, higher than at its 8 neighbouring shifts (Supplemental Methods). ")
USABLE_DEF = (rf"Usable: not on the edge of the search, and in at least {100 * STABLE:.0f}\% of {DRAWS} draws of spatial blocks of the slide the shift found "
              r"lies within one location of the full-data one. ")
# S6 and S12 refer to the definitions given in full in S5
CR_USABLE_REF = r"Correlation ratio and Usable: as in Supplemental Table~S8. "


def placement_rows(name):
    base = CR.placements_dir(name)   # SPATCH masks moved by the ratio's own fit and the Cellpose-SAM placements: their own records
    meta = jload(base / f"{name}.meta.json")
    need(meta["all_reproduced"] and meta["inputs_sha_ok"] and meta["search"] == SEARCH and meta["draws"] == DRAWS,
         f"{name}: placement record is not a reproduced run of the final rule")
    return jload(base / f"{name}.rows.json")


def whole_row(name):
    row = [r for r in placement_rows(name) if r["surface"] == "whole"][0]
    need(row["tested"] and row["dip"] and row["interior"] and row["usable"], f"S5 {name}: whole-slide dip is not usable")
    return row


def regions_of(name):
    regions = [r for r in placement_rows(name) if r["surface"] != "whole" and not r["surface"].startswith("3x3_")]
    need(len(regions) == DIVISION ** 2, "S5: not a full division")
    usable = [r for r in regions if r["usable"]]
    need(all(r["interior"] and r["share_within1"] >= STABLE and r["nucleus_in_tissue"] >= MIN_NUCLEUS for r in usable), "S5: usable without the rule")
    return regions


def dip_cells(name):
    w = whole_row(name)
    return [shift(w["dy"], w["dx"]), f"{100 * w['depth']:.1f}"]


def ratio_cells(name):
    """Whole slide: the ratio's shift and offset; regions: dip usable; ratio usable, at zero, largest; both usable: equal, one off."""
    rows_, _ = CR.placement(name)
    found, usable_, sh, _ = CR.ratio(rows_["whole"])
    need(usable_, f"S5 {name}: whole-slide correlation ratio not usable")
    own = {r["surface"]: r for r in regions_of(name)}
    regs = [g for g in CR.tested(rows_, whole=False)]
    need(all(own[g["surface"]]["nucleus_in_tissue"] >= MIN_NUCLEUS for g in regs) and
         sum(r["nucleus_in_tissue"] >= MIN_NUCLEUS for r in own.values()) == len(regs), f"S5 {name}: tested regions differ")
    usable = [g for g in regs if CR.ratio(g)[1]]
    equal = one = 0
    for g in regs:
        d = own[g["surface"]]
        if d["usable"] and CR.ratio(g)[1]:
            c = CR.cheb((d["dy"], d["dx"]), CR.ratio(g)[2])
            need(c <= 1, f"S5 {name} {g['surface']}: dip and correlation ratio differ by 2 or more locations")
            equal += c == 0
            one += c == 1
    largest = max(distance_um(*CR.ratio(g)[2]) for g in usable)
    return ([shift(*sh), f"{distance_um(*sh):.1f}"],
            [str(sum(r["usable"] for r in own.values())), str(len(usable)), str(sum(CR.ratio(g)[2] == (0, 0) for g in usable)), f"{largest:.1f}"],
            [str(equal), str(one)])


def group(title, members):
    return title, [(nuclei, registration, dip_cells(name), *ratio_cells(name)) for nuclei, registration, name in members]


groups = [
    group(f"Ovarian dataset, H1-YQJBZ7X/D1", [
        ("Cellpose", "Ours, from the images", "ov_cellpose_img"),
        ("StarDist", "Ours, from the images", "ov_stardist_img"),
        ("Cellpose", "Space Ranger 4.0.1, deposited", "ov_cellpose_dep"),
        ("StarDist", "Space Ranger 4.0.1, deposited", "ov_stardist_dep")]),
    group(f"Colorectal slide, H1-GMHFWPH/D1", [
        ("Cellpose", "Ours, from the images", "crc_cellpose_img"),
        ("StarDist", "Ours, from the images", "crc_stardist_img"),
        ("Cellpose", "Space Ranger 4.1.0, deposited", "crc_cellpose_dep"),
        ("StarDist", "Space Ranger 4.1.0, deposited", "crc_stardist_dep")]),
    group(f"SPATCH slide, H1-WP2FTZD/D1", [
        ("Cellpose", "Ours, from the images", "spatch_before"),
        ("Cellpose", "Ours, with our fitted correction", "spatch_after_ratio"),
        ("Cellpose", "Space Ranger 3.0.0, our run", "spatch_sr3"),
        ("Cellpose", "Space Ranger 4.1.0, our run", "spatch_sr410")]),
    group("Colorectal slide H1-VM2JXXK/A1", [
        ("Cellpose", "Space Ranger 3.0.0, deposited in GEO", "a1_cellpose_sr3"),
        ("StarDist", "Space Ranger 3.0.0, deposited in GEO", "a1_sr3"),
        ("Cellpose", "Space Ranger 4.1.0, our run", "a1_cellpose_sr410"),
        ("StarDist", "Space Ranger 4.1.0, our run", "a1_sr410")])]
# The values the main text prints must agree with these rows.
for name, value in (("regcrSpatchSRthreeUm", groups[2][1][2][3][1]), ("regcrSpatchSRfourUm", groups[2][1][3][3][1]),
                    ("regcrSpatchBeforeUsable", groups[2][1][0][4][1]), ("regcrSpatchAfterUsable", groups[2][1][1][4][1]),
                    ("regcrSpatchAfterZero", groups[2][1][1][4][2]), ("regcrSpatchSRthreeUsable", groups[2][1][2][4][1]),
                    ("regcrSpatchSRthreeZero", groups[2][1][2][4][2]), ("regcrSpatchSRthreeMaxUm", groups[2][1][2][4][3]),
                    ("regcrSpatchSRfourUsable", groups[2][1][3][4][1]), ("regcrSpatchSRfourZero", groups[2][1][3][4][2]),
                    ("regcrSpatchSRfourMaxUm", groups[2][1][3][4][3]), ("regcrSpatchBeforeMaxUm", groups[2][1][0][4][3])):
    need(SET[name] == value, f"S5: {name} is {SET[name]} in the text and {value} here")
need(f"{distance_um(3, 3):.1f}" == groups[0][1][0][3][1], "S5: ovarian offset under our registration")
need(SET["regchkSearchLoc"] == str(SEARCH) and SET["regchkStableDraws"] == str(DRAWS), "S5: settings differ from registration_numbers.tex")
equal_all = sum(int(m[5][0]) for _, ms in groups for m in ms)
body = []
for title, members in groups:
    if body:
        body.append(r"\addlinespace[3pt]")
    body.append(rf"\multicolumn{{12}}{{l}}{{\textit{{{title}}}}} \\")
    for nuclei, registration, dip, whole_cr, regions, agree in members:
        body.append([nuclei, registration] + dip + whole_cr + regions + agree)
blocks.append(table(
    r"\textbf{Supplemental Table S8. Whole-slide and regional offsets under each registration: the dip and the correlation ratio.} Our nucleus masks were placed on the capture "
    r"grid by each registration and moved over it one capture location at a time. Shifts are (rows, columns) of capture locations of 2\,\textmu m, rows running down "
    rf"the slide. Dip: the position of the deepest dip within {SEARCH} locations; Depth: one minus the mean count at the dip divided by the median over its 8 neighbouring "
    r"shifts. " + CR_DEF + r"Offset: the distance of the shift from zero. "
    r"Cellpose: nuclei segmented with Cellpose-SAM, as \STARgrid{} does; StarDist: nuclei segmented with StarDist, the control on the ovarian dataset and the two colorectal slides. "
    rf"Regions: the capture area divided {DIVISION} $\times$ {DIVISION}; a region is measured when it holds at least {MIN_NUCLEUS:,} nucleus locations in tissue. "
    + USABLE_DEF + r"At zero: usable regions whose shift is zero. Largest: the largest regional offset. Both usable: of the regions in which the dip and the correlation ratio "
    r"are both usable, those where they give the same shift (Equal) and shifts one location apart (One off); in none do they differ by more. "
    r"Deposited: the registration in the output deposited with "
    rf"the dataset; our run: our Space Ranger run on a subsample of {SET['regchkSubsampleReads']} reads (Supplemental Methods). For each set of nuclei on a slide, the "
    r"masks and the count image are the same in every row; only the registration changes. Every whole-slide dip and correlation ratio is usable, and on every whole slide the two give the same shift.",
    "tab:s6", ["Nuclei", "Registration", "Shift", "Depth (\\%)", "Shift", "Offset (\\textmu m)", "Dip", "Ratio", "At zero",
               "Largest (\\textmu m)", "Equal", "One off"],
    body, "llrrrrrrrrrr", size=r"\scriptsize", colsep="3pt",
    above=(r" & & \multicolumn{2}{c}{Whole slide, dip} & \multicolumn{2}{c}{Whole slide, ratio} & \multicolumn{4}{c}{Regions usable} & \multicolumn{2}{c}{Both usable} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-10}\cmidrule(l){11-12}")))
for _, ms in groups:
    for m in ms:
        need(m[2][0] == m[3][0], f"S5 {m[0]} {m[1]}: whole-slide dip and ratio differ")

# --- S6: lung tissue, same section ------------------------------------------------------------------
lung_s = jload(C / "lung_same_section.json")
need(lung_s["criteria_written_after_whole_core_surfaces"] is True, "S6: chronology of the lung criteria not recorded")
need(SET["regchkLungDivision"] == "3", "S6: lung division differs from registration_numbers.tex")


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def mark(found, usable_, sh, share):
    """A shift with its share of draws within one location; not usable in italics."""
    if not found:
        return ["none", "---"]
    text = shift(*sh)
    return [text if usable_ else rf"\textit{{{text}}}", f"{100 * share:.1f}"]


def lung_cells(ring_g, cr_g):
    d = ring_g["f0"]["dip"]
    du = bool(d["found"] and d["usable"])
    need(not d["found"] or du == (d["interior"] and d["share_within1"] >= STABLE), "S6: dip usable without the rule")
    f, u, sh, p = CR.ratio(cr_g)
    need(not f or u == (p["interior"] and p["share_within1"] >= STABLE), "S6: ratio usable without the rule")
    return (mark(d["found"], du, (d["dy"], d["dx"]) if d["found"] else None, d.get("share_within1")) +
            mark(f, u, sh, p.get("share_within1")))


def regional_cells(ring_rows, cr_rows, truth_of):
    """Regions: measured; dip usable at the truth and one off; ratio usable at the truth, one off, two or more off."""
    regs = [g for k, g in cr_rows.items() if k != "whole" and g.get("tested")]
    d = [0, 0]
    r = [0, 0, 0]
    for g in regs:
        truth = truth_of(g)
        _, du, ds = RR.dip(ring_rows[g["surface"]])
        if du:
            c = RR.cheb(ds, truth)
            need(c <= 1, "S6: a usable dip lies 2 or more locations from the truth")
            d[c] += 1
        _, ru, rs, _ = CR.ratio(g)
        if ru:
            r[min(CR.cheb(rs, truth), 2)] += 1
    return [str(len(regs))] + [str(v) for v in d + r]


def section_name(rel):
    return Path(rel).name[: -len(".rows.json")]


CP_TRUTH = jload(HERE.parents[1] / "paper_results" / "registration_cellpose_sam_lung_20261110" / "truth" / "truth_table.json")
SAME = r"\multicolumn{4}{c}{Xenium nuclei: as in the row above}"          # the Xenium nuclei are the same in both rows of a tissue
SAME_REG = r"\multicolumn{4}{c}{as above}"
body, regional = [], []


def two_rows(name, cp_he, sd_he, xen):
    """Rows of one tissue: Cellpose-SAM H&E nuclei (with the Xenium nuclei) and StarDist H&E nuclei (the control).
    Each argument is (ring rows, ratio rows, truth of the whole core or section)."""
    for label, (he_ring, he_cr, t), last in ((name + ", Cellpose-SAM", cp_he, False), (name + ", StarDist (control)", sd_he, True)):
        xen_ring, xen_cr = xen
        need(tuple(he_cr["whole"]["truth"]) == tuple(t), f"S6 {label}: truth differs from the mask displacement")
        body.append([label, shift(*t), f"{distance_um(*t):.1f}"] + lung_cells(he_ring["whole"], he_cr["whole"])
                    + ([SAME] if last else lung_cells(xen_ring["whole"], xen_cr["whole"])))
        xr = regional_cells(xen_ring, xen_cr, lambda g: (0, 0))
        regional.append([label] + regional_cells(he_ring, he_cr, lambda g: tuple(g["truth"])) + ([SAME_REG] if last else xr[0:1] + xr[3:]))


for core, c in lung_s["criteria"].items():
    xen = (RR.placement(f"lung_xenium_{core}")[0], CR.placement(f"lung_xenium_{core}")[0])
    need(tuple(CP_TRUTH["vannan"][core]) == tuple(CR.placement(f"lung_cpsam_{core}")[0]["whole"]["truth"]), f"S6 {core}: Cellpose-SAM truth differs from truth_table.json")
    two_rows(core, (RR.placement(f"lung_cpsam_{core}")[0], CR.placement(f"lung_cpsam_{core}")[0], CP_TRUTH["vannan"][core]),
             (RR.placement(f"lung_he_{core}")[0], CR.placement(f"lung_he_{core}")[0], (c["mask_dy"], c["mask_dx"])), xen)
for sec in RR.sections():
    cp = RR.sections(cpsam=True)[[x["id"] for x in RR.sections()].index(sec["id"])]
    xen_ring, _ = RR.rows_file(sec["xen_rows"])
    xen_cr, _ = CR.placement(section_name(sec["xen_rows"]))
    need(tuple(xen_cr["whole"]["truth"]) == (0, 0), f"S6 {sec['id']}: Xenium truth is not zero")
    he_cr = CR.placement(section_name(sec["he_rows"]))[0]
    cp_cr = CR.placement(section_name(cp["he_rows"]))[0]
    two_rows(sec["id"], (RR.rows_file(cp["he_rows"])[0], cp_cr, cp_cr["whole"]["truth"]),
             (RR.rows_file(sec["he_rows"])[0], he_cr, he_cr["whole"]["truth"]), (xen_ring, xen_cr))
# The totals of the main text: shifts, at the truth, one off, two or more off, over the whole cores and sections and their regions.
lung_tot = [0, 0, 0, 0]
for core, c in lung_s["criteria"].items():
    for kind, truth in (("he", (c["mask_dy"], c["mask_dx"])), ("cpsam", tuple(CP_TRUTH["vannan"][core])), ("xenium", (0, 0))):
        for g in CR.tested(CR.placement(f"lung_{kind}_{core}")[0]):
            _, u, sh, _ = CR.ratio(g)
            lung_tot[0] += 1
            if u:
                lung_tot[1 + min(CR.cheb(sh, tuple(g["truth"])), 2)] += 1
for sec in RR.sections() + [dict(x, xen_rows=None) for x in RR.sections(cpsam=True)]:
    for rel in (sec["he_rows"], sec["xen_rows"]):
        if rel is None:
            continue
        for g in CR.tested(CR.placement(section_name(rel))[0]):
            _, u, sh, _ = CR.ratio(g)
            lung_tot[0] += 1
            if u:
                lung_tot[1 + min(CR.cheb(sh, tuple(g["truth"])), 2)] += 1
need([str(v) for v in lung_tot] == [SET["regcrLungSurfaces"], SET["regcrLungExact"], SET["regcrLungOneOff"], SET["regcrLungTwoOff"]],
     f"S6: lung totals {lung_tot} differ from registration_numbers.tex")
LUNG_SPEC = "lrr" + "rrrr" + "rrrr"


def tabular(header, rows_, spec, above=()):
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *above, " & ".join(header) + r" \\", r"\midrule"]
    out += [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in rows_]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])


s6_caption = (
    r"\textbf{Supplemental Table S11. Lung tissue measured by Xenium and then Visium HD on the same section.} Tissue: the four cores of one slide (H1-K3WKZHN/D1) "
    rf"and the whole sections of {NUMBER_WORDS[len(RR.sections())]} further slides (identifiers given). "
    r"Each tissue has two rows for the nuclei segmented from the H\&E image: Cellpose-SAM, as \STARgrid{} does, and StarDist, the control; the Xenium nuclei are the same in both rows and their results are given in the first. "
    r"Registration: Space Ranger 3.0.0 in all of them; "
    r"by hand for the four cores"
    + "".join(f" and, in Loupe Browser 8.0, for {x['id']}" for x in RR.sections() if "by hand" in x["registration"])
    + r". Mask displacement: the "
    r"shift that brings the nuclei segmented from the H\&E image onto the nuclei imaged by Xenium, measured from the two masks "
    rf"alone, without counts, for the segmentation of the row; this is the truth for the H\&E nuclei of the row, and zero is the truth for the Xenium nuclei. Dip: the deepest dip within {SEARCH} locations in the counts "
    r"deposited with the dataset. " + CR_USABLE_REF + r"Within one: the percentage of draws of spatial blocks in which the shift found lies within one location of the full-data one. "
    r"Shifts not usable are in italics. Shifts are (rows, "
    r"columns) in capture locations of 2\,\textmu m, rows running down the slide. "
    r"Lower part, regions (the 3 $\times$ 3 division of each core, the 5 $\times$ 5 division of a section; regions holding at least "
    rf"{MIN_NUCLEUS:,} nucleus locations in tissue): the truth of a region is the mask displacement of that region for the H\&E nuclei and zero for the Xenium nuclei. "
    r"Usable dips and usable ratios: regions where the dip or the correlation ratio is usable, by the distance of its shift from the truth. No usable dip lies two or more locations from its truth.")
s6_upper = tabular(["Tissue", "Shift", "\\textmu m", "Shift", "Within one (\\%)", "Shift", "Within one (\\%)",
                    "Shift", "Within one (\\%)", "Shift", "Within one (\\%)"], body, "lrr" + "rrrr" + "rrrr",
                   above=(r" & \multicolumn{2}{c}{Mask displacement} & \multicolumn{2}{c}{H\&E nuclei, dip} & \multicolumn{2}{c}{H\&E nuclei, ratio} & "
                          r"\multicolumn{2}{c}{Xenium nuclei, dip} & \multicolumn{2}{c}{Xenium nuclei, ratio} \\",
                          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}"))
s6_lower = tabular(["Tissue", "Regions", "At truth", "One off", "At truth", "One off", "Two off", "Regions", "At truth", "One off", "Two off"], regional,
                   "lrrrrrrrrrr",
                   above=(r" & \multicolumn{6}{c}{H\&E nuclei} & \multicolumn{4}{c}{Xenium nuclei} \\",
                          r"\cmidrule(lr){2-7}\cmidrule(l){8-11}",
                          r" & & \multicolumn{2}{c}{Usable dips} & \multicolumn{3}{c}{Usable ratios} & & \multicolumn{3}{c}{Usable ratios} \\",
                          r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(l){9-11}"))
blocks.append("\n".join([r"\begin{table}[H]", r"\centering\scriptsize", rf"\caption{{{s6_caption}}}", r"\label{tab:s7}",
                         r"\setlength{\tabcolsep}{2.5pt}", s6_upper, r"\medskip", s6_lower, r"\end{table}"]))

# --- S7: sources of the additional molecules (B1 with v1.12.2; A1 on the probe set's genes) ----------------
B1_SPLITS = {"examined": HERE.parents[1] / "paper_results" / "b1_location_effect_target_20261109" / "results.json",
             "ovarian": HERE.parents[1] / "paper_results" / "b1_location_effect_ovarian_v1_12_2_20261109" / "results.json"}
body = []
for slide, label in (("examined", "Colorectal (Flex), H1-VM2JXXK/A1"), ("ovarian", "Ovarian dataset (RNA-seq), H1-YQJBZ7X/D1")):
    record = json.loads(B1_SPLITS[slide].read_text())
    need(record["status"] == "complete" and record["all_gates_passed"] is True
         and record["release_revision"] == "2ca281c96f3dd2b9d596da1991b397fd05f9f73d", f"S7 {slide}: not a complete v1.12.2 record")
    split = record[slide]
    m, (b, h, t) = split["M"], (split["primary"][q] for q in ("B", "H", "T"))
    need(split["D"] == t - m == (h - b) + (t - h) + (b - m), f"S7 {slide}: split does not close")
    share = lambda v: f"{100 * v / (t - m):+.1f}".replace("-", "$-$")
    body.append([label, n(m), n(t), n(b), n(h)])
    body.append([r"\quad molecules", signed_n(t - m), signed_n(h - b), signed_n(t - h), signed_n(b - m)])
    body.append([r"\quad \% of $D$", "100", share(h - b), share(t - h), share(b - m)])
    if slide == "examined":
        body.append(r"\addlinespace[3pt]")
blocks.append(table(
    r"\textbf{Supplemental Table S6. Where the molecules STARgrid counts beyond Space Ranger come from, on two slides.} $M$: the molecules that Space Ranger counted "
    r"(Space Ranger 4.0.1 on slide H1-VM2JXXK/A1, on the genes of its probe set; 4.1.0 with introns included on the ovarian dataset). $T$: STARgrid's total. "
    r"$B$ and $H$ count the reads that Space Ranger mapped to a single gene and gave a UMI, with Space Ranger's gene and the raw "
    r"UMI, at Space Ranger's locations ($B$) or at the locations STARgrid chooses ($H$), under the UMI correction of "
    r"the matrices. Columns of the lower rows: $D=T-M$; placement $H-B$; gene assignment $T-H$ (STARgrid's genes, selection "
    r"of reads and handling of UMIs with reads for two genes); deduplication $B-M$. The three components add up to $D$.",
    "tab:s8", ["", "$M$ / $D$", "$T$ / $H-B$", "$B$ / $T-H$", "$H$ / $B-M$"],
    body, "lrrrr", size=r"\scriptsize", colsep="5pt"))

# --- S12: the dip and the correlation ratio on four public Space Ranger 4.0.1 tissues ---------------------------------------
PUB = NEW / "public_tissues"
pub = rows(PUB / "rule_results.tsv")
edge = rows(PUB / "edge_profiles.tsv")
body = []
pub_counts = [0, 0, 0]
for ds in list(dict.fromkeys(r["dataset"] for r in pub)):
    rs = [r for r in pub if r["dataset"] == ds]
    whole_r = [r for r in rs if r["surface"] == "whole"]
    need(len(whole_r) == 1, f"S12 {ds}: whole-slide row")
    w = whole_r[0]
    regs = [r for r in rs if r["surface"] != "whole"]
    usable = [r for r in regs if r["usable"] == "True"]
    need(all(r["interior"] == "True" and float(r["share_within_one"]) >= STABLE - 1e-12 and int(r["nucleus_locations"]) >= MIN_NUCLEUS for r in usable),
         f"S12 {ds}: usable without the rule")
    eds = [e for e in edge if e["dataset"] == ds]
    inside = [e for e in eds if e["bin"].startswith("nucleus,")]
    outside = [e for e in eds if e["bin"] == "outside nucleus, 0-2 um"]
    need(len(inside) == 3 and len(outside) == 1, f"S12 {ds}: edge profile bins")
    m_in = sum(float(e["mean_count"]) * int(e["locations"]) for e in inside) / sum(int(e["locations"]) for e in inside)
    m_out = float(outside[0]["mean_count"])
    has = w["dip_dy"] != ""
    ring_rows, _ = RR.placement(f"public_{ds}")
    cr_rows, _ = CR.placement(f"public_{ds}")
    need(RR.dip(ring_rows["whole"])[0] == has and (not has or RR.dip(ring_rows["whole"])[2] == (int(w["dip_dy"]), int(w["dip_dx"]))),
         f"S12 {ds}: ring-run dip differs from the rule results")
    need(has or m_in > m_out, f"S12 {ds}: slide without a dip whose nuclei hold fewer counts than just outside")
    f, u, sh, _ = CR.ratio(cr_rows["whole"])
    need(u and CR.cheb(sh, (0, 0)) <= 1, f"S12 {ds}: whole-slide ratio not usable within one location of zero")
    tested_regs = CR.tested(cr_rows, whole=False)
    need(len(tested_regs) == len(regs), f"S12 {ds}: regions differ between the records")
    ok = [g for g in tested_regs if CR.ratio(g)[1]]
    offs = [CR.cheb(CR.ratio(g)[2], (0, 0)) for g in ok]
    need(max(offs, default=0) <= 1, f"S12 {ds}: a usable regional ratio two or more locations from zero")
    nodip = [g for g in tested_regs if not RR.dip(ring_rows[g["surface"]])[0]]
    pub_counts[0] += 1 + len(tested_regs)
    pub_counts[1] += 1 + len(ok)
    pub_counts[2] += (sh == (0, 0)) + offs.count(0)
    body.append([w["tissue"], shift(w["dip_dy"], w["dip_dx"]) if has else "no dip", f"{100 * float(w['depth']):.1f}" if has else "---",
                 "yes" if w["usable"] == "True" else "no", shift(*sh), f"{len(usable)} of {len(regs)}",
                 f"{len(ok)} of {len(tested_regs)}", str(offs.count(0)), str(offs.count(1)),
                 f"{sum(CR.ratio(g)[1] for g in nodip)} of {len(nodip)}" if nodip else "---", f"{m_in:.1f}", f"{m_out:.1f}"])
need([str(v) for v in pub_counts] == [SET["regcrPublicSurfaces"], SET["regcrPublicShifts"], SET["regcrPublicExact"]],
     f"S12: totals {pub_counts} differ from registration_numbers.tex")
blocks.append(table(
    r"\textbf{Supplemental Table S10. The dip and the correlation ratio on four public Visium HD datasets processed with Space Ranger 4.0.1.} Nuclei: the nucleus "
    r"map deposited with each dataset, placed by the deposited registration; counts: the deposited 2\,\textmu m counts in tissue. Dip: the deepest dip "
    rf"within {SEARCH} locations of the whole slide, in (rows, columns) of capture locations of 2\,\textmu m; depth is one minus the mean count at the dip divided by the median over its 8 neighbouring "
    r"shifts, in \%. " + CR_USABLE_REF +
    rf"Regions: of the {DIVISION} $\times$ {DIVISION} regions holding at least {MIN_NUCLEUS:,} nucleus locations in tissue, those where the dip, or the correlation ratio, is usable; "
    r"At zero and One off: usable regional ratios at zero shift, Space Ranger's registration, and one location from it; none is farther. No dip: of the regions with no dip at all, those where the correlation ratio is usable "
    r"(---, every region has a dip). Inside, Outside: mean count per capture location in tissue inside nuclei (all locations of nuclei) and 0 to 2\,\textmu m outside the nucleus edge. "
    r"Every whole-slide correlation ratio is usable.",
    "tab:s12", ["Tissue", "Shift", "Depth (\\%)", "Usable", "Shift", "Dip", "Ratio", "At zero", "One off", "No dip", "Inside", "Outside"],
    body, "lrrrrrrrrrrr", size=r"\scriptsize", colsep="3pt",
    above=(r" & \multicolumn{3}{c}{Whole slide, dip} & Ratio & \multicolumn{5}{c}{Regions usable} & \multicolumn{2}{c}{Mean count} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-10}\cmidrule(l){11-12}")))

# --- S13: the window of the local mean (the baseline sweep of the correlation ratio) -----------------------------------------
TRUTH_NAMES = [n for n in CR.names() if any(g.get("truth") is not None for g in CR.tested(CR.placement(n)[0]))]
TUMOURS = list(RR.TUMOUR)
need("spatch_after_ratio" in TUMOURS and "spatch_after_new" not in TUMOURS,
     "S13: the tumour set's SPATCH-after placement is not the one S5 shows (spatch_after_ratio)")


def sweep_row(setting):
    lung = [CR.cheb(CR.ratio(g, setting)[2], tuple(g["truth"])) for n in TRUTH_NAMES for g in CR.tested(CR.placement(n)[0]) if CR.ratio(g, setting)[1]]
    tum_n = sum(CR.ratio(g, setting)[1] for n in TUMOURS for g in CR.tested(CR.placement(n)[0]))
    tum = []
    for n in TUMOURS:
        ring_rows, _ = RR.placement(n)
        for g in CR.tested(CR.placement(n)[0]):
            _, du, ds = RR.dip(ring_rows[g["surface"]])
            if du and CR.ratio(g, setting)[1]:
                tum.append(CR.cheb(CR.ratio(g, setting)[2], ds))
    pubo = [CR.cheb(CR.ratio(g, setting)[2], (0, 0)) for n in CR.PUBLIC for g in CR.tested(CR.placement(n)[0]) if CR.ratio(g, setting)[1]]
    return [str(len(lung)), str(lung.count(0)), str(sum(o >= 2 for o in lung)), str(tum_n), str(len(tum)), str(sum(o >= 2 for o in tum)),
            str(len(pubo)), str(sum(o >= 2 for o in pubo))]


lung_total = sum(len(CR.tested(CR.placement(n)[0])) for n in TRUTH_NAMES)
tum_total = sum(len(CR.tested(CR.placement(n)[0])) for n in TUMOURS)
pub_total = sum(len(CR.tested(CR.placement(n)[0])) for n in CR.PUBLIC)
body = [["none", "---"] + sweep_row("raw")]
for b in CR.BOXES:
    cells = sweep_row(f"box{b}")
    if b == CR.BASELINE:
        body.append([rf"\textbf{{{v}}}" for v in [str(2 * b + 1) + r" $\times$ " + str(2 * b + 1), f"{2 * b}"] + cells])
    else:
        body.append([str(2 * b + 1) + r" $\times$ " + str(2 * b + 1), f"{2 * b}"] + cells)
default = sweep_row(CR.SETTING)
need(default[0] == SET["regcrLungShifts"] and default[1] == SET["regcrLungExact"], "S13: the default row differs from registration_numbers.tex")
need(str(lung_total) == SET["regcrLungSurfaces"] and str(pub_total) == SET["regcrPublicSurfaces"], "S13: totals differ from registration_numbers.tex")
blocks.append(table(
    r"\textbf{Supplemental Table S9. The window of the local mean.} The correlation ratio (Supplemental Methods) with each count divided by the mean count of the "
    r"capture locations in tissue in a square window around it, of the side given (in capture locations; half-width in \textmu m), or not divided (none); in bold, the window "
    rf"\STARgrid{{}} uses. Shifts: usable shifts, by the rule of Supplemental Table S8. Lung: the {lung_total} cores, whole sections and regions of Supplemental Table S11, with nuclei from the H\&E image or imaged by "
    r"Xenium; At truth: shifts at the truth; 2+ off: two or more locations from it. Tumour slides: the "
    rf"{tum_total} whole slides and regions of the placements of Supplemental Table S8; With dip: those where the dip is also usable; 2+ from dip: of these, shifts two or more locations from the dip's. "
    rf"Public slides: the {pub_total} whole slides and regions of Supplemental Table S10; 2+ from SR: shifts two or more locations from Space Ranger 4's registration.",
    "tab:s13", ["Window", "Half-width (\\textmu m)", "Shifts", "At truth", "2+ off", "Shifts", "With dip", "2+ from dip", "Shifts", "2+ from SR"],
    body, "lrrrrrrrrr", size=r"\scriptsize", colsep="4pt",
    above=(rf" & & \multicolumn{{3}}{{c}}{{Lung ({lung_total})}} & \multicolumn{{3}}{{c}}{{Tumour slides ({tum_total})}} & \multicolumn{{2}}{{c}}{{Public slides ({pub_total})}} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}")))


def stored(block):
    """Store a table under its number; additional_file_1.tex prints the tables of both files in numeric order."""
    number = re.search(r"Supplemental Table S(\d+)\.", block).group(1)
    block, labels = re.subn(r"\\label\{tab:[^}]*\}", rf"\\label{{supptab:{number}}}", block, count=1)
    need(labels == 1, f"Table S{number}: no label")
    return rf"\supptable{{{number}}}{{%" + "\n" + block + "\n}"


OUT.write_text("% Generated by supplementary/build_additional_file_1.py from the accepted tables. Each table is\n"
               "% stored by \\supptable{number}{...} and printed in numeric order by additional_file_1.tex.\n"
               + "\n\n".join(stored(b) for b in blocks) + "\n")
print("wrote", OUT)
