#!/usr/bin/env python3
"""Export the macros of the registration check from the accepted results.

Method (final, 2026-11-06): a dip is an offset whose mean count under the masks is lower than at all 8 neighbouring
offsets; the masks are moved to the deepest dip within 12 locations; a region or whole slide is usable when its deepest dip
is interior and, in at least 95% of 200 spatial-block draws, lies within one location of the full-data dip (regions also
need at least 30,000 nucleus locations in tissue). Reads paper_results/registration_local_dip_20261106/ (every placement,
whole slide and regions), Space Ranger's metrics files of the registration check of 2026-10-18, and the paper tests of
2026-10-21 (paper_results/paper_tests_20261021/: far-shift control, within-nucleus paired depths, cancer-gene sensitivity
and source provenance), and checks the values that the manuscript already prints from numbers.tex.
Usage: registration_numbers.py > paper/registration_numbers.tex
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_macros  # noqa: E402
import registration_ratio as CR0  # noqa: E402   (where each placement's dip-rule rows are kept)

ROOT = HERE.parents[2]
R = ROOT / "paper_results" / "registration_check_20261018"
NEW = ROOT / "paper_results" / "registration_local_dip_20261106"
RATIO_FIT = ROOT / "paper_results" / "spatch_rescore_ratio_20261109" / "placements"   # SPATCH masks moved by the correlation-ratio fit
T = ROOT / "paper_results" / "paper_tests_20261021"
UM = 2.0                                    # one capture location
SEARCH, DRAWS, SEED, STABLE = 12, 200, 20261007, 0.95
DIVISION, MIN_NUCLEUS = 5, 30000


def load(name, base=R):
    return json.loads((base / name).read_text())


def distance_um(dy, dx):
    return UM * math.hypot(dy, dx)


def require(condition, message):
    if not condition:
        sys.exit(f"registration_numbers.py: {message}")


macros = {}
known = paper_macros.load()

# --- The placements under the dip rule (paper_results/registration_local_dip_20261106/placements) -------------
def place(name):
    """(rows of every surface, meta) of one placement; every surface was reproduced from its inputs."""
    base = CR0.placements_dir(name)
    rows = json.loads((base / f"{name}.rows.json").read_text())
    meta = json.loads((base / f"{name}.meta.json").read_text())
    require(meta["all_reproduced"] and meta["inputs_sha_ok"], f"{name}: surfaces not reproduced from the recorded inputs")
    require(meta["search"] == SEARCH and meta["draws"] == DRAWS and meta["seed"] == SEED, f"{name}: not the final rule's settings")
    return rows, meta


def whole_of(name):
    w = [r for r in place(name)[0] if r["surface"] == "whole"][0]
    require(w["tested"] and w["dip"], f"{name}: no dip on the whole slide")
    return w


def regions_of(name):
    """The 5 x 5 regions of a placement (the 3 x 3 regions of the gate are kept in the record, not used here)."""
    out = [r for r in place(name)[0] if r["surface"] != "whole" and not r["surface"].startswith("3x3_")]
    require(len(out) == DIVISION ** 2, f"{name}: not a {DIVISION} x {DIVISION} division")
    for r in out:
        require(r["usable"] == (r["reason"] == "usable") and
                r["reason"] in ("usable", f"fewer than {MIN_NUCLEUS} nucleus locations in tissue", "dip on the search boundary",
                                "unstable under resampling", "no dip"), f"{name} {r['surface']}: region rule differs: {r['reason']}")
        if r["usable"]:
            require(r["interior"] and r["share_within1"] >= STABLE and r["nucleus_in_tissue"] >= MIN_NUCLEUS, f"{name} {r['surface']}: usable without the rule")
    return out


def usable_of(name):
    return [r for r in regions_of(name) if r["usable"]]


def wdist(r):
    return distance_um(r["dy"], r["dx"])


def zero(r):
    return r["dy"] == 0 and r["dx"] == 0


# The dip: our masks placed with Space Ranger's registration; depth against the 8 neighbouring shifts.
dips = {}
# Colorectal: the Cellpose-SAM masks of the registration (Crc) and the StarDist control (CrcStardist).
for key, name in (("OvCellpose", "ov_cellpose_dep"), ("OvStardist", "ov_stardist_dep"), ("Crc", "crc_cellpose_dep"),
                  ("CrcStardist", "crc_stardist_dep")):
    w = whole_of(name)
    require(zero(w) and w["usable"], f"{name}: dip not usable at zero shift")
    dips[key] = 100 * w["depth"]
    macros[f"regchkDip{key}Pct"] = f"{dips[key]:.1f}"
macros["regchkDipMinPct"] = f"{min(dips.values()):.1f}"
macros["regchkDipMaxPct"] = f"{max(dips.values()):.1f}"

# Our registration on the ovarian and colorectal slides (image only).
for name in ("ov_cellpose_img", "ov_stardist_img"):
    w = whole_of(name)
    require((w["dy"], w["dx"]) == (-3, -3) and w["usable"], f"{name}: dip not at (-3, -3)")
require(f"{distance_um(3, 3):.2f}" == known["regOvShiftUm"], "ovarian shift differs from numbers.tex")
for name in ("crc_cellpose_img", "crc_stardist_img"):
    require(zero(whole_of(name)), f"colorectal {name}: our placement not at zero shift")

# SPATCH: our placement before and after the refitted correction, Space Ranger 3.0.0 and 4.1.0 placements.
macros["regchkSpatchOursUm"] = f"{wdist(whole_of('spatch_before')):.1f}"
after = usable_of("spatch_after_ratio")
require(zero(whole_of("spatch_after_ratio")), "SPATCH: the corrected masks do not lie at zero shift")
macros["regchkSpatchAfterUsable"] = str(len(after))
macros["regchkSpatchAfterZero"] = str(sum(zero(r) for r in after))
macros["regchkSpatchOursUsable"] = str(len(usable_of("spatch_before")))
for key, name in (("SRthree", "spatch_sr3"), ("SRfour", "spatch_sr410")):
    regions = usable_of(name)
    offsets = [wdist(r) for r in regions]
    macros[f"regchkSpatch{key}Um"] = f"{wdist(whole_of(name)):.1f}"
    macros[f"regchkSpatch{key}Usable"] = str(len(regions))
    macros[f"regchkSpatch{key}Zero"] = str(sum(zero(r) for r in regions))
    macros[f"regchkSpatch{key}MinUm"] = f"{min(offsets):.1f}"
    macros[f"regchkSpatch{key}MaxUm"] = f"{max(offsets):.1f}"
require(macros["regchkSpatchSRfourUm"] == "0.0", "SPATCH: Space Ranger 4.1.0 placement not at zero")

# July colorectal slide: GEO deposit (Space Ranger 3.0.0) and a Space Ranger 4.1.0 run, with Cellpose-SAM nuclei (StarDist as the control).
for key, name in (("Geo", "a1_cellpose_sr3"), ("SRfour", "a1_cellpose_sr410"), ("GeoStardist", "a1_sr3"), ("SRfourStardist", "a1_sr410")):
    regions = usable_of(name)
    macros[f"regchkJuly{key}Um"] = f"{wdist(whole_of(name)):.1f}"
    macros[f"regchkJuly{key}Usable"] = str(len(regions))
    macros[f"regchkJuly{key}Zero"] = str(sum(zero(r) for r in regions))
    macros[f"regchkJuly{key}WithinOne"] = str(sum(abs(r["dy"]) <= 1 and abs(r["dx"]) <= 1 for r in regions))
require(macros["regchkJulySRfourUm"] == "0.0" == macros["regchkJulySRfourStardistUm"], "July slide: Space Ranger 4.1.0 placement not at zero")

# Tumour regions usable under both the earlier ring rule and the dip rule, and those where the two give one shift.
both = agree = 0
for name in ("spatch_before", "spatch_after", "spatch_sr3", "spatch_sr410", "a1_sr3", "a1_sr410", "a1_cellpose_sr3", "a1_cellpose_sr410"):
    for r in usable_of(name):
        if r["ring_usable"]:
            both += 1
            agree += (r["ring_dy"], r["ring_dx"]) == (r["dy"], r["dx"])
macros["regchkRingBothRegions"] = str(both)
macros["regchkRingAgreeRegions"] = str(agree)

# Lung, same section: H&E masks against Xenium masks (no counts), and the dip of each nucleus map.
lung = load("lung_same_section.json")["criteria"]
# The H&E nuclei of the lung sets are segmented with Cellpose-SAM (lung_cpsam_<core>, postxen_cpsam, expt2_cpsam; record
# registration_cellpose_sam_lung_20261110) and, as the control, with StarDist (lung_he_<core>, postxen_he, expt2_he). The lung macros
# count both; the truth of each H&E map is its own mask displacement against the Xenium nuclei (truth/truth_table.json of that record).
LUNG_CP_ROOT = ROOT / "paper_results" / "registration_cellpose_sam_lung_20261110"
cp_truth = json.loads((LUNG_CP_ROOT / "truth" / "truth_table.json").read_text())
lung_cp = {core: tuple(cp_truth["vannan"][core]) for core in lung}
shifts = [distance_um(c["mask_dy"], c["mask_dx"]) for c in lung.values()] + [distance_um(*t) for t in lung_cp.values()]
macros["regchkLungMinUm"] = f"{min(shifts):.1f}"
macros["regchkLungMaxUm"] = f"{max(shifts):.1f}"
macros["regchkLungCores"] = str(len(lung))
he = {core: whole_of(f"lung_he_{core}") for core in lung}
he_cp = {core: [r for r in place(f"lung_cpsam_{core}")[0] if r["surface"] == "whole"][0] for core in lung}   # may have no dip or an unstable one: reported
xen = {core: [r for r in place(f"lung_xenium_{core}")[0] if r["surface"] == "whole"][0] for core in lung}
for core in lung:
    require(he[core]["usable"], f"lung {core}: H&E dip not usable")
    if not (he_cp[core]["tested"] and he_cp[core]["dip"] and he_cp[core]["usable"]):     # reported, not repaired (record README)
        print(f"registration_numbers.py: REPORTED (not repaired): lung {core} Cellpose-SAM whole core: dip "
              f"{(he_cp[core].get('dy'), he_cp[core].get('dx')) if he_cp[core].get('dip') else 'none'}, usable={he_cp[core]['usable']}", file=sys.stderr)
match_sd = sum((he[c]["dy"], he[c]["dx"]) == (lung[c]["mask_dy"], lung[c]["mask_dx"]) for c in lung)
match_cp = sum(bool(he_cp[c].get("dip")) and (he_cp[c]["dy"], he_cp[c]["dx"]) == lung_cp[c] for c in lung)
macros["regchkLungMatch"] = str(match_sd + match_cp)
macros["regchkLungMatchStardist"] = str(match_sd)
macros["regchkLungMatchCpsam"] = str(match_cp)
require(match_sd == sum(c["N2"] == "met" for c in lung.values()), "lung: dip and mask displacement count")

existing = list(macros)                     # the macros above are already in the manuscript



# --- Helpers for the additions of 2026-10-24 ----------------------------------------------------
def tsv_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def tsv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def interval(text):
    low, high = json.loads(text)
    return float(low), float(high)


def pp(fraction, places=2):
    """A fraction as percentage points; a value that rounds to zero prints without a sign."""
    text = f"{100 * fraction:.{places}f}"
    return text.lstrip("-") if float(text) == 0 else text


def pct(fraction, places=1):
    return f"{100 * fraction:.{places}f}"


def sig(p, figures=3):
    return f"{p:.{figures}g}"


def count(n):
    return f"{int(n):,}"


def chebyshev_ring(radius):
    return sum(max(abs(dy), abs(dx)) == radius for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1))


MANIFEST = {row["sha256"]: row["file"] for row in tsv(T / "provenance_MANIFEST.tsv")}


def archived(record_sources, script):
    """The hash a record gives for a script, required to be in the source archive."""
    hashes = {sha for path, sha in record_sources.items() if path.endswith("/" + script)}
    require(len(hashes) == 1, f"{script}: not recorded exactly once")
    sha = hashes.pop()
    require(sha in MANIFEST and MANIFEST[sha].endswith("/" + script), f"{script}: recorded copy not archived")
    return sha



# --- Dip depth on every slide with a version-4 placement ----------------------------------------
# (slide, placement, whether the registration was deposited with the dataset or is our run)
SR_FOUR = (("ovarian", "ov_cellpose_dep", "deposited"),
           ("ovarian", "ov_stardist_dep", "deposited"),
           ("colorectal", "crc_cellpose_dep", "deposited"),
           ("colorectal", "crc_stardist_dep", "deposited"),
           ("spatch", "spatch_sr410", "our run"),
           ("examined", "a1_cellpose_sr410", "our run"),
           ("examined", "a1_sr410", "our run"))
four = {}
for slide, name, origin in SR_FOUR:
    w = whole_of(name)
    require(zero(w) and w["usable"] and w["interior"], f"{name}: version-4 dip not usable at zero shift")
    four[name] = 100 * w["depth"]
macros["regchkDipSpatchPct"] = f"{four['spatch_sr410']:.1f}"
macros["regchkDipJulyPct"] = f"{four['a1_cellpose_sr410']:.1f}"
macros["regchkDipJulyStardistPct"] = f"{four['a1_sr410']:.1f}"
macros["regchkDipSRfourMinPct"] = f"{min(four.values()):.1f}"
macros["regchkDipSRfourMaxPct"] = f"{max(four.values()):.1f}"
macros["regchkSRfourSlides"] = str(len({s for s, _, _ in SR_FOUR}))
macros["regchkSRfourDeposited"] = str(len({s for s, _, o in SR_FOUR if o == "deposited"}))
macros["regchkSRfourOurRuns"] = str(len({s for s, _, o in SR_FOUR if o == "our run"}))
require(f"{dips['OvCellpose']:.1f}" == f"{four['ov_cellpose_dep']:.1f}", "ovarian dip changed")

# Size of the subsample behind our Space Ranger runs, from Space Ranger's own metrics file of each run.
subsample = set()
for name in ("spatch_sr410", "spatch_sr300", "july_colorectal_sr410"):
    subsample |= {int(r["Number of Reads"]) for r in tsv_csv(R / "subsample_metrics" / f"{name}_metrics_summary.csv")}
require(len(subsample) == 1, f"subsampled Space Ranger runs differ in size: {sorted(subsample)}")
macros["regchkSubsampleReads"] = f"{subsample.pop():,}"

# --- Settings of the check ---------------------------------------------------------------------
# The ring survives only as a description: the median over shifts at Chebyshev distance 3 to 5 from the dip.
RING = (3, 5)
macros["regchkRingLo"], macros["regchkRingHi"] = str(RING[0]), str(RING[1])
macros["regchkRingLoUm"], macros["regchkRingHiUm"] = f"{UM * RING[0]:.0f}", f"{UM * RING[1]:.0f}"

# The search, the block draws and the stability threshold are the settings of every record of the new folder.
macros["regchkSearchLoc"] = str(SEARCH)
macros["regchkStableDraws"] = str(DRAWS)
macros["regchkStablePct"] = f"{100 * STABLE:.0f}"
# Surfaces are drawn out to 8 locations each way in Figs. 4 and S1 (display only; the search is 12).
macros["regchkSurfaceHalfWidthUm"] = f"{UM * 8:.0f}"
macros["regchkDivision"] = str(DIVISION)
macros["regchkRegions"] = str(DIVISION ** 2)
macros["regchkRegionMinNucleusLocations"] = count(MIN_NUCLEUS)
macros["regchkLocalNeighbours"] = str(chebyshev_ring(1))
require(int(macros["regchkLocalNeighbours"]) == 8, "a dip is compared with its 8 neighbouring shifts")

# --- Lung, same section: depth of the dips of the two nucleus maps -------------------------------
macros["regchkLungXeniumZero"] = str(sum(r["usable"] and zero(r) for r in xen.values() if r.get("dip")))
depths = {"he": [100 * r["depth"] for r in list(he.values()) + list(he_cp.values()) if r.get("dip")], "xenium": [100 * r["depth"] for r in xen.values() if r.get("dip")]}
for r in list(he.values()) + [r for r in he_cp.values() if r.get("dip")] + [r for r in xen.values() if r.get("dip")]:
    require(r["interior"], f"lung {r['set']}: dip on the edge of the search")
macros["regchkLungHeDepthMinPct"] = f"{min(depths['he']):.1f}"
macros["regchkLungHeDepthMaxPct"] = f"{max(depths['he']):.1f}"
macros["regchkLungXeniumDepthMinPct"] = f"{min(depths['xenium']):.1f}"
macros["regchkLungXeniumDepthMaxPct"] = f"{max(depths['xenium']):.1f}"
macros["regchkLungDepthMinPct"] = f"{min(depths['he'] + depths['xenium']):.1f}"
macros["regchkLungDepthMaxPct"] = f"{max(depths['he'] + depths['xenium']):.1f}"

# --- Public tissues (Space Ranger 4.0.1 datasets), from the record of the rule's run ---------------
pub_rows = tsv(NEW / "public_tissues" / "rule_results.tsv")
macros["pubTissues"] = str(len({r["dataset"] for r in pub_rows}))

# --- The ring (used where a dip is present but not usable) and the whole post-Xenium sections ---------------------
# Ring: capture locations within WIDTH locations of a nucleus location that are not nucleus; STARgrid moves the nuclei
# to the highest stable peak of the mean count under the ring where a dip is present but not usable (stargrid-register
# 0.5.0; 0.4.0 also used it where there is no dip, which changes no count below). Width 1 (2 um) was the
# pre-declared primary; width 3 (6 um) was chosen after the lung results were seen (README of registration_ring_postxenium_20261106).
import registration_ring as RR  # noqa: E402

macros["regchkFringeWidthLoc"] = str(RR.WIDTH)
macros["regchkFringeWidthUm"] = f"{UM * RR.WIDTH:.0f}"
macros["regchkFringeFirstWidthUm"] = f"{UM * RR.FIRST_WIDTH:.0f}"
require((macros["regchkFringeWidthLoc"], macros["regchkFringeWidthUm"], macros["regchkFringeFirstWidthUm"]) == ("3", "6", "2"),
        "ring widths: expected 3 locations (6 um), first width 2 um")

# The ring run's nuclei rows are this record's dips: every measured surface agrees with the dip rule's rows.
LUNG_NAMES = [f"lung_{k}_{c}" for c in lung for k in ("he", "cpsam", "xenium")]
compared = 0
for name in list(RR.TUMOUR) + LUNG_NAMES:
    ring_rows, ring_meta = RR.placement(name)
    own = {r["surface"]: r for r in place(name)[0]}
    for key, g in ring_rows.items():
        if not g.get("tested"):
            continue
        found, usable_, shift_ = RR.dip(g)
        r = own[key]
        require(found == bool(r.get("dip")) and usable_ == r["usable"] and (not found or (shift_ == (r["dy"], r["dx"]) and
                abs(g["f0"]["dip"]["value"] - r["depth"]) < 1e-9)), f"ring run and dip rule differ: {name} {key}")
        compared += 1
require(compared > 300, "too few surfaces compared between the ring run and the dip rule")

# Lung cores: truth = mask displacement for the H&E nuclei, zero for the Xenium nuclei; 3 x 3 regions of each core.
region_names = {f"r{i}c{j}" for i in range(3) for j in range(3)}
for core, c in lung.items():
    for kind, truth in (("he", (c["mask_dy"], c["mask_dx"])), ("cpsam", lung_cp[core]), ("xenium", (0, 0))):
        rows_, _ = RR.placement(f"lung_{kind}_{core}")
        require(set(rows_) == {"whole"} | region_names, f"lung {kind} {core}: not a 3 x 3 division of the core")
        require(all(tuple(g["truth"]) == truth for k, g in rows_.items() if g.get("tested")), f"lung {kind} {core}: truth is not the displacement")
macros["regchkLungDivision"] = "3"

# H&E-nuclei cases: the whole cores, each whole section, and their regions holding enough nucleus locations.
he_cases = []
for core in lung:
    for pre in ("lung_he_", "lung_cpsam_"):
        he_cases += [(f"{pre}{core}", g) for g in RR.tested(RR.placement(f"{pre}{core}")[0])]
for cp_ in (False, True):
    for sec in RR.sections("discovery", cpsam=cp_):
        rows_, meta_ = RR.rows_file(sec["he_rows"])
        require("whole" in rows_ and rows_["whole"]["tested"], f"{sec['id']}: no whole-section row")
        require(len(rows_) == 1 + sec["regions_division"] ** 2, f"{sec['id']}: not a {sec['regions_division']} x {sec['regions_division']} division")
        he_cases += [(sec["id"] + (" cpsam" if cp_ else ""), g) for g in RR.tested(rows_)]
require(all(g["truth"] is not None and g["nucleus_in_tissue"] >= MIN_NUCLEUS for _, g in he_cases), "an H&E case lacks a truth or nucleus locations")
no_dip = [(n, g) for n, g in he_cases if not RR.dip(g)[1]]
shifted = [(n, g) for n, g in no_dip if RR.dip(g)[0] and RR.peak(g)[1]]
off = [RR.cheb(RR.peak(g)[2], tuple(g["truth"])) for _, g in shifted]
require(max(off) <= 1, "a usable ring peak lies 2 or more locations from the truth")
macros["regchkFringeHeCases"] = str(len(no_dip))
macros["regchkFringeHeShifts"] = str(len(shifted))
macros["regchkFringeHeExact"] = str(sum(o == 0 for o in off))
# nowhere on the truth sets is a usable width-3 ring peak 2 or more locations from the truth (H&E and Xenium nuclei, cores and sections)
truth_sets = [(n, RR.placement(n)[0]) for n in LUNG_NAMES]
for sec in RR.sections():            # discovery and confirmation sections alike: no usable ring peak two or more off its truth
    truth_sets += [(sec["id"] + " he", RR.rows_file(sec["he_rows"])[0]), (sec["id"] + " xenium", RR.rows_file(sec["xen_rows"])[0])]
for sec in RR.sections(cpsam=True):
    truth_sets += [(sec["id"] + " cpsam", RR.rows_file(sec["he_rows"])[0])]
for n, rows_ in truth_sets:
    for g in RR.tested(rows_):
        ok_, us_, sh_, _ = RR.peak(g)
        require(not us_ or RR.cheb(sh_, tuple(g["truth"])) <= 1, f"{n} {g['surface']}: usable ring peak 2 or more locations from the truth")

# Confirmation set: the whole sections tested after the ring width was fixed, kept apart from the discovery counts above.
confirm_cases = []
for sec in RR.sections("confirmation", cpsam=True):      # the Cellpose-SAM H&E nuclei of the confirmation section: counted with the StarDist ones below
    rows_, _ = RR.rows_file(sec["he_rows"])
    require(len(rows_) == 1 + sec["regions_division"] ** 2, f"{sec['id']}: not a {sec['regions_division']} x {sec['regions_division']} division")
    confirm_cases += [(sec["id"] + " cpsam", g) for g in RR.tested(rows_)]
    require(all(g["truth"] is not None and g["nucleus_in_tissue"] >= MIN_NUCLEUS for g in RR.tested(rows_)), f"{sec['id']} cpsam: a case lacks a truth")
for sec in RR.sections("confirmation"):
    rows_, _ = RR.rows_file(sec["he_rows"])
    require(len(rows_) == 1 + sec["regions_division"] ** 2, f"{sec['id']}: not a {sec['regions_division']} x {sec['regions_division']} division")
    confirm_cases += [(sec["id"], g) for g in RR.tested(rows_)]
    require(all(g["truth"] is not None and g["nucleus_in_tissue"] >= MIN_NUCLEUS for g in RR.tested(rows_)), f"{sec['id']}: a case lacks a truth")
    wf, wu, wsh = RR.dip(rows_["whole"])
    require(wf and not wu and wsh == tuple(rows_["whole"]["truth"]) and abs(rows_["whole"]["f0"]["dip"]["share_within1"] - 0.65) < 1e-9,
            f"{sec['id']}: the H&E whole-section dip is not at the truth and unstable (65% within one)")
    xrows, _ = RR.rows_file(sec["xen_rows"])
    xf, xu, xsh = RR.dip(xrows["whole"])
    require(xf and xu and xsh == (0, 0), f"{sec['id']}: the Xenium-nucleus whole-section dip is not usable at (0,0)")
    if sec.get("cli_check"):                                    # the released package's own run agrees
        cli = json.loads((RR.ROOT / sec["cli_check"]).read_text())
        for kind, rr in (("he", rows_), ("xen", xrows)):
            for key, g in rr.items():
                if g.get("tested") and f"{kind}|default|{key}" in cli:
                    c, fr = cli[f"{kind}|default|{key}"], cli[f"{kind}|fringe|{key}"]
                    f_, u_, s_ = RR.dip(g)
                    _, pu_, ps_, _ = RR.peak(g)
                    require(c["usable"] == u_ and (not u_ or tuple(c["shift"]) == s_), f"{sec['id']} {kind} {key}: package dip differs from the ring run")
                    require(u_ or (fr["usable"] == pu_ and (not pu_ or tuple(fr["shift"]) == ps_)), f"{sec['id']} {kind} {key}: package ring peak differs from the ring run")
# On every whole section (StarDist H&E nuclei) the dip lies at the truth but is not usable, and the ring peak is usable at the truth.
# This statement is about the StarDist maps; the Cellpose-SAM sections are counted in the macros and reported, not asserted here.
for sec in RR.sections():
    w_ = RR.rows_file(sec["he_rows"])[0]["whole"]
    f_, u_, s_ = RR.dip(w_)
    _, pu_, ps_, _ = RR.peak(w_)
    require(f_ and not u_ and s_ == tuple(w_["truth"]) and pu_ and ps_ == tuple(w_["truth"]),
            f"{sec['id']}: whole section is not 'dip at the truth but unstable, ring peak usable at the truth'")
c_no_dip = [(n, g) for n, g in confirm_cases if not RR.dip(g)[1]]
c_shifted = [(n, g) for n, g in c_no_dip if RR.dip(g)[0] and RR.peak(g)[1]]
c_off = [RR.cheb(RR.peak(g)[2], tuple(g["truth"])) for _, g in c_shifted]
# The StarDist section: every usable ring peak exactly at the truth (as before); the Cellpose-SAM section is counted in the macros and reported.
require(all(o == 0 for (n, _), o in zip(c_shifted, c_off) if not n.endswith(" cpsam")), "a ring peak of the confirmation set is one or more locations from the truth")
for (n, g), o in zip(c_shifted, c_off):
    if o:
        print(f"registration_numbers.py: REPORTED (not repaired): {n} {g['surface']}: ring peak {RR.peak(g)[2]} usable, truth {tuple(g['truth'])}, {o} location(s) off", file=sys.stderr)
macros["regchkFringeConfirmCases"] = str(len(c_no_dip))
macros["regchkFringeConfirmShifts"] = str(len(c_shifted))
macros["regchkFringeConfirmExact"] = str(sum(o == 0 for o in c_off))
macros["regchkLungSections"] = str(len(RR.sections()))

# Tumour placements: wherever the dip and the width-3 ring peak are both usable, they lie within one location of each other.
for name in RR.TUMOUR:
    for g in RR.tested(RR.placement(name)[0]):
        _, du, ds = RR.dip(g)
        _, pu, ps, _ = RR.peak(g)
        require(not (du and pu) or RR.cheb(ds, ps) <= 1, f"{name} {g['surface']}: dip and ring peak differ by 2 or more locations")

# --- The disk: the nuclei and their ring of width 3 moved as one mask (disk/, dpr.py). Where there is no dip, STARgrid -------
# can move the nuclei to its highest stable peak (stargrid-register 0.5.0 --disk); by default the first registration stands.
truth_disks = dict([(n, RR.disks(n)) for n in LUNG_NAMES] + [x for sec in RR.sections() for x in (
    (sec["id"] + " he", RR.disks(sec["he_rows"])), (sec["id"] + " xenium", RR.disks(sec["xen_rows"])))]
    + [(sec["id"] + " cpsam", RR.disks(sec["he_rows"])) for sec in RR.sections(cpsam=True)])
require(sorted(truth_disks) == sorted(n for n, _ in truth_sets), "disk rows for every truth set")
disk_cases = [(n, g, truth_disks[n][g["surface"]]) for n, rows_ in truth_sets for g in RR.tested(rows_)]
require(all(k["tested"] and k["bounds"] == g["bounds"] and k.get("truth") == g["truth"] for _, g, k in disk_cases),
        "disk rows do not match the ring rows")
disk_off = [(n, g, RR.cheb(RR.disk(k)[2], tuple(g["truth"]))) for n, g, k in disk_cases if RR.disk(k)[1]]
require(disk_off and all(o <= 1 for _, _, o in disk_off), "a usable disk on the truth sets 2 or more locations from the truth")
macros["regchkDiskLungUsable"] = str(len(disk_off))
macros["regchkDiskLungUsableExact"] = str(sum(o == 0 for _, _, o in disk_off))
nodip_off = [o for n, g, o in disk_off if not RR.dip(g)[0]]
macros["regchkDiskLungNoDip"] = str(sum(not RR.dip(g)[0] for _, g, _ in disk_cases))
macros["regchkDiskLungNoDipShifts"] = str(len(nodip_off))
macros["regchkDiskLungNoDipExact"] = str(sum(o == 0 for o in nodip_off))
# Kidney and mouse brain have no dip as a whole; there the disk is usable within one location of zero shift (Space Ranger's
# registration), and so is every usable disk of a public region without a dip (Table S12).
for ds in RR.PUBLIC:
    rows_, discs = RR.placement(f"public_{ds}")[0], RR.disks(f"public_{ds}")
    for g in RR.tested(rows_):
        _, du_, dsh_, _ = RR.disk(discs[g["surface"]])
        require(RR.dip(g)[0] or not du_ or RR.cheb(dsh_, (0, 0)) <= 1, f"{ds} {g['surface']}: usable disk without a dip 2 or more from zero")
    if ds in ("kidney", "mouse_brain"):
        require(not RR.dip(rows_["whole"])[0] and RR.disk(discs["whole"])[1], f"{ds}: the whole slide is not 'no dip, disk usable'")
# Tumour placements: wherever the dip and the disk are both usable, they lie within one location of each other.
for name in RR.TUMOUR:
    discs = RR.disks(name)
    for g in RR.tested(RR.placement(name)[0]):
        _, du, ds = RR.dip(g)
        _, ku, ks, _ = RR.disk(discs[g["surface"]])
        require(not (du and ku) or RR.cheb(ds, ks) <= 1, f"{name} {g['surface']}: dip and disk differ by 2 or more locations")


# --- The correlation ratio (stargrid-register 0.6.0; paper_results/registration_correlation_ratio_20261108) ----------------
# The dip progressed into the profile of counts across the nucleus edge, matched by the correlation ratio after each count
# is divided by its local mean (BASELINE locations each way). The surfaces and truths are those of the ring run (above).
import registration_ratio as CR  # noqa: E402

macros["regcrBaselineLoc"] = str(CR.BASELINE)
macros["regcrBaselineUm"] = f"{UM * CR.BASELINE:.0f}"
macros["regcrBaselineSide"] = str(2 * CR.BASELINE + 1)
macros["regcrBandsIn"] = str(sum(b.startswith("I") for b in CR.BANDS))
macros["regcrBandsOut"] = str(sum(b.startswith("O") for b in CR.BANDS))
macros["regcrSweepMinUm"] = f"{UM * min(CR.BOXES):.0f}"
macros["regcrSweepMaxUm"] = f"{UM * max(CR.BOXES):.0f}"
macros["regcrSweepWindows"] = str(len(CR.BOXES))
require(CR.names() == sorted([n for n in RR.TUMOUR] + LUNG_NAMES + [f"public_{p}" for p in RR.PUBLIC]
                             + ["postxen_he", "postxen_xen", "expt2_he", "expt2_xen", "postxen_cpsam", "expt2_cpsam"]), "correlation-ratio placements")
cr_rows = {n: CR.placement(n)[0] for n in CR.names()}
require("spatch_after_ratio" in RR.TUMOUR and "spatch_after_new" not in RR.TUMOUR and "spatch_after_ratio" in cr_rows
        and "spatch_after_new" not in cr_rows, "the tumour set's SPATCH-after placement is not spatch_after_ratio")
ring_of = {n: RR.placement(n)[0] for n in CR.names() if n in RR.TUMOUR or n in LUNG_NAMES or n.startswith("public_")}
for sec in RR.sections():
    ring_of[Path(sec["he_rows"]).name[: -len(".rows.json")]] = RR.rows_file(sec["he_rows"])[0]
    ring_of[Path(sec["xen_rows"]).name[: -len(".rows.json")]] = RR.rows_file(sec["xen_rows"])[0]
for sec in RR.sections(cpsam=True):
    ring_of[Path(sec["he_rows"]).name[: -len(".rows.json")]] = RR.rows_file(sec["he_rows"])[0]
require(sorted(ring_of) == CR.names(), "ring rows for every correlation-ratio placement")
for n, rows_ in cr_rows.items():                 # the same surfaces, bounds and truths as the ring run
    for key, g in rows_.items():
        h = ring_of[n][key]
        require(g["tested"] == h["tested"] and g["bounds"] == h["bounds"] and g.get("truth") == h.get("truth"), f"{n} {key}: surfaces differ")
truth_names = [n for n in CR.names() if any(g.get("truth") is not None for g in CR.tested(cr_rows[n]))]
require(sorted(truth_names) == sorted(LUNG_NAMES + ["postxen_he", "postxen_xen", "expt2_he", "expt2_xen", "postxen_cpsam", "expt2_cpsam"]), "truth sets")
# the truth of each Cellpose-SAM H&E map is the mask displacement of that map (truth/ of the lung record), as written into its rows
for n in [f"lung_cpsam_{c}" for c in lung]:
    for g in CR.tested(cr_rows[n]):
        require(tuple(g["truth"]) == tuple(cp_truth["vannan"][n[len("lung_cpsam_"):]]), f"{n} {g['surface']}: truth differs from truth_table.json")
for n, short in (("postxen_cpsam", "expt1"), ("expt2_cpsam", "expt2")):
    for g in CR.tested(cr_rows[n]):
        require(tuple(g["truth"]) == tuple(cp_truth[short][g["surface"]]), f"{n} {g['surface']}: truth differs from truth_table.json")


def cr_eval(setting):
    """Per baseline setting: lung truth offsets, tumour offsets from the usable dip, Space Ranger 4 offsets from zero."""
    lung_o = [CR.cheb(CR.ratio(g, setting)[2], tuple(g["truth"])) for n in truth_names for g in CR.tested(cr_rows[n])
              if CR.ratio(g, setting)[1]]
    tum = [CR.cheb(CR.ratio(g, setting)[2], RR.dip(ring_of[n][g["surface"]])[2]) for n in RR.TUMOUR for g in CR.tested(cr_rows[n])
           if CR.ratio(g, setting)[1] and RR.dip(ring_of[n][g["surface"]])[1]]
    sr = [CR.cheb(CR.ratio(g, setting)[2], (0, 0)) for n in CR.SR4 for g in CR.tested(cr_rows[n]) if CR.ratio(g, setting)[1]]
    return lung_o, tum, sr


sweep = {b: cr_eval(f"box{b}") for b in CR.BOXES}
lung_o, tum_o, sr_o = sweep[CR.BASELINE]
require(max((s[0].count(0), -b) for b, s in sweep.items())[1] == -CR.BASELINE, "the default window does not give the most exact lung shifts")
dens = [b for b in CR.BOXES if sum(o >= 2 for o in sweep[b][1])]
require(dens and all(sum(o >= 2 for o in sweep[b][1]) for b in CR.BOXES if b >= min(dens)), "tumour disagreements do not start at one window")
require(not any(o >= 2 for b in CR.BOXES if b < min(dens) for o in sweep[b][1]), "tumour disagreements below the density window")
macros["regcrDensityFromUm"] = f"{UM * min(dens):.0f}"
macros["regcrDensityTumourMax"] = str(max(sum(o >= 2 for o in sweep[b][1]) for b in CR.BOXES))
lung_n = sum(len(CR.tested(cr_rows[n])) for n in truth_names)
macros["regcrLungSurfaces"] = str(lung_n)
macros["regcrLungShifts"] = str(len(lung_o))
macros["regcrLungExact"] = str(lung_o.count(0))
macros["regcrLungOneOff"] = str(lung_o.count(1))
macros["regcrLungTwoOff"] = str(sum(o >= 2 for o in lung_o))
require(max(lung_o) == 2 and sum(o >= 2 for o in lung_o) == 1, "lung: expected one shift two locations off and none further")
dip_lung = [CR.cheb(RR.dip(ring_of[n][g["surface"]])[2], tuple(g["truth"])) for n in truth_names for g in CR.tested(cr_rows[n])
            if RR.dip(ring_of[n][g["surface"]])[1]]
macros["regcrLungDipShifts"] = str(len(dip_lung))
macros["regcrLungDipExact"] = str(dip_lung.count(0))
is_xen = lambda n: "xenium" in n or n.endswith("_xen")
is_cp = lambda n: "cpsam" in n
he_whole = [g for n in truth_names if not is_xen(n) for g in CR.tested(cr_rows[n]) if g["surface"] == "whole"]
xen_whole = [g for n in truth_names if is_xen(n) for g in CR.tested(cr_rows[n]) if g["surface"] == "whole"]
at_truth = lambda g: bool(CR.ratio(g)[1] and CR.ratio(g)[2] == tuple(g["truth"]))
macros["regcrLungHeWhole"] = str(len(he_whole))
macros["regcrLungHeWholeExact"] = str(sum(at_truth(g) for g in he_whole))
macros["regcrLungXenWhole"] = str(len(xen_whole))
macros["regcrLungXenWholeExact"] = str(sum(at_truth(g) for g in xen_whole))
# per H&E nucleus map: Cellpose-SAM and StarDist (control)
for key, sel in (("Cpsam", is_cp), ("Stardist", lambda n: not is_cp(n) and not is_xen(n))):
    wh = [g for n in truth_names if sel(n) for g in CR.tested(cr_rows[n]) if g["surface"] == "whole"]
    allg = [g for n in truth_names if sel(n) for g in CR.tested(cr_rows[n])]
    off_ = [CR.cheb(CR.ratio(g)[2], tuple(g["truth"])) for g in allg if CR.ratio(g)[1]]
    macros[f"regcrLungHe{key}Whole"] = str(len(wh))
    macros[f"regcrLungHe{key}WholeExact"] = str(sum(at_truth(g) for g in wh))
    macros[f"regcrLungHe{key}Surfaces"] = str(len(allg))
    macros[f"regcrLungHe{key}Shifts"] = str(len(off_))
    macros[f"regcrLungHe{key}Exact"] = str(off_.count(0))
    macros[f"regcrLungHe{key}OneOff"] = str(off_.count(1))
    macros[f"regcrLungHe{key}TwoOff"] = str(sum(o >= 2 for o in off_))
macros["regcrLungHeCpsamWholeOneOff"] = str(sum(bool(CR.ratio(g)[1]) and CR.cheb(CR.ratio(g)[2], tuple(g["truth"])) == 1
                                                for n in truth_names if is_cp(n) for g in CR.tested(cr_rows[n]) if g["surface"] == "whole"))
# The two H&E segmentations side by side: the cores, sections and regions where both give a usable shift. The text says the two
# are comparable (no direction is claimed).
CONTROL = {**{f"lung_cpsam_{c}": f"lung_he_{c}" for c in lung}, "postxen_cpsam": "postxen_he", "expt2_cpsam": "expt2_he"}
pairs = []
for n, c in CONTROL.items():
    ctl = {g["surface"]: g for g in CR.tested(cr_rows[c]) if CR.ratio(g)[1]}
    pairs += [(g, ctl[g["surface"]]) for g in CR.tested(cr_rows[n]) if CR.ratio(g)[1] and g["surface"] in ctl]
macros["regcrLungPairBoth"] = str(len(pairs))
macros["regcrLungPairWithinOne"] = str(sum(CR.cheb(CR.ratio(a)[2], CR.ratio(b)[2]) <= 1 for a, b in pairs))
macros["regcrLungPairCpsamExact"] = str(sum(at_truth(a) for a, _ in pairs))
macros["regcrLungPairStardistExact"] = str(sum(at_truth(b) for _, b in pairs))
require(int(macros["regcrLungHeCpsamTwoOff"]) == 0, "the abstract says every Cellpose-SAM lung shift is within one location of the truth")
n_maps = int(macros["regchkLungCores"]) + len(RR.sections())          # whole cores and sections per H&E map
require(macros["regcrLungHeStardistWholeExact"] == macros["regcrLungHeStardistWhole"] == str(n_maps),
        "lung: a StarDist H&E whole core or section not at the truth")
require(macros["regcrLungHeWhole"] == str(2 * n_maps), "lung: whole cores and sections of the two H&E maps")
for sec in RR.sections():                         # the whole sections: the dip at the truth but not usable (above), the ratio usable there
    g = cr_rows[Path(sec["he_rows"]).name[: -len(".rows.json")]]["whole"]
    require(CR.ratio(g)[1] and CR.ratio(g)[2] == tuple(g["truth"]), f"{sec['id']}: correlation ratio not at the truth")
# Cellpose-SAM whole cores and sections not at their truth (the stop rule of the 2026-11-10 lung run; kept, with the paper reporting both
# segmentations, by decision of 2026-11-10): counted in regcrLungHeCpsamWholeExact and WholeOneOff, and listed here.
for n in truth_names:
    if is_cp(n):
        g = cr_rows[n]["whole"]
        if not at_truth(g):
            print(f"registration_numbers.py: Cellpose-SAM whole not at its truth: {n}: ratio {CR.ratio(g)[2]} usable={CR.ratio(g)[1]}, truth {tuple(g['truth'])}", file=sys.stderr)
# Tumour placements: against the usable dip; every whole slide at the dip's shift.
require(max(tum_o) <= 1, "tumours: correlation ratio two or more locations from a usable dip")
macros["regcrTumourBoth"] = str(len(tum_o))
macros["regcrTumourSame"] = str(tum_o.count(0))
tw = [(CR.ratio(cr_rows[n]["whole"]), RR.dip(ring_of[n]["whole"])) for n in RR.TUMOUR]
require(all(c[1] and d[1] and c[2] == d[2] for c, d in tw), "tumours: a whole slide where the ratio and the dip differ")
macros["regcrTumourWhole"] = str(len(tw))
tum_reg = [(CR.ratio(g)[1], RR.dip(ring_of[n][g["surface"]])[1]) for n in RR.TUMOUR for g in CR.tested(cr_rows[n], whole=False)]
macros["regcrTumourRegions"] = str(len(tum_reg))
macros["regcrTumourRegionsRatio"] = str(sum(c for c, _ in tum_reg))
macros["regcrTumourRegionsDip"] = str(sum(d for _, d in tum_reg))
# SPATCH: the corrected masks (fitted to the regional dips) under the ratio; Space Ranger's runs region by region.
def cr_usable(name, whole=False):
    return [g for g in CR.tested(cr_rows[name], whole=whole) if CR.ratio(g)[1]]


after_cr = cr_usable("spatch_after_ratio")
require(CR.ratio(cr_rows["spatch_after_ratio"]["whole"])[2] == (0, 0), "SPATCH: corrected masks not at zero by the ratio")
macros["regcrSpatchAfterUsable"] = str(len(after_cr))
macros["regcrSpatchAfterZero"] = str(sum(CR.ratio(g)[2] == (0, 0) for g in after_cr))
macros["regcrSpatchBeforeUsable"] = str(len(cr_usable("spatch_before")))
before_um = [distance_um(*CR.ratio(g)[2]) for g in cr_usable("spatch_before")]
macros["regcrSpatchBeforeMinUm"] = f"{min(before_um):.1f}"
macros["regcrSpatchBeforeMaxUm"] = f"{max(before_um):.1f}"
for key, name in (("SRthree", "spatch_sr3"), ("SRfour", "spatch_sr410")):
    offs = [distance_um(*CR.ratio(g)[2]) for g in cr_usable(name)]
    macros[f"regcrSpatch{key}Usable"] = str(len(offs))
    macros[f"regcrSpatch{key}Zero"] = str(sum(o == 0 for o in offs))
    macros[f"regcrSpatch{key}MaxUm"] = f"{max(offs):.1f}"
    macros[f"regcrSpatch{key}MinUm"] = f"{min(offs):.1f}"
    macros[f"regcrSpatch{key}WithinOne"] = str(sum(max(map(abs, CR.ratio(g)[2])) <= 1 for g in cr_usable(name)))
    macros[f"regcrSpatch{key}Um"] = f"{distance_um(*CR.ratio(cr_rows[name]['whole'])[2]):.1f}"
require(macros["regcrSpatchSRfourUm"] == "0.0", "SPATCH: Space Ranger 4.1.0 whole slide not at zero by the ratio")
# Every version-4 placement of our masks lies at zero by the ratio, as do our corrected placements.
for name in [n for _, n, _ in SR_FOUR] + ["crc_stardist_img"]:
    require(CR.ratio(cr_rows[name]["whole"])[1] and CR.ratio(cr_rows[name]["whole"])[2] == (0, 0), f"{name}: not at zero by the ratio")
for name in ("ov_cellpose_img", "ov_stardist_img"):
    require(CR.ratio(cr_rows[name]["whole"])[2] == (-3, -3), f"{name}: ratio not at (-3, -3)")
# Public slides (Space Ranger 4.0.1): every usable shift within one location of Space Ranger's registration.
pub = [(n, g) for n in CR.PUBLIC for g in CR.tested(cr_rows[n])]
pub_o = [CR.cheb(CR.ratio(g)[2], (0, 0)) for _, g in pub if CR.ratio(g)[1]]
require(max(pub_o) <= 1, "public slides: a shift two or more locations from Space Ranger's registration")
macros["regcrPublicSurfaces"] = str(len(pub))
macros["regcrPublicShifts"] = str(len(pub_o))
macros["regcrPublicExact"] = str(pub_o.count(0))
pub_nodip = [(n, g) for n, g in pub if not RR.dip(ring_of[n][g["surface"]])[0]]
macros["regcrPublicNoDip"] = str(len(pub_nodip))
macros["regcrPublicNoDipShifts"] = str(sum(CR.ratio(g)[1] for _, g in pub_nodip))
for n in ("public_kidney", "public_mouse_brain"):
    require(not RR.dip(ring_of[n]["whole"])[0] and CR.ratio(cr_rows[n]["whole"])[1], f"{n}: whole slide not 'no dip, ratio usable'")
# All surfaces and placements.
alls = [g for n in CR.names() for g in CR.tested(cr_rows[n])]
macros["regcrSurfaces"] = str(len(alls))
macros["regcrSurfacesShifts"] = str(sum(CR.ratio(g)[1] for g in alls))
macros["regcrSurfacesDip"] = str(sum(RR.dip(ring_of[n][g["surface"]])[1] for n in CR.names() for g in CR.tested(cr_rows[n])))


# --- Nucleus by nucleus: paired depth in place and with the far-shift background (A1) ------------
# Cases: C1, C2 and July use the nucleus maps deposited with the datasets; T1 our Cellpose masks moved
# to their dip; the four lung cores the Xenium nuclei. C1 and T1 are on the ovarian slide.
TUMOUR = ("C1", "C2", "July", "T1")
CASE_SLIDE = {"C1": "ovarian", "C2": "colorectal", "July": "examined", "T1": "ovarian"}
QUARTERS = tuple(f"density_Q{q}" for q in range(1, 5))
far_rows = tsv(T / "far_shift_all_summaries.tsv")
far = {(r["case"], r["group"], int(r["radius"])): r for r in far_rows if r["offset"] == "pooled"}
real = {(r["case"], r["group"], int(r["radius"])): r for r in tsv(T / "amendment6_all_summaries.tsv")}
cases = list(dict.fromkeys(r["case"] for r in far_rows))
lung_cases = [c for c in cases if c not in TUMOUR]
require(set(TUMOUR) <= set(cases) and set(lung_cases) == set(lung), "far-shift cases differ from the lung cores and tumour sets")
for key, row in far.items():
    if key[1] in ("all",) + QUARTERS:
        require(float(row["real_median"]) == float(real[key]["median"]), f"far shift {key}: real median differs from Amendment 6")
        require(math.isclose(float(row["real_minus_control"]), float(row["real_median"]) - float(row["median"]), abs_tol=1e-12),
                f"far shift {key}: difference is not real minus control")
draws = {r["valid_replicates"] for r in far_rows} | {r["valid_replicates"] for r in real.values()}
require(len(draws) == 1, "the number of block draws differs between rows")
offsets = {r["offset"] for r in far_rows} - {"pooled"}
steps = {abs(int(v)) for o in offsets for v in o.split("_")}
require(len(steps) == 1 and len(offsets) == 4, "far-shift offsets are not the four diagonals at one distance")
far_step = steps.pop()
smallest = {json.loads(r["size_range"])[0] for (c, group, _), r in real.items() if group == "all"}
require(len(smallest) == 1, "smallest nucleus differs between cases")
macros["regchkFarCases"] = str(len(cases))
macros["regchkFarOffsetLocations"] = str(far_step)
macros["regchkFarShiftUm"] = f"{UM * far_step:.0f}"
macros["regchkFarDirections"] = str(len(offsets))
macros["regchkPairMinLocations"] = str(smallest.pop())
macros["regchkPairReplicates"] = draws.pop()
macros["regchkPairShiftsOne"] = str(chebyshev_ring(1))
macros["regchkPairShiftsTwo"] = str(chebyshev_ring(2))


def lo_hi(row, column):
    return interval(row[column])


# The background itself: positive, with its interval above zero, in every case and in every direction.
per_direction = [r for r in far_rows if r["offset"] != "pooled" and r["group"] == "all" and r["radius"] == "1"]
require(len(per_direction) == len(cases) * len(offsets) and all(lo_hi(r, "interval")[0] > 0 for r in per_direction),
        "a per-direction background at radius 1 is not above zero")
macros["regchkFarOffsetControls"] = str(len(per_direction))
for radius, suffix in ((1, ""), (2, "Two")):
    control = [float(far[(c, "all", radius)]["median"]) for c in cases]
    require(all(lo_hi(far[(c, "all", radius)], "interval")[0] > 0 for c in cases), f"radius {radius}: a background not above zero")
    macros[f"regchkFarCtl{suffix}MinPP"] = pp(min(control))
    macros[f"regchkFarCtl{suffix}MaxPP"] = pp(max(control))

positive = {r: [c for c in cases if lo_hi(far[(c, "all", r)], "difference_interval")[0] > 0] for r in (1, 2)}
require(positive[1] == positive[2] and set(TUMOUR) <= set(positive[1]), "positive cases differ between radii")
macros["regchkFarPositiveCases"] = str(len(positive[1]))
macros["regchkFarPositiveCasesTwo"] = str(len(positive[2]))
macros["regchkFarTumourSets"] = str(len(TUMOUR))
macros["regchkFarTumourSlides"] = str(len({CASE_SLIDE[c] for c in TUMOUR}))
for radius, word in ((1, "One"), (2, "Two")):
    diff = [float(far[(c, "all", radius)]["real_minus_control"]) for c in TUMOUR]
    macros[f"regchkFarTumour{word}MinPP"] = pp(min(diff))
    macros[f"regchkFarTumour{word}MaxPP"] = pp(max(diff))
    quarters_up = sum(lo_hi(far[(c, q, radius)], "difference_interval")[0] > 0 for c in TUMOUR for q in QUARTERS)
    require(quarters_up == len(TUMOUR) * len(QUARTERS), f"radius {radius}: a tumour density quarter is not above zero")
macros["regchkFarTumourQuarters"] = str(len(TUMOUR) * len(QUARTERS))
lung_up = [c for c in lung_cases if c in positive[1]]
macros["regchkFarLungPositive"] = str(len(lung_up))
lung_diff = [float(far[(c, "all", 1)]["real_minus_control"]) for c in lung_up]
macros["regchkFarLungOneMinPP"] = pp(min(lung_diff))
macros["regchkFarLungOneMaxPP"] = pp(max(lung_diff))
lung_quarters = [(c, q) for c in lung_cases for q in QUARTERS]
macros["regchkFarLungQuarters"] = str(len(lung_quarters))
macros["regchkFarLungQuartersNotPositive"] = str(
    sum(lo_hi(far[(c, q, 1)], "difference_interval")[0] <= 0 for c, q in lung_quarters))
null = [c for c in lung_cases if lo_hi(far[(c, "all", 1)], "difference_interval")[0] <= 0 <= lo_hi(far[(c, "all", 1)], "difference_interval")[1]]
negative = [c for c in lung_cases if lo_hi(far[(c, "all", 1)], "difference_interval")[1] < 0]
require(len(null) == 1 and len(negative) == 1, "expected one lung core with an interval including zero and one below zero")
for core in null + negative:
    require(lo_hi(far[(core, "all", 2)], "difference_interval")[1] < 0, f"{core}: not negative at radius 2")
for label, core in (("Null", null[0]), ("Neg", negative[0])):
    for radius, word in ((1, "One"), (2, "Two")):
        row = far[(core, "all", radius)]
        low, high = lo_hi(row, "difference_interval")
        macros[f"regchkFar{label}Core{word}PP"] = pp(float(row["real_minus_control"]))
        macros[f"regchkFar{label}Core{word}Lo"] = pp(low)
        macros[f"regchkFar{label}Core{word}Hi"] = pp(high)
raw = [float(real[(c, "all", 1)]["median"]) for c in TUMOUR]
macros["regchkPairedTumourMinPct"] = pp(min(raw))
macros["regchkPairedTumourMaxPct"] = pp(max(raw))

# Names used by the drafted text edits (review of 2026-10-24) and figure captions; same values.
for alias, name in (("regchkNucFarOffsetLoc", "regchkFarOffsetLocations"), ("regchkNucFarCases", "regchkFarCases"),
                    ("regchkNucFarPositive", "regchkFarPositiveCases"), ("regchkNucFarCtrlMinPp", "regchkFarCtlMinPP"),
                    ("regchkNucFarCtrlMaxPp", "regchkFarCtlMaxPP"), ("regchkNucFarTumourMinPp", "regchkFarTumourOneMinPP"),
                    ("regchkNucFarTumourMaxPp", "regchkFarTumourOneMaxPP"), ("regchkNucFarLungMinPp", "regchkFarLungOneMinPP"),
                    ("regchkNucFarLungMaxPp", "regchkFarLungOneMaxPP"), ("regchkNucFarLungPositive", "regchkFarLungPositive"),
                    ("regchkPairLungPositive", "regchkFarLungPositive")):
    macros[alias] = macros[name]

# --- Cancer genes: sensitivity of the top-gene comparison (B5) ------------------------------------
sens = {r["variant"]: r for r in tsv(T / "cancer_sensitivity.tsv")}
primary = sens.pop("primary")
require(primary["hard_wins"] == known["xenHardWins"] and sig(float(primary["sign_test_P"])) == known["xenHardSignP"]
        and int(primary["N"]) == int(known["xenTopGenes"]), "primary sensitivity row differs from numbers.tex")
for column, name in (("median_hard", "xenHardPearson"), ("median_space_ranger", "xenSRPearson"), ("median_strict", "xenStrictPearson")):
    require(f"{float(primary[column]):.3f}" == known[name], f"primary {column} differs from {name}")
gate128 = load("cancer_reproduction_128um_summary.json", T)
require(gate128["top20_byte_identical"] is True and gate128["largest_absolute_pearson_difference"] == 0.0,
        "the 128 um generator did not reproduce the accepted top-20 table")
for r in [primary] + list(sens.values()):
    require(r["ties"] == "0" and r["nonfinite"] == "0" and int(r["sign_test_N"]) == int(r["hard_wins"]) + int(r["hard_losses"]),
            f"sensitivity {r['variant']}: ties, undefined correlations or sign-test N")
same_direction = [v for v, r in sens.items() if int(r["hard_wins"]) > int(r["hard_losses"])]
require(len(same_direction) == load("cancer_report_summary.json", T)["directional_agreement_variants"], "directional agreement")
significant = [v for v, r in sens.items() if float(r["sign_test_P"]) < 0.05]
outcomes = {tuple(r[k] for k in r if k != "variant") for r in sens.values()}
macros["xenSensVariants"] = str(len(sens))
macros["xenSensSameDirection"] = str(len(same_direction))
macros["xenSensSignificant"] = str(len(significant))
macros["xenSensNotSignificant"] = str(len(sens) - len(significant))
macros["xenSensDistinct"] = str(len(outcomes))
macros["xenSensDistinctSignificant"] = str(len({tuple(r[k] for k in r if k != "variant") for v, r in sens.items()
                                                if v in significant}))
macros["xenSensHardMedianAbove"] = str(sum(float(r["median_hard"]) > float(r["median_space_ranger"]) for r in sens.values()))
macros["xenSensPrimaryPatchUm"] = primary["patch_um"]
# Patch thresholds: the defaults of the archived comparison script, the same file the 128 um run recorded.
xen_script = T / "scripts" / "compare_visium_to_adjacent_xenium.py"
xen_recorded = load("cancer_reproduction_128um_summary.json", T)["scripts"]
require(any(Path(k).name == xen_script.name and v == __import__("hashlib").sha256(xen_script.read_bytes()).hexdigest()
            for k, v in xen_recorded.items()), "archived Xenium comparison script differs from the one the run recorded")
xen_text = xen_script.read_text()
for flag, name in (("--min-xenium-cells", "xenMinCells"), ("--min-visium-tissue-bins", "xenMinTissueBins")):
    found = re.search(r'add_argument\("%s", type=int, default=(\d+)\)' % re.escape(flag), xen_text)
    require(found is not None, f"default of {flag} not found")
    macros[name] = found.group(1)
require(sens["N50"] == {**sens["all"], "variant": "N50"}, "N50 and all-gene variants differ")
macros["xenEligibleGenes"] = sens["all"]["N"]
short = [(int(m.group(1)), int(sens[v]["N"])) for v in sens if (m := re.fullmatch(r"N(\d+)", v)) and int(m.group(1)) != int(sens[v]["N"])]
require(len(short) == 1 and short[0][1] == int(sens["all"]["N"]), "only the largest gene-number variant should run short of genes")
macros["xenSensFiftyRequested"] = str(short[0][0])
for variant, word in (("N10", "Ten"), ("N30", "Thirty"), ("all", "All"), ("spearman", "Spearman")):
    r = sens[variant]
    macros[f"xenSens{word}Wins"] = r["hard_wins"]
    macros[f"xenSens{word}N"] = r["N"]
    macros[f"xenSens{word}P"] = sig(float(r["sign_test_P"]))
for variant in ("N10", "N30"):
    require(int(sens[variant]["N"]) == int(variant[1:]), f"{variant}: number of genes")
small, large = sens["patch64"], sens["patch256"]
require(all(small[k] == large[k] for k in ("N", "hard_wins", "hard_losses", "sign_test_P")), "64 and 256 um variants differ in wins or P")
macros["xenSensPatchSmallUm"], macros["xenSensPatchLargeUm"] = small["patch_um"], large["patch_um"]
macros["xenSensPatchWins"], macros["xenSensPatchN"], macros["xenSensPatchP"] = small["hard_wins"], small["N"], sig(float(small["sign_test_P"]))
for column, name in (("median_hard", "Hard"), ("median_space_ranger", "SR"), ("median_strict", "Strict")):
    macros[f"xenSensSpearman{name}"] = f"{float(sens['spearman'][column]):.3f}"



# --- Source provenance of the registration results ----------------------------------------------
prov = load("provenance_summary.json", T)
audit = tsv(T / "provenance_SOURCE_AUDIT.tsv")
require(prov["recorded_entries"] == len(audit) and prov["matched_entries"] == sum(r["status"] == "matched" for r in audit),
        "provenance counts differ from the audit table")
require(prov["archived_unique_files"] == len(MANIFEST) and len(prov["missing"]) == len(audit) - prov["matched_entries"],
        "provenance archive or gap counts")
require(len(prov["runs"]) == len({r["run"] for r in audit}), "provenance runs")
macros["regsrcEntries"] = str(prov["recorded_entries"])
macros["regsrcExact"] = str(prov["matched_entries"])
macros["regsrcFiles"] = str(prov["archived_unique_files"])
macros["regsrcMissing"] = str(len(prov["missing"]))
macros["regsrcMissingRuns"] = str(len({m["run"] for m in prov["missing"]}))
macros["regsrcRuns"] = str(len(prov["runs"]))

print("% Generated by figures/scripts/registration_numbers.py from paper_results/registration_check_20261018")
print("% and the SPATCH placement record of 2026-10-05. Distances are in micrometres.")
for name in existing:
    print(rf"\newcommand{{\{name}}}{{{macros[name]}}}")
print("% Added 2026-10-24: settings of the check, every version-4 placement, the lung local dips, and the paper")
print("% tests of 2026-10-21 (paper_results/paper_tests_20261021). PP: percentage points of paired depth.")
print("% Negative values carry a hyphen; set them in math mode.")
for name, value in macros.items():
    if name not in existing:
        print(rf"\newcommand{{\{name}}}{{{value}}}")
