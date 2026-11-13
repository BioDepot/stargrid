"""Readers of the correlation-ratio registration record, shared by the registration builders.

Reads paper_results/registration_correlation_ratio_20261108/sweep/: <placement>.rows_c.json (csw_c.py output: per
surface, for each baseline setting, the highest local maximum of the correlation ratio eta^2 with its shift, stability
and profile; the surfaces, bounds and truths are those of the ring run) and eta_box2_full.npz (the full-data eta^2
surface at the default baseline, by '<placement>__<surface>').
The method (stargrid-register 0.6.0): bands by distance from the nucleus edge, one location wide (I3, I2, I1 inside, O1-O6
outside, and the rest); each count divided by the mean count of the supported locations within BASELINE locations on each
axis; eta^2 of the corrected counts on the bands at every shift; the shift is the highest local maximum within the search,
usable when interior and within one location of it in at least 95% of 200 spatial-block draws.
"""
import json
from pathlib import Path

import numpy as np

from registration_ring import CELLPOSE, CELLPOSE_ROOT, CPSAM_LUNG, CPSAM_LUNG_ROOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[3] / "paper_results" / "registration_correlation_ratio_20261108"
SWEEP = ROOT / "sweep"
# The SPATCH masks moved by the displacement fitted to the regional shifts of the correlation ratio (2026-11-09), placement
# spatch_after_ratio, replace spatch_after_new of the sweep. Its sweep rows (same harness, every window) are in
# paper_results/spatch_rescore_ratio_20261109/tumour_set/sweep; its dip-rule rows in .../placements.
RATIO_PLACE = Path(__file__).resolve().parents[3] / "paper_results" / "spatch_rescore_ratio_20261109" / "placements"
RATIO_SWEEP = Path(__file__).resolve().parents[3] / "paper_results" / "spatch_rescore_ratio_20261109" / "tumour_set" / "sweep"
SUPERSEDED = "spatch_after_new"
# The Cellpose-SAM placements of the colorectal slide and slide A1 (record registration_cellpose_sam_20261110): sweep rows, full-data
# surfaces at the default window, and the dip rule's rows, from the same harnesses as the other placements.
CP_SWEEP = CELLPOSE_ROOT / "set" / "sweep"
CP_PLACE = CELLPOSE_ROOT / "placements"
# The Cellpose-SAM H&E nuclei on the lung sets (record registration_cellpose_sam_lung_20261110): the same files for those placements.
LUNG_SWEEP = CPSAM_LUNG_ROOT / "set" / "sweep"
LUNG_PLACE = CPSAM_LUNG_ROOT / "placements"
NEW_PLACE = Path(__file__).resolve().parents[3] / "paper_results" / "registration_local_dip_20261106" / "placements"
SEARCH, DRAWS, SEED, RADIUS = 12, 200, 20261007, 13
BASELINE = 2                                   # locations; the default of stargrid-register 0.6.0
SETTING = f"box{BASELINE}"
BOXES = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30)     # the half-widths of the sweep, in locations
BANDS = ("I3", "I2", "I1", "O1", "O2", "O3", "O4", "O5", "O6", "far")
SR4 = ("public_kidney", "public_mouse_brain", "public_lymph_node", "public_prostate", "ov_cellpose_dep", "ov_stardist_dep",
       "crc_stardist_dep", "a1_sr410", "spatch_sr410", "crc_cellpose_dep", "a1_cellpose_sr410")
PUBLIC = ("public_lymph_node", "public_prostate", "public_kidney", "public_mouse_brain")


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def placements_dir(name):
    """Where the dip rule's rows (<name>.rows.json, .meta.json, .surfaces.npz) of a placement are."""
    return RATIO_PLACE if name == "spatch_after_ratio" else CP_PLACE if name in CELLPOSE else LUNG_PLACE if name in CPSAM_LUNG else NEW_PLACE


def placement(name):
    """Rows of one placement by surface, and the file's header."""
    base = RATIO_SWEEP if name == "spatch_after_ratio" else CP_SWEEP if name in CELLPOSE else LUNG_SWEEP if name in CPSAM_LUNG else SWEEP
    d = json.loads((base / f"{name}.rows_c.json").read_text())
    assert d["search"] == SEARCH and d["draws"] == DRAWS and d["seed"] == SEED and d["radius"] == RADIUS, name
    assert list(d["bands"]) == list(BANDS) and SETTING in d["settings"], name
    return {r["surface"]: r for r in d["rows"]}, d


def names():
    found = {p.name[: -len(".rows_c.json")] for p in SWEEP.glob("*.rows_c.json")}
    assert SUPERSEDED in found, "sweep record changed"
    return sorted(found - {SUPERSEDED} | {"spatch_after_ratio"} | set(CELLPOSE) | set(CPSAM_LUNG))


def ratio(g, setting=SETTING):
    """(found, usable, (dy, dx), row) of the correlation ratio's highest local maximum in a row."""
    p = g[setting]
    return p["found"], bool(p["found"] and p["usable"]), ((p["dy"], p["dx"]) if p["found"] else None), p


def tested(rows, whole=True):
    return [g for k, g in rows.items() if g.get("tested") and (whole or k != "whole")]


def surface(name, key):
    """The full-data eta^2 surface (27 x 27, shifts -13..13) of one surface at the default baseline."""
    if name == "spatch_after_ratio":
        with np.load(RATIO_SWEEP / "eta_box2_full_spatch_after_ratio.npz") as d:
            return d[f"{name}__{key}"]
    if name in CELLPOSE:
        with np.load(CP_SWEEP / "eta_box2_full_cellpose_sam.npz") as d:
            return d[f"{name}__{key}"]
    if name in CPSAM_LUNG:
        with np.load(LUNG_SWEEP / "eta_box2_full_cellpose_sam_lung.npz") as d:
            return d[f"{name}__{key}"]
    with np.load(SWEEP / "eta_box2_full.npz") as d:
        return d[f"{name}__{key}"]
