"""Readers of the ring (used where the dip is not usable) and post-Xenium lung records, shared by the registration builders.

Reads paper_results/registration_ring_postxenium_20261106/: ring/<placement>.rows.json (fpr.py output: per surface, f0 =
nuclei with the dip, f1-f3 = ring of width 1-3 locations with its peak) and whole_sections.json (the whole tissue sections
other than the four Vannan cores; to add one, copy its rows files and add an entry), and disk/<placement>.rows.json (dpr.py
output, the same surfaces: disk3 = the nuclei and their ring of width 3 moved as one mask, the disk, with its peak).
The ring: capture locations within W locations (Euclidean distance transform) of a nucleus location that are not nucleus.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "paper_results" / "registration_ring_postxenium_20261106"
SEARCH, DRAWS, SEED = 12, 200, 20261007
WIDTH = 3                                     # locations; the width of STARgrid's ring
FIRST_WIDTH = 1                               # the pre-declared width
CORES = ("VUILD49LA", "TILD113LA", "TILD111LA", "VUHD049")
# Cellpose-SAM nuclei on the colorectal slide H1-GMHFWPH/D1 (image-only stage and the deposited Space Ranger 4.1.0 registration) and on
# slide H1-VM2JXXK/A1 (Space Ranger 3.0.0 GEO deposit and our 4.1.0 run), measured with the same harnesses in the 2026-11-10 record
# (rows in its set/, the dip rule's in its placements/); the StarDist placements on the same registrations stay as the control.
CELLPOSE = ("crc_cellpose_img", "crc_cellpose_dep", "a1_cellpose_sr3", "a1_cellpose_sr410")
CELLPOSE_ROOT = Path(__file__).resolve().parents[3] / "paper_results" / "registration_cellpose_sam_20261110"
TUMOUR = ("spatch_before", "spatch_after_ratio", "spatch_sr3", "spatch_sr410", "a1_sr3", "a1_sr410", "ov_cellpose_dep",
          "ov_stardist_dep", "crc_stardist_dep", "ov_cellpose_img", "ov_stardist_img", "crc_stardist_img") + CELLPOSE
PUBLIC = ("lymph_node", "prostate", "kidney", "mouse_brain")
# Cellpose-SAM H&E nuclei on the lung sets (the four Vannan cores and the two post-Xenium sections), measured with the same harnesses in the
# 2026-11-10 lung record; the StarDist placements (lung_he_<core>, postxen_he, expt2_he) stay as the control and the Xenium placements are shared.
CPSAM_LUNG = tuple(f"lung_cpsam_{c}" for c in CORES) + ("postxen_cpsam", "expt2_cpsam")
CPSAM_LUNG_ROOT = Path(__file__).resolve().parents[3] / "paper_results" / "registration_cellpose_sam_lung_20261110"
SECTION_CPSAM = {"H1-9QZXXNP/D1": "postxen_cpsam", "H1-JGHRVB7/D1": "expt2_cpsam"}


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# spatch_after_ratio (SPATCH masks moved by the correlation-ratio fit) is measured with the same harnesses in the 2026-11-09 record.
RATIO_ROOT = Path(__file__).resolve().parents[3] / "paper_results" / "spatch_rescore_ratio_20261109" / "tumour_set"


def _root(rel):
    if Path(rel).name.split(".")[0] in CELLPOSE:
        return CELLPOSE_ROOT / "set"
    if Path(rel).name.split(".")[0] in CPSAM_LUNG:
        return CPSAM_LUNG_ROOT / "set"
    return RATIO_ROOT if "spatch_after_ratio" in rel else ROOT


def rows_file(rel):
    d = json.loads((_root(rel) / rel).read_text())
    assert d["search"] == SEARCH and d["draws"] == DRAWS and d["seed"] == SEED and d["expansions"] == [0, 1, 2, 3], rel
    return {r["surface"]: r for r in d["rows"]}, d


def placement(name):
    return rows_file(f"ring/{name}.rows.json")


def disks(rel):
    """The disk rows matching a ring-run rows file (ring/<placement>.rows.json, or the placement name), by surface."""
    rel = rel if "/" in rel else f"ring/{rel}.rows.json"
    d = json.loads((_root(rel) / rel.replace("ring/", "disk/", 1)).read_text())
    assert d["search"] == SEARCH and d["draws"] == DRAWS and d["seed"] == SEED, rel
    return {r["surface"]: r for r in d["rows"]}


def sections(group=None, cpsam=False):
    """Whole post-Xenium sections; group 'discovery' or 'confirmation' (tested after the ring width was fixed), or all.
    cpsam=True: the same sections with the Cellpose-SAM H&E nuclei (he_rows of the lung record); the Xenium rows are shared."""
    out = json.loads((ROOT / "whole_sections.json").read_text())["sections"]
    out = [x for x in out if group is None or x["group"] == group]
    if cpsam:
        out = [dict(x, he_rows=f"ring/{SECTION_CPSAM[x['id']]}.rows.json") for x in out]
    return out


def dip(g):
    """(found, usable, (dy, dx)) of the nuclei's dip in a row."""
    d = g["f0"]["dip"]
    return d["found"], bool(d["found"] and d["usable"]), ((d["dy"], d["dx"]) if d["found"] else None)


def peak(g, width=WIDTH):
    p = g[f"f{width}"]["peak"]
    return p["found"], bool(p["found"] and p["usable"]), ((p["dy"], p["dx"]) if p["found"] else None), p


def disk(g):
    """(found, usable, (dy, dx), row) of the disk's highest peak in a row of `disks`."""
    p = g["disk3"]["peak"]
    return p["found"], bool(p["found"] and p["usable"]), ((p["dy"], p["dx"]) if p["found"] else None), p


def tested(rows, whole=True):
    """Surfaces measured: the whole slide and the regions holding enough nucleus locations."""
    return [g for k, g in rows.items() if g.get("tested") and (whole or k != "whole")]
