"""The two worked reads of the examination of A1 (Fig. 3d, Table 5 macros), from the v1.12.2 examination record.

Rows, columns, edit counts and the read's barcode part come from
paper_results/release_examination_v1_12_2_20261105/data/metadata_v1/metadata/sr_audit_worked_cases.tsv; the oligo
at a location comes from the slide-design oligo lists (Space Ranger's reference data, as barcode_design_stats.py):
BC1 is the oligo at the column, BC2 the oligo at the row. Returns one row per read and square ("closest" and
"Space Ranger") with the columns of the earlier July table.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TABLE = (ROOT / "paper_results" / "release_examination_v1_12_2_20261105" / "data" / "metadata_v1" / "metadata"
         / "sr_audit_worked_cases.tsv")
# The folder written by make_visium_hd_contract.py (STAR Suite) from the slide-design file.
OLIGOS = Path(os.environ.get("VISIUM_HD_BARCODE_INPUTS", "barcode_inputs"))
CASES = {"original_one": "same 8 um bin", "original_two": "crosses 16 um bin"}


def load(table=TABLE):
    halves = {h: [l.strip() for l in open(OLIGOS / f"{h}_full_oligos.txt") if l.strip()] for h in ("bc1", "bc2")}
    out = []
    for r in csv.DictReader(open(table), delimiter="\t"):
        if r["case"] not in CASES:
            continue
        n1 = int(r["closest_bc1_length"])
        split = r["recorded_CR"][:n1] + "|" + r["recorded_CR"][n1:]
        assert int(r["closest_bc2_length"]) == len(r["recorded_CR"]) - n1
        assert (r["same_8um"] == "1") == (r["case"] == "original_one") and (r["same_16um"] == "1") == (r["case"] == "original_one")
        for square, key in (("closest", "closest"), ("Space Ranger", "sr")):
            row, col = int(r[f"{key}_row"]), int(r[f"{key}_col"])
            b1, b2 = int(r[f"{key}_bc1_edits"]), int(r[f"{key}_bc2_edits"])
            out.append({"case": CASES[r["case"]], "read_id": r["read_id"], "square": square, "row": str(row), "col": str(col),
                        "split": split, "bc1_oligo": halves["bc1"][col], "bc1_edits": str(b1),
                        "bc2_oligo": halves["bc2"][row], "bc2_edits": str(b2), "total": str(b1 + b2)})
    assert len(out) == 4
    return out
