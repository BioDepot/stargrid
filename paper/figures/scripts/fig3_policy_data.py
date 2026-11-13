#!/usr/bin/env python3
"""Source data for Fig. 3 (policy difference), from the examination of A1 rerun with STAR Suite v1.12.2.

Copies fig3_edit_joint.tsv (closest-location edits x Space Ranger location edits, every eligible read whose
Space Ranger location is not a closest location) from paper_results/release_examination_v1_12_2_20261105/data/
read_audit_v1/audit/, after checking it against that record's checksums, and writes fig3_summary.json (partition,
geometry and parent relation) from the record's results.json ("R", primary column) and the frame-class table of
the same audit, as the report step of the examination does. Both files equal those of the earlier examination.
Usage: fig3_policy_data.py [OUT_DIR]
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parents[2] / "paper_results" / "release_examination_v1_12_2_20261105"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data"
AUDIT = RECORD / "data" / "read_audit_v1" / "audit"

sums = {}
for line in (RECORD / "checksums.sha256").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    sums[name.strip().lstrip("./")] = digest
for source in (AUDIT / "fig3_edit_joint.tsv", AUDIT / "frame_classes.tsv", RECORD / "results.json"):
    name = str(source.relative_to(RECORD))
    if hashlib.sha256(source.read_bytes()).hexdigest() != sums[name]:
        sys.exit(f"fig3_policy_data.py: {name} does not match the record's checksum")
shutil.copyfile(AUDIT / "fig3_edit_joint.tsv", OUT / "fig3_edit_joint.tsv")
print("copied fig3_edit_joint.tsv")

R = {k: v["primary"] for k, v in json.loads((RECORD / "results.json").read_text())["R"].items()}
headline = {k: R[v] for k, v in {
    "shared_eligible_reads": "eligible", "star_candidate_reads": "candidate", "star_no_candidate_reads": "no_candidate",
    "sr_selected_retained_minimum": "minimum", "sr_selected_nonminimum": "nonminimum",
    "sr_unassigned_with_star_candidate": "unassigned", "short_bc2_overlap_nonminimum": "overlap",
    "nonminimum_not_short_bc2_overlap": "other", "sr_raw_split_cost_equal_to_star_minimum": "sr_equal",
    "sr_raw_split_cost_greater_than_star_minimum": "sr_greater", "sr_raw_split_max_half_above_ed2": "frame_above_half"}.items()}
classes = {r["class"]: int(r["reads"]) for r in csv.DictReader((AUDIT / "frame_classes.tsv").open(), delimiter="\t")}
for key in ("deletion_frame+insertion_frame", "same_length_edit+insertion_frame"):
    classes.setdefault(key, 0)
summary = {"headline": headline,
           "parent_relation": {"same_8um": R["parent_same8"], "same_16um_only": R["parent_same16_only"], "cross_16um": R["parent_cross16"]},
           "sr_edit_class": classes, "ledger_rows": R["nonminimum"],
           "sources": {"summary": str(RECORD.relative_to(HERE.parents[2]) / "results.json"),
                       "read_counts": str(AUDIT.relative_to(HERE.parents[2]) / "counts.tsv")}}
(OUT / "fig3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("wrote fig3_summary.json")
