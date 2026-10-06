#!/usr/bin/env python3
"""Source data for Fig. 3 (policy difference) from the July 2026 audit.

Writes figures/data/fig3_edit_joint.tsv (closest-square edits x Space Ranger
square edits, all reads whose Space Ranger square is not a closest square)
and figures/data/fig3_summary.json (partition, geometry and parent relation
copied from the sealed audit summary).
"""
import collections
import csv
import gzip
import json
import sys

LEDGER = "<runs>/20260720_vm2jxxk_full_sr_disagreement_raw_edit_audit_v1/nonminimum_raw_edit_ledger.tsv.gz"
SUMMARY = "<runs>/20260720_vm2jxxk_full_sr_disagreement_paper_taxonomy_v1/summary.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/data"

joint = collections.Counter()
n = 0
with gzip.open(LEDGER, "rt") as fh:
    rd = csv.reader(fh, delimiter="\t")
    head = next(rd)
    i_min, i_sr = head.index("candidate_min_whole_edit"), head.index("sr_whole_edit")
    for row in rd:
        joint[(int(row[i_min]), int(row[i_sr]))] += 1
        n += 1
with open(f"{OUT}/fig3_edit_joint.tsv", "w") as out:
    out.write("closest_edits\tspace_ranger_edits\treads\n")
    for (a, b), c in sorted(joint.items()):
        out.write(f"{a}\t{b}\t{c}\n")

s = json.load(open(SUMMARY))
keep = {"headline": s["headline"], "parent_relation": s["raw_edit_strata"]["parent_relation"],
        "sr_edit_class": s["raw_edit_strata"]["sr_edit_class"], "ledger_rows": n,
        "sources": {"ledger": LEDGER, "summary": SUMMARY}}
json.dump(keep, open(f"{OUT}/fig3_summary.json", "w"), indent=1)
print("rows", n, "joint cells", len(joint))
