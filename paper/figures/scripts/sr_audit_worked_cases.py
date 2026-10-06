#!/usr/bin/env python3
"""Worked cases for the Space Ranger coordinate audit (Table 4).

Selects, in ledger file order, the first two reads of the dominant geometry
(deletion_frame+insertion_frame, short-BC2 overlap, a single minimum square):
one whose two squares share an 8 um bin and one whose squares cross a 16 um
bin. For each, it finds every square at the minimum split-aware edit distance
from the raw barcode (<=2 edits per half) and scores Space Ranger's square
over every split.

Inputs are the audit ledger (Space Ranger CR/CB per read, July 2026 run) and
the slide codebook used by the decoder. BC1 indexes the column and BC2 the
row, both 0-based; the script checks that Space Ranger's recomputed cost
matches the ledger, which confirms the mapping.

Usage: sr_audit_worked_cases.py [LEDGER] [OLIGO_DIR]
"""
import gzip
import sys

LEDGER = (sys.argv[1] if len(sys.argv) > 1 else
          "<runs>/20260720_vm2jxxk_full_sr_disagreement_raw_edit_audit_v1/"
          "nonminimum_raw_edit_ledger.tsv.gz")
OLIGOS = (sys.argv[2] if len(sys.argv) > 2 else
          "<storage>/runs/cleanroom_hd_mouse_brain/slide_oligos")

bc1 = [l.strip() for l in open(f"{OLIGOS}/bc1_full_oligos.txt") if l.strip()]
bc2 = [l.strip() for l in open(f"{OLIGOS}/bc2_full_oligos.txt") if l.strip()]


def lev(a, b, cap):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def sr_cost(cr, o1, o2):
    """Best split for a fixed square, with no per-half limit."""
    return min((lev(cr[:s], o1, 99) + lev(cr[s:], o2, 99), lev(cr[:s], o1, 99), lev(cr[s:], o2, 99), s)
               for s in range(10, len(cr) - 9))


def minimum_set(cr):
    best, hits = 99, []
    for s in range(12, 19):
        h1 = [(i, d) for i, o in enumerate(bc1) if (d := lev(cr[:s], o, 2)) <= 2]
        h2 = [(j, d) for j, o in enumerate(bc2) if (d := lev(cr[s:], o, 2)) <= 2] if h1 else []
        for i, d1 in h1:
            for j, d2 in h2:
                if d1 + d2 < best:
                    best, hits = d1 + d2, []
                if d1 + d2 == best:
                    hits.append((j, i, d1, d2, s))
    return best, sorted(set(hits))


def select_cases():
    cases = {}
    with gzip.open(LEDGER, "rt") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        col = {name: k for k, name in enumerate(head)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if (f[col["sr_edit_class"]] != "deletion_frame+insertion_frame"
                    or f[col["short_bc2_overlap"]] != "1" or f[col["candidate_count"]] != "1"):
                continue
            if f[col["same_8um_parent"]] == "1":
                cases.setdefault("same 8 um bin", f)
            if f[col["same_16um_parent"]] == "0":
                cases.setdefault("crosses 16 um bin", f)
            if len(cases) == 2:
                return col, cases
    raise SystemExit("ledger ended before both cases were found")


col, cases = select_cases()
print("case\tread_id\tsquare\trow\tcol\tsplit\tbc1_oligo\tbc1_edits\tbc2_oligo\tbc2_edits\ttotal")
for label, f in cases.items():
    cr, row, cl = f[col["raw_cr"]], int(f[col["sr_row"]]), int(f[col["sr_col"]])
    total, e1, e2, s = sr_cost(cr, bc1[cl], bc2[row])
    expected = (int(f[col["sr_best_bc1_edit"]]), int(f[col["sr_best_bc2_edit"]]), int(f[col["sr_best_split"]]))
    if (e1, e2, s) != expected:
        raise SystemExit(f"mapping check failed for {f[col['read_id']]}: {(e1, e2, s)} != {expected}")
    best, hits = minimum_set(cr)
    for j, i, d1, d2, sm in hits:
        print(f"{label}\t{f[col['read_id']]}\tclosest\t{j}\t{i}\t{cr[:sm]}|{cr[sm:]}\t{bc1[i]}\t{d1}\t{bc2[j]}\t{d2}\t{best}")
    print(f"{label}\t{f[col['read_id']]}\tSpace Ranger\t{row}\t{cl}\t{cr[:s]}|{cr[s:]}\t{bc1[cl]}\t{e1}\t{bc2[row]}\t{e2}\t{total}")
