#!/usr/bin/env python3
"""Build the factorized raw-H0 barcode prior from a feature-joined ledger."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--bc1-oligos", type=Path, required=True)
    parser.add_argument("--bc2-oligos", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.candidate_ledger, args.bc1_oligos, args.bc2_oligos):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    oligos = {
        "BC1": [x.strip() for x in args.bc1_oligos.read_text().splitlines() if x.strip()],
        "BC2": [x.strip() for x in args.bc2_oligos.read_text().splitlines() if x.strip()],
    }
    if any(not values or len(values) != len(set(values)) for values in oligos.values()):
        raise SystemExit("oligo lists must be non-empty and unique")
    counts = {half: [0] * len(values) for half, values in oligos.items()}
    eligible = rows = 0
    opener = gzip.open if args.candidate_ledger.suffix == ".gz" else open
    with opener(args.candidate_ledger, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "candidate_count", "min_tier", "row2", "col2"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit("candidate ledger lacks H0-prior fields")
        seen: set[str] = set()
        for row in reader:
            rows += 1
            if int(row["candidate_count"]) != 1 or int(row["min_tier"]) != 0:
                continue
            if row["read_id"] in seen:
                raise SystemExit(f"duplicate declared-unique read: {row['read_id']}")
            seen.add(row["read_id"])
            col, row2 = int(row["col2"]), int(row["row2"])
            if not 0 <= col < len(counts["BC1"]) or not 0 <= row2 < len(counts["BC2"]):
                raise SystemExit("candidate coordinate outside oligo namespace")
            counts["BC1"][col] += 1
            counts["BC2"][row2] += 1
            eligible += 1
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("barcode_half", "oligo_index", "oligo_sequence", "oligo_length", "exact_h0_read_count"))
        for half in ("BC1", "BC2"):
            for index, sequence in enumerate(oligos[half]):
                writer.writerow((half, index, sequence, len(sequence), counts[half][index]))
    summary = {
        "schema": "visium_hd_processing.hd_h0_prior_from_candidate_ledger.v1",
        "method": "unique_min_tier_zero_feature_joined_reads",
        "counts": {"candidate_rows": rows, "eligible_h0_reads": eligible},
        "inputs": {
            "candidate_ledger": {"path": str(args.candidate_ledger.resolve()), "sha256": sha256(args.candidate_ledger)},
            "bc1_oligos": {"path": str(args.bc1_oligos.resolve()), "sha256": sha256(args.bc1_oligos)},
            "bc2_oligos": {"path": str(args.bc2_oligos.resolve()), "sha256": sha256(args.bc2_oligos)},
        },
        "output": {"path": str(args.out_tsv.resolve()), "sha256": sha256(args.out_tsv)},
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
