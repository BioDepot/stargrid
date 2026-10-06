#!/usr/bin/env python3
"""Attach STAR alignment-fallback features only to candidate-bearing hash misses.

The hash classifier has already emitted feature-bearing KEEP candidates and a
small PASS-only candidate stream.  This adapter joins the latter to the STAR
BAM without constructing an all-read text ledger.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from merge_hd_star_hash_alignment_features import load_alignment_features


FIELDS = (
    "read_id", "feature_id", "raw_umi", "sr_corrected_umi", "sr_cb", "min_tier",
    "candidate_count", "row2", "col2", "bc1_edit", "bc2_edit", "bc1_obs_len",
    "bc2_obs_len", "tier_profile", "log_sequence_likelihood",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miss-candidates", type=Path, required=True)
    parser.add_argument("--alignment-bam", type=Path, required=True)
    parser.add_argument("--feature-id-list", type=Path, required=True)
    parser.add_argument("--read-id-prefix", default="")
    parser.add_argument("--expected-alignment-reads", type=int, required=True)
    parser.add_argument("--bc1-count", type=int, default=3350)
    parser.add_argument("--bc2-count", type=int, default=3350)
    parser.add_argument("--out-candidates", type=Path, required=True)
    parser.add_argument("--out-h0-prior", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.miss_candidates, args.alignment_bam, args.feature_id_list):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    for path in (args.out_candidates, args.out_h0_prior, args.summary_json):
        if path.exists():
            raise SystemExit(f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    accepted = {
        line.strip() for line in args.feature_id_list.read_text(encoding="ascii").splitlines()
        if line.strip()
    }
    if not accepted:
        raise SystemExit("feature ID list is empty")
    (
        features, conflicts, alignment_records, programs, alignment_stats,
        alignment_read_ids, incomplete_alignment_reads,
    ) = load_alignment_features(args.alignment_bam, accepted)
    if len(alignment_read_ids) != args.expected_alignment_reads:
        raise SystemExit(
            f"alignment BAM has {len(alignment_read_ids)} reads, expected "
            f"{args.expected_alignment_reads} hash PASS reads"
        )
    if incomplete_alignment_reads:
        raise SystemExit(
            f"alignment BAM has {len(incomplete_alignment_reads)} incomplete NH groups"
        )

    bc1_h0 = [0] * args.bc1_count
    bc2_h0 = [0] * args.bc2_count
    counts: Counter[str] = Counter()
    candidate_miss_ids: set[str] = set()
    with gzip.open(args.miss_candidates, "rt", encoding="utf-8", newline="") as source, gzip_writer(
        args.out_candidates,
    ) as destination:
        reader = csv.DictReader(source, fieldnames=FIELDS, delimiter="\t")
        writer = csv.DictWriter(destination, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        pending: list[dict[str, str]] = []
        current = ""

        def flush() -> None:
            nonlocal pending, current
            if not pending:
                return
            counts["candidate_miss_reads"] += 1
            counts["candidate_miss_rows"] += len(pending)
            raw_id = current
            if args.read_id_prefix:
                if not raw_id.startswith(args.read_id_prefix):
                    raise ValueError(f"candidate read lacks declared prefix: {raw_id}")
                raw_id = raw_id[len(args.read_id_prefix):]
            candidate_miss_ids.add(raw_id)
            if raw_id not in alignment_read_ids:
                raise ValueError(f"candidate hash miss absent from alignment BAM: {raw_id}")
            assignment = features.get(raw_id)
            if assignment is None:
                counts["candidate_reads_without_alignment_feature"] += 1
                pending = []
                return
            feature, umi = assignment
            if any(row["raw_umi"] != umi for row in pending):
                raise ValueError(f"raw UMI differs between candidates and STAR BAM: {raw_id}")
            declared = int(pending[0]["candidate_count"])
            if declared != len(pending) or any(int(row["candidate_count"]) != declared for row in pending):
                raise ValueError(f"candidate declaration does not reconcile: {raw_id}")
            for row in pending:
                row["feature_id"] = feature
                writer.writerow(row)
                counts["output_candidate_rows"] += 1
            counts["alignment_feature_candidate_reads"] += 1
            if declared == 1 and int(pending[0]["min_tier"]) == 0:
                col, row2 = int(pending[0]["col2"]), int(pending[0]["row2"])
                if not 0 <= col < args.bc1_count or not 0 <= row2 < args.bc2_count:
                    raise ValueError(f"candidate coordinate outside barcode namespace: {raw_id}")
                bc1_h0[col] += 1
                bc2_h0[row2] += 1
                counts["eligible_h0_reads"] += 1
            pending = []

        for row in reader:
            read_id = row["read_id"]
            if current and read_id != current:
                flush()
            current = read_id
            pending.append(row)
        flush()

    with args.out_h0_prior.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("barcode_half", "oligo_index", "exact_h0_read_count"))
        for index, count in enumerate(bc1_h0):
            writer.writerow(("BC1", index, count))
        for index, count in enumerate(bc2_h0):
            writer.writerow(("BC2", index, count))
    if sum(bc1_h0) != sum(bc2_h0) or sum(bc1_h0) != counts["eligible_h0_reads"]:
        raise SystemExit("alignment-fallback H0 barcode-half mass does not reconcile")

    summary = {
        "schema": "visium_hd_processing.hash_miss_candidate_alignment_join.v1",
        "method": "candidate-bearing STAR hash PASS join to primary alignment feature",
        "counts": {
            **dict(sorted(counts.items())),
            "alignment_bam_records": alignment_records,
            "alignment_conflicting_reads": len(conflicts),
            **alignment_stats,
        },
        "inputs": {
            "miss_candidates": {"path": str(args.miss_candidates.resolve()), "sha256": sha256(args.miss_candidates)},
            "alignment_bam": {"path": str(args.alignment_bam.resolve()), "sha256": sha256(args.alignment_bam)},
            "feature_id_list": {"path": str(args.feature_id_list.resolve()), "sha256": sha256(args.feature_id_list)},
        },
        "star_program_records": programs,
        "outputs": {
            "candidates": {"path": str(args.out_candidates.resolve()), "sha256": sha256(args.out_candidates)},
            "h0_prior": {"path": str(args.out_h0_prior.resolve()), "sha256": sha256(args.out_h0_prior)},
        },
        "invariants": {
            "bam_read_count_equals_hash_pass_count": True,
            "candidate_hash_misses_are_bam_subset": candidate_miss_ids <= alignment_read_ids,
            "complete_reported_NH_sets": True,
            "primary_alignment_only_for_feature_acceptance": True,
        },
        "prohibited_fields_used": {
            "Space_Ranger_GX": False, "Space_Ranger_CB": False,
            "Space_Ranger_UB": False, "Space_Ranger_pr": False,
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
