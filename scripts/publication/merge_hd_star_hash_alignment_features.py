#!/usr/bin/env python3
"""Build the native Flex H0/H1-hash plus STAR-alignment feature surface.

Hash KEEP is authoritative, hash DENY remains denied, and only hash PASS/miss
reads may fall through to an unambiguous STAR alignment feature.  Space Ranger
fields are neither read nor accepted by this adapter.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def load_hash_ledger(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "action", "hash_tier", "cache_class", "feature_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("hash ledger has the wrong schema")
        for row in reader:
            if row["read_id"] in result:
                raise ValueError(f"duplicate hash-ledger read: {row['read_id']}")
            result[row["read_id"]] = row
    return result


def load_hash_features(path: Path) -> dict[str, tuple[str, str, str]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "feature_id", "raw_umi", "hash_tier"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("hash feature table has the wrong schema")
        for row in reader:
            read_id = row["read_id"]
            value = (row["feature_id"], row["raw_umi"], row["hash_tier"])
            if read_id in result and result[read_id] != value:
                raise ValueError(f"conflicting hash feature: {read_id}")
            result[read_id] = value
    return result


def alignment_record_is_feature_evidence(record: object) -> bool:
    """Preserve the established primary-record feature acceptance surface."""
    return not record.is_unmapped and not record.is_secondary and not record.is_supplementary


def reported_alignment_set_is_complete(mapped_records: int, nh_values: set[int]) -> bool:
    """Require the BAM to contain every alignment counted by STAR's NH tag."""
    return mapped_records == 0 or (
        len(nh_values) == 1 and next(iter(nh_values)) == mapped_records
    )


def load_alignment_features(
    path: Path, accepted_features: set[str],
) -> tuple[
    dict[str, tuple[str, str]], set[str], int, list[dict[str, object]],
    dict[str, int], set[str], set[str],
]:
    import pysam

    features: dict[str, tuple[str, str]] = {}
    conflicts: set[str] = set()
    alignments = 0
    records_per_read: Counter[str] = Counter()
    mapped_records_per_read: Counter[str] = Counter()
    nh_per_read: dict[str, set[int]] = defaultdict(set)
    primary_feature_reads: set[str] = set()
    secondary_feature_reads: set[str] = set()
    record_classes: Counter[str] = Counter()
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        program = [dict(row) for row in bam.header.to_dict().get("PG", [])]
        if not any(row.get("PN") == "STAR" or str(row.get("ID", "")).startswith("STAR") for row in program):
            raise ValueError("alignment feature BAM lacks a STAR @PG record")
        for record in bam.fetch(until_eof=True):
            alignments += 1
            records_per_read[record.query_name] += 1
            if record.is_supplementary:
                record_classes["supplementary"] += 1
            elif record.is_unmapped:
                record_classes["unmapped"] += 1
            elif record.is_secondary:
                record_classes["secondary_mapped"] += 1
            else:
                record_classes["primary_mapped"] += 1
            if not record.is_unmapped and not record.is_supplementary:
                mapped_records_per_read[record.query_name] += 1
                if record.has_tag("NH"):
                    nh_per_read[record.query_name].add(int(record.get_tag("NH")))
            gene = str(record.get_tag("GX")) if record.has_tag("GX") else ""
            umi = str(record.get_tag("UR")) if record.has_tag("UR") else ""
            if not gene or gene == "-" or gene not in accepted_features or ";" in gene or "," in gene or not umi:
                continue
            (secondary_feature_reads if record.is_secondary else primary_feature_reads).add(
                record.query_name
            )
            # Alternate mappings remain available in the BAM and in the
            # diagnostic counters below.  They do not change the established
            # STAR-Flex feature acceptance surface in this adapter.
            if not alignment_record_is_feature_evidence(record):
                continue
            value = (gene, umi)
            if record.query_name in features and features[record.query_name] != value:
                conflicts.add(record.query_name)
            else:
                features[record.query_name] = value
    for read_id in conflicts:
        features.pop(read_id, None)
    incomplete_nh = {
        read_id for read_id in records_per_read
        if not reported_alignment_set_is_complete(
            mapped_records_per_read[read_id], nh_per_read[read_id],
        )
    }
    stats = {
        "alignment_bam_reads": len(records_per_read),
        "alignment_bam_mapped_reads": sum(
            mapped_records_per_read[read_id] > 0 for read_id in records_per_read
        ),
        "alignment_bam_unmapped_reads": sum(
            mapped_records_per_read[read_id] == 0 for read_id in records_per_read
        ),
        "alignment_bam_primary_mapped_records": record_classes["primary_mapped"],
        "alignment_bam_secondary_mapped_records": record_classes["secondary_mapped"],
        "alignment_bam_unmapped_records": record_classes["unmapped"],
        "alignment_bam_supplementary_records": record_classes["supplementary"],
        "alignment_bam_reads_with_multiple_records": sum(
            value > 1 for value in records_per_read.values()
        ),
        "alignment_bam_max_records_per_read": max(records_per_read.values(), default=0),
        "alignment_bam_reads_with_complete_reported_NH": len(records_per_read) - len(incomplete_nh),
        "alignment_bam_reads_with_incomplete_reported_NH": len(incomplete_nh),
        "alignment_secondary_only_feature_reads": len(
            secondary_feature_reads - primary_feature_reads
        ),
    }
    return features, conflicts, alignments, program, stats, set(records_per_read), incomplete_nh


def merge_assignments(
    ledger: dict[str, dict[str, str]],
    hashes: dict[str, tuple[str, str, str]],
    alignments: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str, str, str]]:
    result = {}
    expected_hashes = {read_id for read_id, row in ledger.items() if row["action"] == "keep"}
    if set(hashes) != expected_hashes:
        raise ValueError("hash feature table does not equal the ledger KEEP set")
    for read_id, row in ledger.items():
        if row["action"] == "keep":
            gene, umi, tier = hashes[read_id]
            if gene != row["feature_id"] or tier != row["hash_tier"]:
                raise ValueError(f"hash table and ledger disagree: {read_id}")
            if read_id in alignments and alignments[read_id] != (gene, umi):
                raise ValueError(f"hash and alignment feature disagree: {read_id}")
            result[read_id] = (gene, umi, f"hash_{tier}", tier)
        elif row["action"] == "miss" and read_id in alignments:
            gene, umi = alignments[read_id]
            result[read_id] = (gene, umi, "alignment_fallback", "")
        elif row["action"] not in {"miss", "deny_probe_ambiguous"}:
            raise ValueError(f"unsupported hash action: {row['action']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-read-ledger", type=Path, required=True)
    parser.add_argument("--hash-feature-table", type=Path, required=True)
    parser.add_argument("--alignment-feature-bam", type=Path, required=True)
    parser.add_argument("--feature-id-list", type=Path, required=True)
    parser.add_argument("--expected-read-count", type=int, default=100000)
    parser.add_argument("--out-feature-table", type=Path, required=True)
    parser.add_argument("--out-read-ledger", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.hash_read_ledger, args.hash_feature_table, args.alignment_feature_bam, args.feature_id_list]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    accepted = {line.strip() for line in args.feature_id_list.read_text().splitlines() if line.strip()}
    if not accepted:
        raise SystemExit("feature ID list is empty")
    ledger = load_hash_ledger(args.hash_read_ledger)
    if len(ledger) != args.expected_read_count:
        raise SystemExit(f"hash ledger has {len(ledger)} reads, expected {args.expected_read_count}")
    hashes = load_hash_features(args.hash_feature_table)
    (
        alignments, conflicts, alignment_rows, programs, alignment_stats,
        alignment_read_ids, incomplete_alignment_reads,
    ) = load_alignment_features(
        args.alignment_feature_bam, accepted,
    )
    expected_alignment_reads = {
        read_id for read_id, row in ledger.items() if row["action"] == "miss"
    }
    if alignment_read_ids != expected_alignment_reads:
        raise SystemExit(
            "alignment BAM and hash-miss universes differ: "
            f"BAM-only={len(alignment_read_ids - expected_alignment_reads)} "
            f"miss-only={len(expected_alignment_reads - alignment_read_ids)}"
        )
    if incomplete_alignment_reads:
        raise SystemExit(
            "alignment BAM does not report complete NH alignment sets for "
            f"{len(incomplete_alignment_reads)} reads"
        )
    merged = merge_assignments(ledger, hashes, alignments)

    with gzip_writer(args.out_feature_table) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("read_id", "feature_id", "raw_umi", "feature_source", "hash_tier"))
        for read_id in sorted(merged):
            writer.writerow((read_id, *merged[read_id]))
    with gzip_writer(args.out_read_ledger) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("read_id", "action", "feature_source", "feature_id"))
        for read_id, row in ledger.items():
            if read_id in merged:
                writer.writerow((read_id, "keep", merged[read_id][2], merged[read_id][0]))
            elif row["action"] == "deny_probe_ambiguous":
                writer.writerow((read_id, "deny_probe_ambiguous", "hash_H1_deny", ""))
            else:
                writer.writerow((read_id, "miss", "alignment_fallback", ""))

    source_counts = Counter(value[2] for value in merged.values())
    summary = {
        "schema": "visium_hd_processing.star_suite_flex_hash_alignment_feature_table.v1",
        "method": "STAR_Suite_H0_H1_hash_then_STAR_alignment_on_hash_PASS",
        "counts": {
            "raw_reads": len(ledger), "feature_reads": len(merged),
            "alignment_feature_reads": len(alignments),
            "alignment_conflicting_reads": len(conflicts),
            "alignment_bam_records": alignment_rows,
            **alignment_stats,
            **dict(sorted(source_counts.items())),
        },
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "star_program_records": programs,
        "outputs": {
            "feature_table": {"path": str(args.out_feature_table.resolve()), "sha256": sha256(args.out_feature_table)},
            "read_ledger": {"path": str(args.out_read_ledger.resolve()), "sha256": sha256(args.out_read_ledger)},
        },
        "invariants": {
            "hash_keep_precedes_alignment": True,
            "hash_deny_never_falls_through": True,
            "alignment_used_only_for_hash_miss": True,
            "primary_alignment_only_for_feature_acceptance": True,
            "secondary_alignments_diagnostic_only": True,
            "alignment_bam_equals_hash_miss_universe": alignment_read_ids == expected_alignment_reads,
            "all_reported_alignment_sets_complete": not incomplete_alignment_reads,
            "hash_alignment_overlap_identical": all(
                read_id not in alignments or alignments[read_id] == hashes[read_id][:2]
                for read_id in hashes
            ),
        },
        "prohibited_fields_used": {
            "Space_Ranger_GX": False, "Space_Ranger_CB": False,
            "Space_Ranger_UB": False, "Space_Ranger_pr": False,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
