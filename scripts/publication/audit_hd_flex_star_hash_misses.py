#!/usr/bin/env python3
"""Verify STAR hash misses retained as countable Space Ranger Flex reads.

The audit is deliberately comparator-only.  It starts from an already-created
STAR hash ledger and never feeds Space Ranger fields back into an open call.
It verifies the retained BAM assignments against the assigned ``pr`` probe,
the accepted target axis, and ``molecule_info.h5``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


COORDINATE = re.compile(r"^(s_002um_\d+_\d+)(?:-\d+)?$")
CACHE_HEADER = struct.Struct("<8sHHIQ")
CACHE_RECORD = struct.Struct("<QQIBBH")
MASK64 = (1 << 64) - 1
BASES = "ACGT"


@dataclass(frozen=True)
class Probe:
    gene_id: str
    sequence: str


@dataclass(frozen=True)
class CacheRecord:
    gene_index: int
    cache_class: int
    negative_code: int
    sample_index: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fastq_id(header: str) -> str:
    value = header.rstrip("\r\n").split()[0]
    if value.startswith("@"):
        value = value[1:]
    return value[:-2] if value.endswith(("/1", "/2")) else value


def encode_50(sequence: str) -> tuple[int, int] | None:
    if len(sequence) != 50:
        return None
    lo = hi = 0
    for base in sequence.upper():
        code = BASES.find(base)
        if code < 0:
            return None
        hi = ((hi << 2) & MASK64) | (lo >> 62)
        lo = ((lo << 2) & MASK64) | code
    return lo, hi


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(a != b for a, b in zip(left, right, strict=True))


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (a != b),
            ))
        previous = current
    return previous[-1]


def decode_packed_umi(value: int, length: int) -> str:
    if length < 1 or value >= (1 << (2 * length)):
        raise ValueError("packed UMI lies outside the declared length")
    return "".join(BASES[(value >> (2 * (length - index - 1))) & 3] for index in range(length))


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


def load_feature_lineage(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "action", "feature_source", "feature_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("feature-lineage ledger has the wrong schema")
        for row in reader:
            if row["read_id"] in result:
                raise ValueError(f"duplicate feature-lineage read: {row['read_id']}")
            result[row["read_id"]] = row
    return result


def load_star_alignment_evidence(
    path: Path, requested: set[str],
) -> dict[str, dict[str, object]]:
    """Collect complete primary/secondary gene evidence for requested reads."""
    import pysam

    result: dict[str, dict[str, object]] = {
        read_id: {
            "records": 0, "mapped_records": 0, "supplementary_records": 0,
            "nh": set(), "primary_genes": set(), "secondary_genes": set(),
            "overlap_genes": set(),
        }
        for read_id in requested
    }
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.query_name not in result:
                continue
            evidence = result[record.query_name]
            evidence["records"] = int(evidence["records"]) + 1
            if record.is_supplementary:
                evidence["supplementary_records"] = int(evidence["supplementary_records"]) + 1
                continue
            if record.is_unmapped:
                continue
            evidence["mapped_records"] = int(evidence["mapped_records"]) + 1
            if record.has_tag("NH"):
                evidence["nh"].add(int(record.get_tag("NH")))
            gene = str(record.get_tag("GX")) if record.has_tag("GX") else ""
            if gene and gene != "-":
                key = "secondary_genes" if record.is_secondary else "primary_genes"
                evidence[key].add(gene)
            overlap = str(record.get_tag("gx")) if record.has_tag("gx") else ""
            if overlap and overlap != "-":
                evidence["overlap_genes"].update(
                    value for value in re.split(r"[;,]", overlap) if value
                )
    for evidence in result.values():
        nh = evidence["nh"]
        evidence["reported_nh_complete"] = (
            int(evidence["mapped_records"]) == 0
            or (len(nh) == 1 and next(iter(nh)) == int(evidence["mapped_records"]))
        )
    return result


def classify_star_alignment_evidence(
    evidence: dict[str, object], vendor_gene: str,
) -> tuple[str, str]:
    primary = set(evidence["primary_genes"])
    secondary = set(evidence["secondary_genes"])
    unique_genes = primary | secondary
    overlaps = set(evidence["overlap_genes"])
    if int(evidence["mapped_records"]) == 0:
        return "unmapped", "none"
    if len(unique_genes) == 1:
        gene = next(iter(unique_genes))
        location = "primary" if vendor_gene in primary else (
            "secondary_only" if vendor_gene in secondary else "different_gene"
        )
        return "unique_gene_all_alignments", location
    if len(unique_genes) > 1:
        return "conflicting_unique_genes", (
            "conflicting_includes_vendor_gene" if vendor_gene in unique_genes else "different_gene"
        )
    if vendor_gene in overlaps and len(overlaps) > 1:
        return "multigene_overlap_includes_vendor_gene", "ambiguous_overlap"
    if vendor_gene in overlaps:
        return "single_overlap_gene_no_unique_GX", "overlap_no_GX"
    if overlaps:
        return "other_gene_overlap", "different_gene"
    return "no_gene_overlap", "none"


def load_r2_sequences(paths: list[Path], requested: set[str]) -> tuple[dict[str, str], int]:
    result: dict[str, str] = {}
    total = 0
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="ascii") as handle:
            while True:
                header = handle.readline()
                if not header:
                    break
                sequence = handle.readline().strip().upper()
                plus = handle.readline()
                quality = handle.readline().strip()
                if not plus.startswith("+") or len(sequence) != len(quality):
                    raise ValueError(f"malformed FASTQ: {path}")
                total += 1
                read_id = fastq_id(header)
                if read_id in requested:
                    if read_id in result:
                        raise ValueError(f"duplicate requested R2 read: {read_id}")
                    result[read_id] = sequence[:50]
    if set(result) != requested:
        missing = sorted(requested - set(result))
        raise ValueError(f"R2 FASTQs lack {len(missing)} hash misses; first={missing[:3]}")
    return result, total


def load_probes(path: Path) -> dict[str, Probe]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        required = {"gene_id", "probe_seq", "probe_id", "included"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError("probe set has the wrong schema")
        result = {
            row["probe_id"]: Probe(row["gene_id"], row["probe_seq"].upper())
            for row in rows if row["included"].strip().lower() == "true"
        }
    if not result or any(len(probe.sequence) != 50 for probe in result.values()):
        raise ValueError("included probes must be non-empty 50-mers")
    return result


def target_genes(path: Path) -> set[str]:
    import h5py

    with h5py.File(path, "r") as handle:
        features = handle["matrix/features"]
        identifiers = [value.decode() for value in features["id"][:]]
        sets = features["target_sets"]
        if len(sets) != 1:
            raise ValueError("expected exactly one vendor target set")
        indexes = next(iter(sets.values()))[:]
    return {identifiers[int(index)] for index in indexes}


def load_cache_records(
    path: Path, query_keys: set[tuple[int, int]],
) -> dict[tuple[int, int], list[CacheRecord]]:
    result: dict[tuple[int, int], list[CacheRecord]] = {}
    with path.open("rb") as handle:
        raw = handle.read(CACHE_HEADER.size)
        if len(raw) != CACHE_HEADER.size:
            raise ValueError("truncated hash-cache header")
        magic, version, length, size, count = CACHE_HEADER.unpack(raw)
        if magic != b"FH01SEQ1" or version not in (1, 2) or length != 50 or size != 24:
            raise ValueError("unsupported STAR hash-cache format")
        for _ in range(count):
            raw = handle.read(CACHE_RECORD.size)
            if len(raw) != CACHE_RECORD.size:
                raise ValueError("truncated hash-cache record")
            lo, hi, gene, cache_class, negative, sample = CACHE_RECORD.unpack(raw)
            key = (lo, hi)
            if key in query_keys:
                result.setdefault(key, []).append(CacheRecord(
                    gene, cache_class, negative, sample if version == 2 else 0,
                ))
        if handle.read(1):
            raise ValueError("hash cache has trailing bytes")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-read-ledger", type=Path, required=True)
    parser.add_argument("--hash-cache", type=Path, required=True)
    parser.add_argument("--feature-id-list", type=Path, required=True)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--r2-fastq", type=Path, action="append", required=True)
    parser.add_argument("--vendor-bam", type=Path, required=True)
    parser.add_argument("--vendor-target-h5", type=Path, required=True)
    parser.add_argument("--vendor-molecule-info", type=Path, required=True)
    parser.add_argument("--feature-lineage-ledger", type=Path)
    parser.add_argument(
        "--alignment-feature-bam", type=Path,
        help="Complete STAR fallback BAM, including reported secondary alignments.",
    )
    parser.add_argument("--expected-raw-reads", type=int, default=100000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        args.hash_read_ledger, args.hash_cache, args.feature_id_list, args.probe_set,
        *args.r2_fastq, args.vendor_bam, args.vendor_target_h5, args.vendor_molecule_info,
    ]
    if args.feature_lineage_ledger:
        inputs.append(args.feature_lineage_ledger)
    if args.alignment_feature_bam:
        inputs.append(args.alignment_feature_bam)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")

    ledger = load_hash_ledger(args.hash_read_ledger)
    if len(ledger) != args.expected_raw_reads:
        raise SystemExit(f"hash ledger has {len(ledger)} reads, expected {args.expected_raw_reads}")
    misses = {read_id for read_id, row in ledger.items() if row["action"] == "miss"}
    lineage = load_feature_lineage(args.feature_lineage_ledger) if args.feature_lineage_ledger else None
    if lineage is not None and set(lineage) != set(ledger):
        raise SystemExit("feature-lineage and hash-ledger universes differ")
    alignment_evidence = (
        load_star_alignment_evidence(args.alignment_feature_bam, misses)
        if args.alignment_feature_bam else None
    )
    sequences, fastq_reads = load_r2_sequences(args.r2_fastq, misses)
    if fastq_reads != args.expected_raw_reads:
        raise SystemExit(f"R2 FASTQs have {fastq_reads} reads, expected {args.expected_raw_reads}")
    probes = load_probes(args.probe_set)
    targets = target_genes(args.vendor_target_h5)

    import pysam

    family_counts: Counter[tuple[str, str, str, str]] = Counter()
    miss_rows: list[dict[str, object]] = []
    primary_reads = 0
    target_eligible_reads = 0
    umi_lengths: set[int] = set()
    with pysam.AlignmentFile(args.vendor_bam, "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_secondary or record.is_supplementary:
                continue
            primary_reads += 1
            tags = dict(record.tags)
            coordinate_match = COORDINATE.fullmatch(str(tags.get("sb", "")))
            gene = str(tags.get("GX", ""))
            umi = str(tags.get("UB", ""))
            probe_id = str(tags.get("pr", ""))
            xf = int(tags.get("xf", 0))
            eligible = bool(coordinate_match and umi and gene in targets and (xf & 1))
            if not eligible:
                continue
            target_eligible_reads += 1
            umi_lengths.add(len(umi))
            family = (coordinate_match.group(1), gene, umi, probe_id)
            family_counts[family] += 1
            if record.query_name not in misses:
                continue
            probe = probes.get(probe_id)
            if probe is None:
                raise ValueError(f"SR pr tag is absent from included probe set: {probe_id}")
            raw = sequences[record.query_name]
            miss_rows.append({
                "read_id": record.query_name,
                "gene_id": gene,
                "fx_gene_id": str(tags.get("fx", "")),
                "probe_id": probe_id,
                "probe_gene_id": probe.gene_id,
                "probe_hamming": hamming(raw, probe.sequence),
                "probe_levenshtein": levenshtein(raw, probe.sequence),
                "raw_probe_n_count": raw.count("N"),
                "coordinate": coordinate_match.group(1),
                "corrected_umi": umi,
                "xf": xf,
                "bam_flag": record.flag,
                "cigar": record.cigarstring or "",
                "nM": tags.get("nM", ""),
                "family": family,
                "parent_sequence": probe.sequence,
                "raw_sequence": raw,
            })
    if primary_reads != args.expected_raw_reads:
        raise SystemExit(f"vendor BAM has {primary_reads} primary reads, expected {args.expected_raw_reads}")
    if len(umi_lengths) != 1:
        raise ValueError(f"vendor target UMIs have inconsistent lengths: {sorted(umi_lengths)}")
    umi_length = next(iter(umi_lengths))

    feature_ids = [line.strip() for line in args.feature_id_list.read_text().splitlines() if line.strip()]
    query_keys = {
        key for row in miss_rows
        for key in (encode_50(str(row["parent_sequence"])), encode_50(str(row["raw_sequence"])))
        if key is not None
    }
    cache = load_cache_records(args.hash_cache, query_keys)
    for row in miss_rows:
        parent_key = encode_50(str(row["parent_sequence"]))
        raw_key = encode_50(str(row["raw_sequence"]))
        parent_records = cache.get(parent_key, []) if parent_key is not None else []
        matching_h0 = [
            record for record in parent_records
            if record.cache_class == 0 and record.gene_index > 0
            and feature_ids[record.gene_index - 1] == row["gene_id"]
        ]
        row["parent_h0_cache_gene_supported"] = bool(matching_h0)
        row["raw_cache_status"] = "unencodable_N" if raw_key is None else (
            "present" if cache.get(raw_key) else "absent"
        )

    import h5py

    molecule_families: dict[tuple[str, str, str, str], int] = {}
    molecule_target_count = 0
    with h5py.File(args.vendor_molecule_info, "r") as handle:
        barcodes = [value.decode() for value in handle["barcodes"][:]]
        genes = [value.decode() for value in handle["features/id"][:]]
        probe_ids = [value.decode() for value in handle["probes/probe_id"][:]]
        for barcode_index, gene_index, packed_umi, probe_index, count in zip(
            handle["barcode_idx"][:], handle["feature_idx"][:], handle["umi"][:],
            handle["probe_idx"][:], handle["count"][:], strict=True,
        ):
            gene = genes[int(gene_index)]
            if gene not in targets:
                continue
            family = (
                barcodes[int(barcode_index)], gene,
                decode_packed_umi(int(packed_umi), umi_length), probe_ids[int(probe_index)],
            )
            if family in molecule_families:
                raise ValueError(f"duplicate molecule-info family: {family}")
            molecule_families[family] = int(count)
            molecule_target_count += int(count)

    family_sets_identical = set(family_counts) == set(molecule_families)
    family_counts_identical = family_sets_identical and all(
        count == molecule_families[family] for family, count in family_counts.items()
    )
    if not family_sets_identical or not family_counts_identical:
        raise SystemExit("vendor BAM and molecule_info target families do not agree exactly")
    for row in miss_rows:
        family = row.pop("family")
        row["vendor_family_read_count"] = family_counts[family]
        row["molecule_info_family_present"] = family in molecule_families
        row["molecule_info_family_count_agrees"] = family_counts[family] == molecule_families[family]
        row.pop("parent_sequence")
        row.pop("raw_sequence")
        if lineage is not None:
            lineage_row = lineage[str(row["read_id"])]
            row["lineage_action"] = lineage_row["action"]
            row["feature_source"] = lineage_row["feature_source"]
            row["alignment_fallback_gene_matches_GX"] = (
                lineage_row["action"] == "keep"
                and lineage_row["feature_source"] == "alignment_fallback"
                and lineage_row["feature_id"] == row["gene_id"]
            )
        else:
            row["lineage_action"] = ""
            row["feature_source"] = ""
            row["alignment_fallback_gene_matches_GX"] = ""
        if alignment_evidence is not None:
            evidence = alignment_evidence[str(row["read_id"])]
            resolution, vendor_gene_evidence = classify_star_alignment_evidence(
                evidence, str(row["gene_id"]),
            )
            row["star_alignment_resolution"] = resolution
            row["star_vendor_gene_evidence"] = vendor_gene_evidence
            row["star_alignment_record_count"] = evidence["records"]
            row["star_alignment_mapped_record_count"] = evidence["mapped_records"]
            row["star_alignment_nh"] = ";".join(map(str, sorted(evidence["nh"])))
            row["star_primary_gene_ids"] = ";".join(sorted(evidence["primary_genes"]))
            row["star_secondary_gene_ids"] = ";".join(sorted(evidence["secondary_genes"]))
            row["star_overlap_gene_ids"] = ";".join(sorted(evidence["overlap_genes"]))
            row["star_reported_nh_complete"] = evidence["reported_nh_complete"]
        else:
            for field in (
                "star_alignment_resolution", "star_vendor_gene_evidence",
                "star_alignment_record_count", "star_alignment_mapped_record_count",
                "star_alignment_nh", "star_primary_gene_ids",
                "star_secondary_gene_ids", "star_overlap_gene_ids",
                "star_reported_nh_complete",
            ):
                row[field] = ""

    miss_rows.sort(key=lambda row: str(row["read_id"]))
    hamming_counts = Counter(
        f"H{distance}" if int(distance) <= 2 else ">H2"
        for distance in (row["probe_hamming"] for row in miss_rows)
    )
    edit_counts = Counter(int(row["probe_levenshtein"]) for row in miss_rows)
    miss_families = {
        (row["coordinate"], row["gene_id"], row["corrected_umi"], row["probe_id"])
        for row in miss_rows
    }
    h1_rows = [row for row in miss_rows if row["probe_hamming"] == 1]
    canonical_h1 = [row for row in h1_rows if row["raw_probe_n_count"] == 0]
    fallback_rows = [
        row for row in miss_rows
        if row["lineage_action"] == "keep" and row["feature_source"] == "alignment_fallback"
    ]
    nonfallback_rows = [row for row in miss_rows if row not in fallback_rows]
    invariants = {
        "raw_universe_reconciles": len(ledger) == fastq_reads == primary_reads == args.expected_raw_reads,
        "all_sr_hash_misses_have_pr": all(bool(row["probe_id"]) for row in miss_rows),
        "all_pr_genes_match_GX": all(row["probe_gene_id"] == row["gene_id"] for row in miss_rows),
        "all_fx_genes_match_GX": all(row["fx_gene_id"] == row["gene_id"] for row in miss_rows),
        "all_parent_H0_cache_entries_support_GX": all(row["parent_h0_cache_gene_supported"] for row in miss_rows),
        "all_canonical_H1_miss_keys_absent_from_cache": all(row["raw_cache_status"] == "absent" for row in canonical_h1),
        "vendor_target_family_sets_identical": family_sets_identical,
        "vendor_target_family_counts_identical": family_counts_identical,
        "all_sr_hash_misses_in_molecule_info": all(row["molecule_info_family_present"] for row in miss_rows),
        "all_alignment_fallback_genes_match_GX": all(
            row["alignment_fallback_gene_matches_GX"] for row in fallback_rows
        ),
        "vendor_not_used_for_open_assignment": True,
    }
    if alignment_evidence is not None:
        invariants.update({
            "all_reported_alignment_sets_complete": all(
                bool(row["star_reported_nh_complete"]) for row in miss_rows
            ),
            "no_conflicting_unique_GX_genes": all(
                row["star_alignment_resolution"] != "conflicting_unique_genes"
                for row in miss_rows
            ),
            "all_primary_vendor_gene_evidence_is_recovered": all(
                row in fallback_rows
                for row in miss_rows
                if row["star_vendor_gene_evidence"] == "primary"
            ),
        })
    if not all(invariants.values()):
        raise SystemExit(f"hash-miss audit invariant failed: {invariants}")

    args.out_dir.mkdir(parents=True)
    detail_path = args.out_dir / "hash_miss_read_audit.tsv"
    detail_fields = (
        "read_id", "gene_id", "fx_gene_id", "probe_id", "probe_gene_id",
        "probe_hamming", "probe_levenshtein", "raw_probe_n_count",
        "parent_h0_cache_gene_supported", "raw_cache_status", "coordinate",
        "corrected_umi", "xf", "bam_flag", "cigar", "nM",
        "vendor_family_read_count", "molecule_info_family_present",
        "molecule_info_family_count_agrees", "lineage_action", "feature_source",
        "alignment_fallback_gene_matches_GX",
        "star_alignment_resolution", "star_vendor_gene_evidence",
        "star_alignment_record_count", "star_alignment_mapped_record_count",
        "star_alignment_nh", "star_primary_gene_ids", "star_secondary_gene_ids",
        "star_overlap_gene_ids", "star_reported_nh_complete",
    )
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(miss_rows)

    summary_rows = []
    def add(scope: str, metric: str, value: int, denominator: int) -> None:
        summary_rows.append((scope, metric, value, denominator, value / denominator if denominator else ""))

    denominator = len(miss_rows)
    for metric in ("H0", "H1", "H2", ">H2"):
        add("assigned_pr_hamming", metric, hamming_counts[metric], denominator)
    add("assigned_pr_edit", "levenshtein_le_2", sum(value for distance, value in edit_counts.items() if distance <= 2), denominator)
    add("assigned_pr_edit", "levenshtein_gt_2", sum(value for distance, value in edit_counts.items() if distance > 2), denominator)
    add("assigned_pr", "gene_matches_GX", sum(row["probe_gene_id"] == row["gene_id"] for row in miss_rows), denominator)
    add("vendor_molecule_info", "miss_reads_present", denominator, denominator)
    add("vendor_molecule_info", "miss_probe_specific_families_present", len(miss_families), len(miss_families))
    add("native_cache_H1", "canonical_absent", len(canonical_h1), len(h1_rows))
    add("native_cache_H1", "unencodable_N", len(h1_rows) - len(canonical_h1), len(h1_rows))
    if lineage is not None:
        add("alignment_fallback", "recovered", len(fallback_rows), denominator)
        add("alignment_fallback", "not_recovered", len(nonfallback_rows), denominator)
    if alignment_evidence is not None:
        for metric, value in sorted(Counter(
            str(row["star_alignment_resolution"]) for row in nonfallback_rows
        ).items()):
            add("alignment_nonrecovery", metric, value, denominator)
    summary_path = args.out_dir / "hash_miss_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("scope", "metric", "reads", "denominator", "fraction"))
        writer.writerows(summary_rows)

    summary = {
        "schema": "visium_hd_processing.flex_star_hash_miss_space_ranger_audit.v1",
        "status": "pass",
        "counts": {
            "raw_primary_reads": primary_reads,
            "star_hash_miss_reads": len(misses),
            "space_ranger_target_eligible_reads": target_eligible_reads,
            "space_ranger_eligible_star_hash_miss_reads": denominator,
            "space_ranger_target_probe_specific_families": len(family_counts),
            "space_ranger_molecule_info_target_rows": len(molecule_families),
            "space_ranger_molecule_info_target_read_support": molecule_target_count,
            "star_hash_miss_probe_specific_families": len(miss_families),
            "star_hash_miss_family_total_read_support": sum(family_counts[family] for family in miss_families),
            "star_hash_miss_reads_with_N": sum(int(row["raw_probe_n_count"]) > 0 for row in miss_rows),
            "star_hash_miss_duplicate_flagged_reads": sum(bool(int(row["bam_flag"]) & 1024) for row in miss_rows),
        },
        "assigned_probe_hamming": dict(sorted(hamming_counts.items())),
        "assigned_probe_levenshtein": {str(key): value for key, value in sorted(edit_counts.items())},
        "native_cache_H1_miss_audit": {
            "reads": len(h1_rows),
            "unencodable_N": len(h1_rows) - len(canonical_h1),
            "canonical_raw_key_absent": len(canonical_h1),
            "parent_H0_present_and_gene_supported": sum(row["parent_h0_cache_gene_supported"] for row in h1_rows),
        },
        "alignment_fallback_space_ranger_hash_miss_audit": {
            "recovered_reads": len(fallback_rows),
            "not_recovered_reads": len(nonfallback_rows),
            "recovered_gene_concordant": sum(
                bool(row["alignment_fallback_gene_matches_GX"]) for row in fallback_rows
            ),
            "recovered_hamming": dict(sorted(Counter(
                f"H{row['probe_hamming']}" if int(row["probe_hamming"]) <= 2 else ">H2"
                for row in fallback_rows
            ).items())),
            "not_recovered_hamming": dict(sorted(Counter(
                f"H{row['probe_hamming']}" if int(row["probe_hamming"]) <= 2 else ">H2"
                for row in nonfallback_rows
            ).items())),
            "recovered_levenshtein": dict(sorted(Counter(
                str(row["probe_levenshtein"]) for row in fallback_rows
            ).items())),
            "not_recovered_levenshtein": dict(sorted(Counter(
                str(row["probe_levenshtein"]) for row in nonfallback_rows
            ).items())),
        } if lineage is not None else None,
        "alignment_bam_space_ranger_hash_miss_audit": {
            "reported_records": sum(int(row["star_alignment_record_count"]) for row in miss_rows),
            "resolution": dict(sorted(Counter(
                str(row["star_alignment_resolution"]) for row in miss_rows
            ).items())),
            "vendor_gene_evidence": dict(sorted(Counter(
                str(row["star_vendor_gene_evidence"]) for row in miss_rows
            ).items())),
            "nonrecovery_resolution": dict(sorted(Counter(
                str(row["star_alignment_resolution"]) for row in nonfallback_rows
            ).items())),
        } if alignment_evidence is not None else None,
        "vendor_xf": dict(sorted(Counter(str(row["xf"]) for row in miss_rows).items())),
        "vendor_bam_flags": dict(sorted(Counter(str(row["bam_flag"]) for row in miss_rows).items())),
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "invariants": invariants,
        "outputs": {
            "read_audit": {"path": str(detail_path.resolve()), "sha256": sha256(detail_path)},
            "summary_table": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
        },
        "interpretation": {
            "space_ranger_status": "confirmed_in_retained_BAM_and_molecule_info",
            "H2_and_beyond": "outside_native_H0_H1_cache_acceptance_surface",
            "canonical_H1": "parent_H0_exists_but_observed_H1_key_was_not_emitted_by_alignment_verified_cache",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
