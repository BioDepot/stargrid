#!/usr/bin/env python3
"""Export HD Flex features by replaying an existing STAR Suite hash cache.

This implements the accepted no-alignment ``classifyReadH0H1Offset0`` surface:
H0 is checked first, followed by the H1/ambiguous-deny table.  The cache is an
input artifact; this program never rebuilds it from a probe CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import struct
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path


CACHE_HEADER = struct.Struct("<8sHHIQ")
CACHE_RECORD = struct.Struct("<QQIBBH")
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class CacheRecord:
    gene_index: int
    cache_class: int
    negative_code: int
    sample_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-fastq", type=Path, action="append", required=True)
    parser.add_argument("--r2-fastq", type=Path, action="append", required=True)
    parser.add_argument("--hash-cache", type=Path, required=True)
    parser.add_argument("--feature-id-list", type=Path, required=True)
    parser.add_argument("--out-feature-tsv", type=Path, required=True)
    parser.add_argument("--out-read-ledger", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii", newline="")
    return path.open("rt", encoding="ascii", newline="")


def gzip_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def fastq_records(path: Path):
    with open_text(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            sequence = handle.readline().rstrip("\r\n")
            plus = handle.readline()
            quality = handle.readline().rstrip("\r\n")
            if not plus.startswith("+") or len(sequence) != len(quality):
                raise ValueError(f"invalid or truncated FASTQ: {path}")
            read_id = header[1:].split()[0] if header.startswith("@") else ""
            if not read_id:
                raise ValueError(f"invalid FASTQ read name: {path}")
            yield read_id, sequence, quality


def encode_50(sequence: str) -> tuple[int, int] | None:
    if len(sequence) != 50:
        return None
    codes = {"A": 0, "C": 1, "G": 2, "T": 3}
    lo = hi = 0
    for base in sequence.upper():
        code = codes.get(base)
        if code is None:
            return None
        hi = ((hi << 2) & MASK64) | (lo >> 62)
        lo = ((lo << 2) & MASK64) | code
    return lo, hi


def load_cache_hits(path: Path, query_keys: set[tuple[int, int]]):
    h0: dict[tuple[int, int], CacheRecord] = {}
    h1_deny: dict[tuple[int, int], CacheRecord] = {}
    classes: Counter[str] = Counter()
    with path.open("rb") as handle:
        raw_header = handle.read(CACHE_HEADER.size)
        if len(raw_header) != CACHE_HEADER.size:
            raise ValueError("truncated STAR hash-cache header")
        magic, version, kmer_length, record_size, record_count = CACHE_HEADER.unpack(raw_header)
        if magic != b"FH01SEQ1" or version not in (1, 2) or kmer_length != 50 or record_size != 24:
            raise ValueError("unsupported STAR Suite Flex hash-cache format")
        for _ in range(record_count):
            raw = handle.read(CACHE_RECORD.size)
            if len(raw) != CACHE_RECORD.size:
                raise ValueError("truncated STAR hash cache")
            lo, hi, gene, cache_class, negative_code, sample = CACHE_RECORD.unpack(raw)
            classes[f"class_{cache_class}_negative_{negative_code}"] += 1
            key = (lo, hi)
            if key not in query_keys:
                continue
            record = CacheRecord(gene, cache_class, negative_code, sample if version == 2 else 0)
            if cache_class == 0 and gene > 0:
                h0.setdefault(key, record)
            elif cache_class == 1 or (cache_class == 2 and negative_code == 1):
                h1_deny.setdefault(key, record)
        if handle.read(1):
            raise ValueError("STAR hash cache has trailing bytes")
    return h0, h1_deny, record_count, dict(sorted(classes.items()))


def main() -> int:
    args = parse_args()
    if len(args.r1_fastq) != len(args.r2_fastq):
        raise SystemExit("R1/R2 FASTQ counts differ")
    for path in (*args.r1_fastq, *args.r2_fastq, args.hash_cache, args.feature_id_list):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    features = [line.strip() for line in args.feature_id_list.read_text(encoding="ascii").splitlines() if line.strip()]
    if not features or len(features) != len(set(features)):
        raise SystemExit("feature ID list is empty or non-unique")

    reads: list[tuple[str, str, tuple[int, int] | None]] = []
    query_keys: set[tuple[int, int]] = set()
    seen: set[str] = set()
    input_rows = []
    for r1_path, r2_path in zip(args.r1_fastq, args.r2_fastq, strict=True):
        lane_reads = 0
        for left, right in zip_longest(fastq_records(r1_path), fastq_records(r2_path)):
            if left is None or right is None:
                raise ValueError(f"paired FASTQ record counts differ: {r1_path}, {r2_path}")
            if left[0] != right[0]:
                raise ValueError(f"paired FASTQ read names differ: {left[0]} != {right[0]}")
            if left[0] in seen:
                raise ValueError(f"duplicate read ID: {left[0]}")
            seen.add(left[0])
            key = encode_50(right[1][:50])
            if key is not None:
                query_keys.add(key)
            reads.append((left[0], left[1][:9], key))
            lane_reads += 1
        input_rows.append({
            "r1": str(r1_path.resolve()), "r1_sha256": sha256(r1_path),
            "r2": str(r2_path.resolve()), "r2_sha256": sha256(r2_path),
            "read_pairs": lane_reads,
        })

    h0, h1_deny, cache_records, cache_classes = load_cache_hits(args.hash_cache, query_keys)
    counts: Counter[str] = Counter()
    args.out_feature_tsv.parent.mkdir(parents=True, exist_ok=True)
    with gzip_writer(args.out_feature_tsv) as feature_handle, gzip_writer(args.out_read_ledger) as ledger_handle:
        feature_writer = csv.writer(feature_handle, delimiter="\t", lineterminator="\n")
        feature_writer.writerow(("read_id", "feature_id", "raw_umi", "hash_tier"))
        ledger_writer = csv.writer(ledger_handle, delimiter="\t", lineterminator="\n")
        ledger_writer.writerow(("read_id", "action", "hash_tier", "cache_class", "feature_id"))
        for read_id, raw_umi, key in reads:
            action, tier, cache_class, feature_id = "miss", "", "", ""
            record = h0.get(key) if key is not None else None
            if record is not None:
                action, tier, cache_class = "keep", "H0", str(record.cache_class)
            else:
                record = h1_deny.get(key) if key is not None else None
                if record is not None and record.cache_class == 2 and record.negative_code == 1:
                    action, tier, cache_class = "deny_probe_ambiguous", "H1", str(record.cache_class)
                elif record is not None and record.gene_index > 0:
                    action, tier, cache_class = "keep", "H1", str(record.cache_class)
            if action == "keep":
                if record is None or not 1 <= record.gene_index <= len(features):
                    raise ValueError(f"cache gene index outside feature list: {record}")
                feature_id = features[record.gene_index - 1]
                feature_writer.writerow((read_id, feature_id, raw_umi, tier))
                counts[f"keep_{tier.lower()}"] += 1
            else:
                counts[action] += 1
            ledger_writer.writerow((read_id, action, tier, cache_class, feature_id))

    counts["read_pairs"] = len(reads)
    counts["feature_reads"] = counts["keep_h0"] + counts["keep_h1"]
    summary = {
        "schema": "visium_hd_processing.star_suite_flex_hash_feature_table.v1",
        "method": "STAR_Suite_FlexHashScreen_classifyReadH0H1Offset0",
        "inputs": input_rows,
        "hash_cache": {
            "path": str(args.hash_cache.resolve()), "sha256": sha256(args.hash_cache),
            "records": cache_records, "record_classes": cache_classes,
        },
        "feature_id_list": {
            "path": str(args.feature_id_list.resolve()), "sha256": sha256(args.feature_id_list),
            "features": len(features),
        },
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "feature_table": {"path": str(args.out_feature_tsv.resolve()), "sha256": sha256(args.out_feature_tsv)},
            "read_ledger": {"path": str(args.out_read_ledger.resolve()), "sha256": sha256(args.out_read_ledger)},
        },
        "prohibited_fields_used": {
            "probe_csv": False, "Space_Ranger_GX": False, "Space_Ranger_CB": False,
            "Space_Ranger_UB": False, "BAM_alignment": False,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
