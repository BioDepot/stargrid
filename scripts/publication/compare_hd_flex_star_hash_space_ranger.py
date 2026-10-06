#!/usr/bin/env python3
"""Compare hash-derived HD Flex hard assignments with Space Ranger reads."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


COORDINATE = re.compile(r"^s_002um_(\d+)_(\d+)(?:-\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate(value: str) -> tuple[int, int] | None:
    match = COORDINATE.fullmatch(value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def target_genes(path: Path) -> set[str]:
    import h5py
    with h5py.File(path, "r") as handle:
        features = handle["matrix/features"]
        identifiers = [value.decode() for value in features["id"][:]]
        sets = features["target_sets"]
        if len(sets) != 1:
            raise ValueError("expected one Space Ranger target set")
        indexes = next(iter(sets.values()))[:]
    return {identifiers[int(index)] for index in indexes}


def load_hash_reads(path: Path) -> dict[str, dict[str, str]]:
    result = {}
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


def load_open_reads(path: Path, umi_mode: str, product: str):
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"umi_mode", "product", "feature_id", "corrected_umi", "unit_2um", "member_read_ids"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("open molecule table has the wrong schema")
        for row in reader:
            if row["umi_mode"] != umi_mode or row["product"] != product:
                continue
            unit = coordinate(row["unit_2um"])
            if unit is None:
                raise ValueError(f"invalid open coordinate: {row['unit_2um']}")
            value = {"gene": row["feature_id"], "umi": row["corrected_umi"], "coordinate": unit}
            for read_id in row["member_read_ids"].split(";"):
                if read_id in result and result[read_id] != value:
                    raise ValueError(f"conflicting open assignment: {read_id}")
                result[read_id] = value
    return result


def load_vendor_reads(path: Path, genes: set[str]):
    import pysam
    result = {}
    with pysam.AlignmentFile(path, "rb", check_sq=False) as handle:
        for record in handle.fetch(until_eof=True):
            if record.is_secondary or record.is_supplementary:
                continue
            read_id = record.query_name
            if read_id in result:
                raise ValueError(f"duplicate Space Ranger primary read: {read_id}")
            unit = coordinate(str(record.get_tag("sb"))) if record.has_tag("sb") else None
            gene = str(record.get_tag("GX")) if record.has_tag("GX") else ""
            umi = str(record.get_tag("UB")) if record.has_tag("UB") else ""
            xf = int(record.get_tag("xf")) if record.has_tag("xf") else 0
            result[read_id] = {
                "gene": gene, "umi": umi, "coordinate": unit,
                "eligible": bool(unit and umi and gene in genes and (xf & 1)),
            }
    return result


def compare(hash_reads, open_reads, vendor_reads, feature_lineage=None) -> dict[str, object]:
    universe = set(hash_reads)
    if set(open_reads) - universe or set(vendor_reads) - universe:
        raise ValueError("assignment read IDs lie outside the hash-ledger universe")
    vendor = {read_id: row for read_id, row in vendor_reads.items() if row["eligible"]}
    open_set, vendor_set = set(open_reads), set(vendor)
    shared = open_set & vendor_set
    categories = {
        "shared": len(shared), "open_only": len(open_set - vendor_set),
        "space_ranger_only": len(vendor_set - open_set),
        "neither": len(universe - (open_set | vendor_set)),
    }
    metrics = Counter()
    for read_id in shared:
        left, right = open_reads[read_id], vendor[read_id]
        metrics["gene"] += left["gene"] == right["gene"]
        metrics["corrected_umi"] += left["umi"] == right["umi"]
        lc, rc = left["coordinate"], right["coordinate"]
        metrics["coordinate_2um"] += lc == rc
        metrics["coordinate_8um"] += (lc[0] // 4, lc[1] // 4) == (rc[0] // 4, rc[1] // 4)
        metrics["coordinate_16um"] += (lc[0] // 8, lc[1] // 8) == (rc[0] // 8, rc[1] // 8)
    hash_keep = {read_id: row for read_id, row in hash_reads.items() if row["action"] == "keep"}
    hash_vendor = set(hash_keep) & vendor_set
    tier_counts = {}
    for tier in sorted({row["hash_tier"] for row in hash_keep.values()}):
        reads = {read_id for read_id in hash_vendor if hash_keep[read_id]["hash_tier"] == tier}
        tier_counts[tier] = {
            "reads": len(reads),
            "gene_concordant": sum(hash_keep[read_id]["feature_id"] == vendor[read_id]["gene"] for read_id in reads),
        }
    result = {
        "counts": {
            "raw_read_pairs": len(universe), "open_hard_reads": len(open_set),
            "space_ranger_eligible_reads": len(vendor_set), **categories,
            "hash_keep_reads": len(hash_keep),
            "hash_keep_space_ranger_eligible_reads": len(hash_vendor),
            "hash_keep_space_ranger_gene_concordant_reads": sum(
                hash_keep[read_id]["feature_id"] == vendor[read_id]["gene"] for read_id in hash_vendor
            ),
        },
        "shared_metrics": dict(sorted(metrics.items())),
        "hash_tier_space_ranger_gene": tier_counts,
    }
    if feature_lineage is not None:
        if set(feature_lineage) != universe:
            raise ValueError("feature-lineage and hash-ledger universes differ")
        source_vendor = {}
        source_open_shared = {}
        sources = sorted({row["feature_source"] for row in feature_lineage.values() if row["action"] == "keep"})
        for source in sources:
            reads = {
                read_id for read_id in vendor_set
                if feature_lineage[read_id]["action"] == "keep"
                and feature_lineage[read_id]["feature_source"] == source
            }
            source_vendor[source] = {
                "reads": len(reads),
                "gene_concordant": sum(
                    feature_lineage[read_id]["feature_id"] == vendor[read_id]["gene"]
                    for read_id in reads
                ),
            }
            shared_reads = reads & shared
            source_open_shared[source] = {
                "reads": len(shared_reads),
                "gene_concordant": sum(
                    open_reads[read_id]["gene"] == vendor[read_id]["gene"]
                    for read_id in shared_reads
                ),
            }
        result["feature_source_space_ranger_gene"] = source_vendor
        result["feature_source_open_shared_gene"] = source_open_shared
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-read-ledger", type=Path, required=True)
    parser.add_argument("--open-hard-molecules", type=Path, required=True)
    parser.add_argument("--vendor-bam", type=Path, required=True)
    parser.add_argument("--vendor-target-h5", type=Path, required=True)
    parser.add_argument("--feature-lineage-ledger", type=Path)
    parser.add_argument("--raw-candidates", type=Path)
    parser.add_argument("--umi-mode", default="1mm_cr")
    parser.add_argument("--product", default="postcollapse_hard")
    parser.add_argument("--expected-read-pairs", type=int, default=100000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.hash_read_ledger, args.open_hard_molecules, args.vendor_bam, args.vendor_target_h5]
    if args.feature_lineage_ledger:
        inputs.append(args.feature_lineage_ledger)
    if args.raw_candidates:
        inputs.append(args.raw_candidates)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    hashes = load_hash_reads(args.hash_read_ledger)
    open_reads = load_open_reads(args.open_hard_molecules, args.umi_mode, args.product)
    vendor = load_vendor_reads(args.vendor_bam, target_genes(args.vendor_target_h5))
    lineage = load_feature_lineage(args.feature_lineage_ledger) if args.feature_lineage_ledger else None
    result = compare(hashes, open_reads, vendor, lineage)
    if result["counts"]["raw_read_pairs"] != args.expected_read_pairs or len(vendor) != args.expected_read_pairs:
        raise SystemExit("raw/hash/vendor read universes do not match the expected count")
    if args.raw_candidates:
        candidates = set()
        with args.raw_candidates.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                candidates.add(row["read_id"])
        keep = {read_id for read_id, row in hashes.items() if row["action"] == "keep"}
        result["counts"].update({
            "raw_candidate_reads": len(candidates),
            "hash_keep_with_raw_candidate": len(keep & candidates),
            "hash_keep_without_raw_candidate": len(keep - candidates),
        })
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    counts = result["counts"]
    for name in ("shared", "open_only", "space_ranger_only", "neither"):
        rows.append(("assignment_partition", name, counts[name], counts["raw_read_pairs"]))
    for name, value in result["shared_metrics"].items():
        rows.append(("shared_assignment", name, value, counts["shared"]))
    for tier, values in result["hash_tier_space_ranger_gene"].items():
        rows.append((f"hash_{tier}", "gene_concordant", values["gene_concordant"], values["reads"]))
    for source, values in result.get("feature_source_space_ranger_gene", {}).items():
        rows.append((f"feature_source_{source}", "space_ranger_gene_concordant", values["gene_concordant"], values["reads"]))
    for source, values in result.get("feature_source_open_shared_gene", {}).items():
        rows.append((f"feature_source_{source}", "open_shared_gene_concordant", values["gene_concordant"], values["reads"]))
    table = args.out_dir / "concordance.tsv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("scope", "metric", "reads", "denominator", "fraction"))
        for scope, metric, value, denominator in rows:
            writer.writerow((scope, metric, value, denominator, value / denominator if denominator else ""))
    summary = {
        "schema": "visium_hd_processing.flex_star_hash_space_ranger_concordance.v1",
        "method": {"umi_mode": args.umi_mode, "product": args.product},
        **result,
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {"concordance": {"path": str(table.resolve()), "sha256": sha256(table)}},
        "invariants": {
            "raw_partition_reconciles": sum(counts[name] for name in ("shared", "open_only", "space_ranger_only", "neither")) == counts["raw_read_pairs"],
            "vendor_not_used_for_open_assignment": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
