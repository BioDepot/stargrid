#!/usr/bin/env python3
"""Export a deterministic Space Ranger read ledger for compatibility audits."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
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


def gzip_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def target_genes(path: Path) -> set[str]:
    import h5py

    with h5py.File(path, "r") as handle:
        features = handle["matrix/features"]
        identifiers = [value.decode() for value in features["id"][:]]
        target_sets = features["target_sets"]
        if len(target_sets) != 1:
            raise ValueError("expected exactly one Space Ranger target set")
        indexes = next(iter(target_sets.values()))[:]
    return {identifiers[int(index)] for index in indexes}


def record_to_row(record: object, genes: set[str]) -> dict[str, object]:
    def tag(name: str, default: object = "") -> object:
        return record.get_tag(name) if record.has_tag(name) else default

    unit = str(tag("sb"))
    gene = str(tag("GX"))
    umi = str(tag("UB"))
    xf = int(tag("xf", 0))
    eligible = bool(COORDINATE.fullmatch(unit) and umi and gene in genes and (xf & 1))
    return {
        "read_id": record.query_name,
        "eligible": int(eligible),
        "feature_id": gene,
        "corrected_umi": umi,
        "unit_2um": unit if COORDINATE.fullmatch(unit) else "",
        "probe_id": str(tag("pr")),
        "fx_gene_id": str(tag("fx")),
        "xf": xf,
        "duplicate": int(bool(record.flag & 1024)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-bam", type=Path, required=True)
    parser.add_argument("--vendor-target-h5", type=Path, required=True)
    parser.add_argument("--expected-primary-reads", type=int, default=100000)
    parser.add_argument("--out-ledger", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.vendor_bam, args.vendor_target_h5):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_ledger.exists() or args.summary_json.exists():
        raise SystemExit("refusing to overwrite vendor-ledger output")

    import pysam

    genes = target_genes(args.vendor_target_h5)
    rows = []
    with pysam.AlignmentFile(args.vendor_bam, "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_secondary or record.is_supplementary:
                continue
            rows.append(record_to_row(record, genes))
    if len(rows) != args.expected_primary_reads:
        raise SystemExit(
            f"vendor BAM has {len(rows)} primary reads, expected {args.expected_primary_reads}"
        )
    rows.sort(key=lambda row: str(row["read_id"]))
    if len({str(row["read_id"]) for row in rows}) != len(rows):
        raise SystemExit("duplicate primary read names in vendor BAM")

    args.out_ledger.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "read_id", "eligible", "feature_id", "corrected_umi", "unit_2um",
        "probe_id", "fx_gene_id", "xf", "duplicate",
    )
    with gzip_writer(args.out_ledger) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    eligible = [row for row in rows if row["eligible"]]
    summary = {
        "schema": "visium_hd_processing.space_ranger_read_ledger.v1",
        "role": "post_seal_vendor_compatibility_diagnostic",
        "counts": {
            "primary_reads": len(rows),
            "matrix_eligible_reads": len(eligible),
            "matrix_eligible_duplicate_flagged_reads": sum(int(row["duplicate"]) for row in eligible),
            "matrix_eligible_probe_specific_families": len({
                (row["unit_2um"], row["feature_id"], row["corrected_umi"], row["probe_id"])
                for row in eligible
            }),
            "matrix_eligible_molecule_families": len({
                (row["unit_2um"], row["feature_id"], row["corrected_umi"])
                for row in eligible
            }),
        },
        "eligible_xf": dict(sorted(Counter(str(row["xf"]) for row in eligible).items())),
        "inputs": {
            str(args.vendor_bam.resolve()): sha256(args.vendor_bam),
            str(args.vendor_target_h5.resolve()): sha256(args.vendor_target_h5),
        },
        "output": {"path": str(args.out_ledger.resolve()), "sha256": sha256(args.out_ledger)},
        "invariants": {
            "one_row_per_primary_read": True,
            "secondary_and_supplementary_records_excluded": True,
            "vendor_not_used_for_open_assignment": True,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
