#!/usr/bin/env python3
"""Create paper-ready strata for the full natural Space Ranger disagreement audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disagreement-audit", type=Path, required=True)
    parser.add_argument("--raw-edit-audit", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"ERROR: output directory exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    disagreement_summary = json.loads((args.disagreement_audit / "summary.json").read_text())
    raw_summary = json.loads((args.raw_edit_audit / "summary.json").read_text())
    candidate_summary = json.loads(args.candidate_summary.read_text())
    dimensions: dict[str, Counter[str]] = {
        "status": Counter(),
        "status_by_tier": Counter(),
        "status_by_candidate_count": Counter(),
        "status_by_shape": Counter(),
        "status_by_short_bc2_overlap": Counter(),
    }
    rows = 0
    for path in sorted((args.disagreement_audit / "disagreement_shards").glob("*.tsv.gz")):
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                status = row["status"]
                dimensions["status"][status] += 1
                dimensions["status_by_tier"][f"{status}|H{row['min_tier']}"] += 1
                dimensions["status_by_candidate_count"][f"{status}|{row['candidate_count']}"] += 1
                dimensions["status_by_shape"][f"{status}|{row['shape_signature']}"] += 1
                dimensions["status_by_short_bc2_overlap"][
                    f"{status}|{row['short_bc2_overlap']}"
                ] += 1
                rows += 1
    expected_rows = int(disagreement_summary["counts"]["disagreement_groups"])
    if rows != expected_rows:
        raise RuntimeError(f"disagreement rows do not reconcile: {rows} != {expected_rows}")

    native = raw_summary["counts"]
    nonminimum = int(dimensions["status"]["sr_selected_nonminimum"])
    unassigned = int(dimensions["status"]["sr_unassigned"])
    short = int(disagreement_summary["counts"]["short_bc2_overlap_groups"])
    selected_minimum = int(disagreement_summary["counts"]["sr_selected_minimum"])
    candidate_reads = int(disagreement_summary["counts"]["read_groups"])
    star_no_candidate = int(candidate_summary["counts"]["star_no_candidate_reads"])
    shared_eligible = int(candidate_summary["counts"]["shared_eligible"])
    raw_greater = int(native["sr_tier_relation"]["greater"])
    raw_equal = int(native["sr_tier_relation"]["equal"])
    raw_half_outside = sum(
        int(count) for distance, count in native["sr_split_max_half_histogram"].items()
        if int(distance) > 2
    )

    headline = {
        "shared_eligible_reads": shared_eligible,
        "star_candidate_reads": candidate_reads,
        "star_no_candidate_reads": star_no_candidate,
        "sr_selected_retained_minimum": selected_minimum,
        "sr_selected_nonminimum": nonminimum,
        "sr_unassigned_with_star_candidate": unassigned,
        "short_bc2_overlap_nonminimum": short,
        "nonminimum_not_short_bc2_overlap": nonminimum - short,
        "sr_raw_split_cost_greater_than_star_minimum": raw_greater,
        "sr_raw_split_cost_equal_to_star_minimum": raw_equal,
        "sr_raw_split_max_half_above_ed2": raw_half_outside,
    }
    fractions = {
        "star_candidate_fraction_of_shared_eligible": candidate_reads / shared_eligible,
        "sr_minimum_fraction_of_star_candidates": selected_minimum / candidate_reads,
        "sr_nonminimum_fraction_of_star_candidates": nonminimum / candidate_reads,
        "sr_unassigned_fraction_of_star_candidates": unassigned / candidate_reads,
        "short_bc2_fraction_of_sr_nonminimum": short / nonminimum,
        "greater_raw_cost_fraction_of_sr_nonminimum": raw_greater / nonminimum,
        "equal_raw_cost_fraction_of_sr_nonminimum": raw_equal / nonminimum,
        "outside_ed2_half_fraction_of_sr_nonminimum": raw_half_outside / nonminimum,
    }
    summary = {
        "schema": "visium_hd_processing.full_sr_disagreement_taxonomy.v1",
        "headline": headline,
        "fractions": fractions,
        "strata": {name: dict(sorted(values.items())) for name, values in dimensions.items()},
        "raw_edit_strata": {
            name: native[name]
            for name in (
                "tier_histogram", "candidate_count_histogram", "sr_split_sum_histogram",
                "sr_split_max_half_histogram", "sr_split_offset_histogram",
                "sr_split_pair_histogram", "sr_tier_relation", "sr_whole_relation",
                "sr_edit_class", "parent_relation",
            )
        },
        "interpretation_boundary": (
            "Categories describe observed raw evidence and emitted coordinates. They do not assert "
            "Space Ranger's internal correction algorithm."
        ),
        "inputs": {
            "disagreement_summary": str((args.disagreement_audit / "summary.json").resolve()),
            "raw_edit_summary": str((args.raw_edit_audit / "summary.json").resolve()),
            "candidate_summary": str(args.candidate_summary.resolve()),
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.out_dir / "taxonomy.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("dimension", "category", "count", "denominator", "fraction"))
        for dimension, values in dimensions.items():
            for category, count in sorted(values.items()):
                writer.writerow((dimension, category, count, rows, count / rows))
        for dimension in summary["raw_edit_strata"]:
            for category, count in sorted(summary["raw_edit_strata"][dimension].items()):
                writer.writerow((f"raw_{dimension}", category, count, nonminimum, count / nonminimum))
    taxonomy_path = args.out_dir / "taxonomy.tsv"
    (args.out_dir / "checksums.sha256").write_text(
        f"{sha256(summary_path)}  summary.json\n{sha256(taxonomy_path)}  taxonomy.tsv\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
