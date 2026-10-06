#!/usr/bin/env python3
"""Compare open strict/soft/hard/gated policies on shared vendor-eligible reads."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path


COORDINATE = re.compile(r"^s_002um_(\d+)_(\d+)(?:-\d+)?$")
SCALES = (2, 8, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate(value: str) -> tuple[int, int]:
    match = COORDINATE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid 2 um unit: {value}")
    return int(match.group(1)), int(match.group(2))


def same_parent(left: tuple[int, int], right: tuple[int, int], scale: int) -> bool:
    factor = scale // 2
    return (left[0] // factor, left[1] // factor) == (
        right[0] // factor, right[1] // factor
    )


def load_vendor(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "eligible", "feature_id", "corrected_umi", "unit_2um"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("vendor read ledger has the wrong schema")
        for row in reader:
            if row["eligible"] not in {"1", "true", "True"}:
                continue
            if row["read_id"] in result:
                raise ValueError(f"duplicate eligible vendor read: {row['read_id']}")
            result[row["read_id"]] = {
                "gene": row["feature_id"], "umi": row["corrected_umi"],
                "coordinate": coordinate(row["unit_2um"]),
            }
    return result


def load_products(path: Path, umi_mode: str) -> dict[str, dict[str, dict[str, object]]]:
    products = {name: {} for name in ("strict", "postcollapse_hard", "gated_hard")}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "umi_mode", "product", "feature_id", "corrected_umi", "unit_2um",
            "member_read_ids",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("open molecule ledger has the wrong schema")
        for row in reader:
            if row["umi_mode"] != umi_mode or row["product"] not in products:
                continue
            value = {
                "gene": row["feature_id"], "umi": row["corrected_umi"],
                "coordinate": coordinate(row["unit_2um"]),
            }
            for read_id in row["member_read_ids"].split(";"):
                previous = products[row["product"]].setdefault(read_id, value)
                if previous != value:
                    raise ValueError(f"read belongs to conflicting {row['product']} molecules: {read_id}")
    return products


def load_cliques(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    cliques: dict[str, dict[str, object]] = {}
    member_to_clique = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_clique_id", "member_read_ids", "candidate", "posterior"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("candidate-clique ledger has the wrong schema")
        for row in reader:
            clique_id = row["read_clique_id"]
            members = tuple(sorted(row["member_read_ids"].split(";")))
            entry = cliques.setdefault(clique_id, {"members": members, "candidates": []})
            if entry["members"] != members:
                raise ValueError(f"inconsistent members for clique: {clique_id}")
            entry["candidates"].append((coordinate(row["candidate"]), float(row["posterior"])))
    for clique_id, entry in cliques.items():
        if not math.isclose(sum(value for _, value in entry["candidates"]), 1.0, abs_tol=1e-12):
            raise ValueError(f"posterior does not normalize for clique: {clique_id}")
        for read_id in entry["members"]:
            previous = member_to_clique.setdefault(read_id, clique_id)
            if previous != clique_id:
                raise ValueError(f"read belongs to multiple cliques: {read_id}")
    return cliques, member_to_clique


def evaluate_policies(
    vendor: dict[str, dict[str, object]],
    products: dict[str, dict[str, dict[str, object]]],
    cliques: dict[str, dict[str, object]],
    member_to_clique: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    hard = products["postcollapse_hard"]
    cohort = set(hard) & set(vendor)
    if not cohort:
        raise ValueError("shared hard/vendor cohort is empty")
    if not cohort.issubset(member_to_clique):
        raise ValueError("shared hard reads are missing candidate cliques")
    subsets = {
        "all_shared": cohort,
        "strict_shared": {
            read_id for read_id in cohort
            if len(cliques[member_to_clique[read_id]]["candidates"]) == 1
        },
    }
    subsets["ambiguous_increment"] = cohort - subsets["strict_shared"]

    rows: list[dict[str, object]] = []
    for subset_name, subset in subsets.items():
        denominator = len(subset)
        for policy, assignments in (
            ("strict", products["strict"]),
            ("hard", hard),
            ("gated_hard", products["gated_hard"]),
        ):
            covered = subset & set(assignments)
            for scale in SCALES:
                support = sum(
                    same_parent(assignments[read_id]["coordinate"], vendor[read_id]["coordinate"], scale)
                    for read_id in sorted(covered)
                )
                rows.append({
                    "subset": subset_name, "policy": policy,
                    "metric_type": "hard_coordinate_call", "scale_um": scale,
                    "cohort_reads": denominator, "covered_reads": len(covered),
                    "coverage": len(covered) / denominator if denominator else 0.0,
                    "compatible_support": float(support),
                    "conditional_fraction": support / len(covered) if covered else 0.0,
                    "unconditional_fraction": support / denominator if denominator else 0.0,
                })
        for scale in SCALES:
            read_support = []
            for read_id in sorted(subset):
                candidates = cliques[member_to_clique[read_id]]["candidates"]
                read_support.append(math.fsum(
                    posterior for candidate, posterior in candidates
                    if same_parent(candidate, vendor[read_id]["coordinate"], scale)
                ))
            support = math.fsum(read_support)
            rows.append({
                "subset": subset_name, "policy": "soft_expected",
                "metric_type": "posterior_support", "scale_um": scale,
                "cohort_reads": denominator, "covered_reads": denominator,
                "coverage": 1.0 if denominator else 0.0,
                "compatible_support": support,
                "conditional_fraction": support / denominator if denominator else 0.0,
                "unconditional_fraction": support / denominator if denominator else 0.0,
            })

    identity = {
        "shared_reads": len(cohort),
        "gene_concordant_reads": sum(hard[read_id]["gene"] == vendor[read_id]["gene"] for read_id in cohort),
        "corrected_umi_concordant_reads": sum(hard[read_id]["umi"] == vendor[read_id]["umi"] for read_id in cohort),
        "strict_shared_reads": len(subsets["strict_shared"]),
        "ambiguous_increment_reads": len(subsets["ambiguous_increment"]),
    }
    return rows, identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-molecules", type=Path, required=True)
    parser.add_argument("--candidate-cliques", type=Path, required=True)
    parser.add_argument("--vendor-read-ledger", type=Path, required=True)
    parser.add_argument("--umi-mode", default="1mm_cr")
    parser.add_argument("--expected-shared-reads", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.open_molecules, args.candidate_cliques, args.vendor_read_ledger)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")

    vendor = load_vendor(args.vendor_read_ledger)
    products = load_products(args.open_molecules, args.umi_mode)
    cliques, member_to_clique = load_cliques(args.candidate_cliques)
    rows, identity = evaluate_policies(vendor, products, cliques, member_to_clique)
    if args.expected_shared_reads is not None and identity["shared_reads"] != args.expected_shared_reads:
        raise SystemExit(
            f"shared cohort has {identity['shared_reads']} reads, expected {args.expected_shared_reads}"
        )

    args.out_dir.mkdir(parents=True)
    table = args.out_dir / "shared_policy_concordance.tsv"
    fields = (
        "subset", "policy", "metric_type", "scale_um", "cohort_reads",
        "covered_reads", "coverage", "compatible_support", "conditional_fraction",
        "unconditional_fraction",
    )
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "visium_hd_processing.flex_shared_policy_concordance.v1",
        "umi_mode": args.umi_mode,
        "cohort": "open_postcollapse_hard_intersection_space_ranger_matrix_eligible",
        "identity": identity,
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "output": {"path": str(table.resolve()), "sha256": sha256(table)},
        "invariants": {
            "strict_and_ambiguous_partition_shared_cohort": (
                identity["strict_shared_reads"] + identity["ambiguous_increment_reads"]
                == identity["shared_reads"]
            ),
            "soft_support_preserves_fractional_mass": True,
            "vendor_not_used_for_open_assignment": True,
            "occupancy_jaccard_not_computed": True,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
