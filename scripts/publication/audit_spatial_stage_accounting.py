#!/usr/bin/env python3
"""Compare release-independent spatial accounting in two sealed native runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_sealed_spatial_primary import sha256


DECODING_FIELDS = (
    "reads_decoded", "reads_with_candidates", "exact_h0_reads",
    "barcode_reads_with_n", "barcode_n_bases", "barcode_dp_recovered_reads",
    "barcode_dp_ambiguous_reads", "barcode_dp_unassigned_reads",
    "barcode_unsupported_reads", "umi_reads_with_n", "umi_reads_with_invalid_base",
)
FLEX_FIELDS = ("feature_hash_h0", "feature_axis_sha256")


def load(root):
    seal_path = root / "PRIMARY_SEAL.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("status") != "sealed" or seal != json.loads((root / "RECIPE_COMPLETE.json").read_text()):
        raise ValueError(f"run lacks a matching completed seal: {root}")
    found = [p for p in (root / "star/SpatialFlex.out", root / "starSpatialGex.out")
             if (p / "RUN_COMPLETE").is_file()]
    if len(found) != 1:
        raise ValueError(f"expected one complete native spatial output: {root}")
    path = found[0] / "run_summary.tsv"
    if sha256(path) != seal["native_summary_sha256"]:
        raise ValueError(f"native summary hash drift: {root}")
    pairs = [line.split("\t", 1) for line in path.read_text().splitlines()]
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("duplicate native accounting field")
    if values["source_revision"] != seal.get("star_source_revision", seal["star_commit"]):
        raise ValueError("native summary source differs from seal")
    return values, {"primary": str(root.resolve()), "seal_sha256": sha256(seal_path),
                    "summary_path": str(path), "summary_sha256": sha256(path)}


def audit(baseline_root, primary_root):
    old, old_input = load(baseline_root)
    new, new_input = load(primary_root)
    if old["schema"] != new["schema"]:
        raise ValueError("cannot compare accounting across different spatial assays")
    fields = DECODING_FIELDS + (FLEX_FIELDS if "spatial_flex" in new["schema"] else ())
    missing = {key for key in fields if key not in old or key not in new}
    if missing:
        raise ValueError(f"missing invariant fields: {sorted(missing)}")
    mismatches = {key: {"baseline": old[key], "primary": new[key]}
                  for key in fields if old[key] != new[key]}
    return {"schema": "visium_hd_processing.spatial_stage_accounting.v1",
            "passed": not mismatches, "invariant_fields": fields,
            "mismatches": mismatches, "inputs": {"baseline": old_input, "primary": new_input},
            "stage_values": {"baseline": old, "primary": new},
            "scope": "raw spatial decoding, invalid UMIs, and exact Flex feature matches on the same reference axis; non-exact feature decisions may change"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing audit: {args.out}")
    result = audit(args.baseline, args.primary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
