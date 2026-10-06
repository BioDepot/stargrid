#!/usr/bin/env python3
"""Extract primary accounting and policy masses from a sealed native STAR run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

POLICIES = ("strict", "soft_expected", "hard", "gated_hard")
SCALES = ("square_002um", "square_008um", "square_016um")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_rows(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("product\tscale\t"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:])), delimiter="\t"))
    if {(r["product"], r["scale"]) for r in rows} != {(p, s) for p in POLICIES for s in SCALES} or len(rows) != 12:
        raise ValueError("expected four policies at all three spatial scales")
    return rows


def summarize(root, expected_release):
    seal_path = root / "PRIMARY_SEAL.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("status") != "sealed" or seal.get("star_release_tag") != expected_release:
        raise ValueError("primary is not sealed for the required release")
    candidates = [root / "star/SpatialFlex.out", root / "starSpatialGex.out"]
    found = [p for p in candidates if (p / "RUN_COMPLETE").is_file()]
    if len(found) != 1:
        raise ValueError("expected exactly one complete native spatial output")
    spatial = found[0]
    run_path, policy_path = spatial / "run_summary.tsv", spatial / "summary.tsv"
    for path, key in [(run_path, "native_summary_sha256"), (policy_path, "policy_summary_sha256")]:
        if sha256(path) != seal.get(key):
            raise ValueError(f"primary seal hash mismatch: {key}")
    run = dict(line.split("\t", 1) for line in run_path.read_text().splitlines())
    if run["source_revision"] != seal["star_source_revision"]:
        raise ValueError("summary source revision does not match primary seal")
    rows = policy_rows(policy_path)
    masses, residuals = {}, {}
    for policy in POLICIES:
        values = [float(r["mass"]) for r in rows if r["product"] == policy]
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("invalid policy mass")
        delta = max(values) - min(values)
        if (policy != "soft_expected" and (delta or any(not x.is_integer() for x in values))) or (policy == "soft_expected" and delta > 1e-9 * max(1, *values)):
            raise ValueError(f"mass does not conserve across spatial scales: {policy}")
        masses[policy] = values[0]
        residuals[policy] = {"absolute": delta, "relative": delta / max(1, *values)}
    for policy, key in [("strict", "strict_molecules"), ("hard", "hard_molecules"), ("gated_hard", "gated_hard_molecules"), ("soft_expected", "soft_expected_mass")]:
        if not math.isclose(masses[policy], float(run[key]), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"run and policy summaries disagree: {policy}")
    fields = ("reads_decoded", "reads_with_candidates", "joined_reads", "candidate_rows", "read_cliques", "exact_h0_reads")
    return {
        "schema": "visium_hd_processing.sealed_primary_summary.v1",
        "release": expected_release, "primary": str(root.resolve()),
        "spatial_output": str(spatial.resolve()),
        "primary_seal_sha256": sha256(seal_path), "source_revision": run["source_revision"],
        "accounting": {k: int(run[k]) for k in fields},
        "policy_masses_full_capture_axis": masses,
        "hard_minus_strict_same_primary": masses["hard"] - masses["strict"],
        "cross_scale_mass_residuals": residuals,
        "not_emitted_by_native_summary": ["feature_eligible_ambiguous_reads", "posterior_clique_mass_residual"],
        "policy_rows": rows,
        "inputs": {str(p): sha256(p) for p in (seal_path, run_path, policy_path)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = summarize(args.primary, args.release)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
