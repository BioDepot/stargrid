#!/usr/bin/env python3
"""Generate a sealed Flex open-primary accounting table before vendor inspection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


POLICIES = ("strict", "postcollapse_soft", "postcollapse_hard", "gated_hard")
SCALES = ("square_002um", "square_008um", "square_016um")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--hash-summary", type=Path, required=True)
    parser.add_argument("--feature-summary", type=Path, required=True)
    parser.add_argument("--join-summary", type=Path, required=True)
    parser.add_argument("--resolver-summary", type=Path, required=True)
    parser.add_argument("--materialized-1mm-summary", type=Path, required=True)
    parser.add_argument("--materialized-exact-summary", type=Path, required=True)
    parser.add_argument("--audit-1mm", type=Path, required=True)
    parser.add_argument("--audit-exact", type=Path, required=True)
    parser.add_argument("--determinism", type=Path, required=True)
    parser.add_argument("--expected-read-pairs", type=int, required=True)
    parser.add_argument("--sentinel-1mm-summary", type=Path)
    parser.add_argument("--sentinel-exact-summary", type=Path)
    parser.add_argument("--provenance-run-id", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--result-id", default="hd_crc_flex_open_primary_v1")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def all_false(mapping: dict) -> bool:
    return bool(mapping) and all(value is False for value in mapping.values())


def row(section: str, metric: str, value, unit: str, denominator="", note="") -> dict:
    return {"section": section, "metric": metric, "value": value, "unit": unit,
            "denominator": denominator, "note": note}


def main() -> int:
    args = parse_args()
    named_paths = {
        "preflight": args.preflight,
        "hash_summary": args.hash_summary,
        "feature_summary": args.feature_summary,
        "join_summary": args.join_summary,
        "resolver_summary": args.resolver_summary,
        "materialized_1mm_summary": args.materialized_1mm_summary,
        "materialized_exact_summary": args.materialized_exact_summary,
        "audit_1mm": args.audit_1mm,
        "audit_exact": args.audit_exact,
        "determinism": args.determinism,
    }
    if args.sentinel_1mm_summary:
        named_paths["sentinel_1mm_summary"] = args.sentinel_1mm_summary
    if args.sentinel_exact_summary:
        named_paths["sentinel_exact_summary"] = args.sentinel_exact_summary
    try:
        data = {name: load(path) for name, path in named_paths.items()}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    failures: list[str] = []
    preflight = data["preflight"]
    hash_summary = data["hash_summary"]
    feature_summary = data["feature_summary"]
    join = data["join_summary"]
    resolver = data["resolver_summary"]
    determinism = data["determinism"]
    audits = {"1mm_cr": data["audit_1mm"], "exact": data["audit_exact"]}
    materialized = {
        "1mm_cr": data["materialized_1mm_summary"],
        "exact": data["materialized_exact_summary"],
    }

    if args.expected_read_pairs < 1:
        failures.append("expected read-pair count must be positive")
    if preflight.get("status") != "pass" or preflight.get("raw_universe", {}).get("total_read_pairs") != args.expected_read_pairs:
        failures.append(f"preflight does not certify exactly {args.expected_read_pairs} read pairs")
    if preflight.get("raw_universe", {}).get("unique_read_names") != args.expected_read_pairs:
        failures.append(f"preflight unique read-name count is not {args.expected_read_pairs}")
    for name, source in (("hash", hash_summary), ("feature", feature_summary), ("join", join)):
        if not all_false(source.get("prohibited_fields_used", {})):
            failures.append(f"{name} lineage used a prohibited field")
    if resolver.get("parameters", {}).get("compatibility_ledger_enabled") is not False:
        failures.append("resolver compatibility ledger was enabled")
    if determinism.get("status") != "pass" or determinism.get("declared_hashes_identical") is not True:
        failures.append("two-primary portable hash determinism did not pass")
    if determinism.get("vendor_artifacts_opened_before_seal") is not False:
        failures.append("vendor artifacts were opened before sealing")
    for mode, audit in audits.items():
        if audit.get("status") != "pass" or audit.get("failures"):
            failures.append(f"{mode} open-primary audit did not pass")

    rows = [
        row("raw_universe", "read_pairs", preflight.get("raw_universe", {}).get("total_read_pairs"), "read_pairs"),
        row("raw_universe", "unique_read_names", preflight.get("raw_universe", {}).get("unique_read_names"), "read_names"),
    ]
    hc = hash_summary.get("counts", {})
    fc = feature_summary.get("counts", {})
    jc = join.get("counts", {})
    rc = resolver.get("counts", {})
    for metric, value in (
        ("probe_hash_h0_reads", hc.get("keep_h0")),
        ("probe_hash_h1_reads", hc.get("keep_h1")),
        ("probe_hash_deny_ambiguous_reads", hc.get("deny_probe_ambiguous")),
        ("probe_hash_miss_reads", hc.get("miss")),
        ("probe_alignment_fallback_reads", fc.get("alignment_fallback")),
        ("probe_feature_assigned_reads", fc.get("feature_reads")),
        ("candidate_reads", rc.get("candidate_reads")),
        ("candidate_rows_without_feature", jc.get("candidate_rows_without_feature")),
        ("read_cliques", rc.get("read_cliques")),
        ("gated_assigned_cliques", rc.get("gated_assigned")),
        ("gated_deferred_cliques", rc.get("gated_deferred")),
    ):
        rows.append(row("open_accounting", metric, value, "reads_or_cliques", args.expected_read_pairs))

    sentinel = {}
    if "sentinel_1mm_summary" in data:
        sentinel["1mm_cr"] = data["sentinel_1mm_summary"]
    if "sentinel_exact_summary" in data:
        sentinel["exact"] = data["sentinel_exact_summary"]
    sentinel_differences = []
    for mode, summary in materialized.items():
        products = summary.get("products", {})
        if set(products) != set(POLICIES):
            failures.append(f"{mode}: missing policy product")
            continue
        for policy in POLICIES:
            product = products[policy]
            if set(key for key in product if key.startswith("square_")) != set(SCALES):
                failures.append(f"{mode}/{policy}: missing scale")
                continue
            masses = [float(product[scale]["mass"]) for scale in SCALES]
            residual = max(masses) - min(masses)
            if product.get("scale_mass_conserved") is not True or residual > 1e-9 * max(1.0, max(masses)):
                failures.append(f"{mode}/{policy}: mass not conserved across scales")
            rows.append(row("policy_mass", f"{mode}.{policy}.raw_molecule_mass", masses[0], "molecules", "not normalized"))
            rows.append(row("policy_mass", f"{mode}.{policy}.scale_mass_residual", residual, "molecules", masses[0]))
            if mode in sentinel:
                old = float(sentinel[mode]["products"][policy]["square_002um"]["mass"])
                delta = masses[0] - old
                sentinel_differences.append({
                    "umi_mode": mode, "policy": policy, "current_mass": masses[0],
                    "historical_sentinel_mass": old, "absolute_delta": delta,
                    "material_difference": abs(delta) > 1e-9 * max(1.0, abs(old)),
                })

    result = {
        "schema": "visium_hd_processing.flex_open_accounting.v1",
        "result_id": args.result_id,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "dataset": "H1-GMHFWPH/D1",
        "provenance_run_id": args.provenance_run_id,
        "generator": {"script": str(Path(__file__).resolve()), "commit": args.generator_commit, "argv": sys.argv[1:]},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in named_paths.items()},
        "vendor_artifacts_opened": False,
        "historical_sentinel_role": "debugging_only_not_acceptance_target" if sentinel else None,
        "historical_sentinel_differences": sentinel_differences,
        "material_historical_difference_observed": any(item["material_difference"] for item in sentinel_differences),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value", "unit", "denominator", "note"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
