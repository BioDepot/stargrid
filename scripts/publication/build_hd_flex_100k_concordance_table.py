#!/usr/bin/env python3
"""Build one long-form table from the registered Flex 100K result products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


FIELDS = (
    "source", "category", "umi_mode", "policy", "subset", "scale_um",
    "metric", "value", "numerator", "denominator", "unit", "interpretation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounting-json", type=Path, required=True)
    parser.add_argument("--vendor-ledger-summary", type=Path, required=True)
    parser.add_argument("--read-summary", action="append", required=True, metavar="MODE=PATH")
    parser.add_argument("--shared-summary", action="append", required=True, metavar="MODE=PATH")
    parser.add_argument("--shared-table", action="append", required=True, metavar="MODE=PATH")
    parser.add_argument("--membership-summary", action="append", required=True, metavar="MODE=PATH")
    parser.add_argument("--matrix-table", action="append", required=True, metavar="MODE=PATH")
    parser.add_argument("--hash-miss-summary", type=Path, required=True)
    parser.add_argument("--probe-audit-summary", type=Path, required=True)
    parser.add_argument("--provenance-run-id", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--result-id", default="hd_crc_flex_100k_sr_concordance_v1")
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def keyed(values: list[str], label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} expects MODE=PATH: {value}")
        mode, raw_path = value.split("=", 1)
        if mode in result:
            raise ValueError(f"duplicate {label} mode: {mode}")
        result[mode] = Path(raw_path)
    if set(result) != {"1mm_cr", "exact"}:
        raise ValueError(f"{label} requires 1mm_cr and exact")
    return result


def make_row(source: str, category: str, metric: str, value, **kwargs) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update({"source": source, "category": category, "metric": metric, "value": value})
    result.update(kwargs)
    return result


def flatten_numbers(prefix: str, value, rows: list[dict[str, object]], source: str, category: str, **dimensions) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        rows.append(make_row(source, category, prefix, value, **dimensions))
    elif isinstance(value, dict):
        for key in sorted(value):
            flatten_numbers(f"{prefix}.{key}" if prefix else key, value[key], rows, source, category, **dimensions)


def load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    args = parse_args()
    try:
        read_paths = keyed(args.read_summary, "--read-summary")
        shared_summary_paths = keyed(args.shared_summary, "--shared-summary")
        shared_table_paths = keyed(args.shared_table, "--shared-table")
        membership_paths = keyed(args.membership_summary, "--membership-summary")
        matrix_paths = keyed(args.matrix_table, "--matrix-table")
        fixed = {
            "accounting_json": args.accounting_json,
            "vendor_ledger_summary": args.vendor_ledger_summary,
            "hash_miss_summary": args.hash_miss_summary,
            "probe_audit_summary": args.probe_audit_summary,
        }
        all_paths = dict(fixed)
        for label, mapping in (
            ("read", read_paths), ("shared_summary", shared_summary_paths),
            ("shared_table", shared_table_paths), ("membership", membership_paths),
            ("matrix", matrix_paths),
        ):
            all_paths.update({f"{label}_{mode}": path for mode, path in mapping.items()})
        payload = {name: load_json(path) for name, path in fixed.items()}
        read = {mode: load_json(path) for mode, path in read_paths.items()}
        shared = {mode: load_json(path) for mode, path in shared_summary_paths.items()}
        membership = {mode: load_json(path) for mode, path in membership_paths.items()}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    failures = []
    accounting = payload["accounting_json"]
    vendor = payload["vendor_ledger_summary"]
    hash_miss = payload["hash_miss_summary"]
    probe = payload["probe_audit_summary"]
    if accounting.get("status") != "pass" or accounting.get("vendor_artifacts_opened") is not False:
        failures.append("sealed open accounting did not pass before vendor access")
    if vendor.get("counts", {}).get("primary_reads") != 100000:
        failures.append("vendor ledger does not contain 100000 primary reads")
    if hash_miss.get("status") != "pass" or probe.get("status") != "pass":
        failures.append("hash-miss or probe/barcode audit failed")

    rows: list[dict[str, object]] = []
    for item in accounting.get("rows", []):
        rows.append(make_row(
            "open_accounting", item.get("section", "accounting"), item.get("metric", ""),
            item.get("value", ""), denominator=item.get("denominator", ""),
            unit=item.get("unit", ""), interpretation=item.get("note", ""),
        ))
    flatten_numbers("", vendor.get("counts", {}), rows, "vendor_ledger", "vendor_accounting", unit="reads_or_families")

    for mode in ("1mm_cr", "exact"):
        counts = read[mode].get("counts", {})
        metrics = read[mode].get("shared_metrics", {})
        if sum(counts.get(name, 0) for name in ("shared", "open_only", "space_ranger_only", "neither")) != 100000:
            failures.append(f"{mode}: read partition does not reconcile")
        if metrics.get("gene") != counts.get("shared"):
            failures.append(f"{mode}: shared gene concordance is not complete")
        flatten_numbers("", counts, rows, "read_concordance", "read_partition", umi_mode=mode, unit="reads")
        for metric, numerator in sorted(metrics.items()):
            denominator = counts.get("shared", 0)
            rows.append(make_row(
                "read_concordance", "shared_identity", metric, numerator / denominator if denominator else 0,
                umi_mode=mode, numerator=numerator, denominator=denominator, unit="fraction",
                interpretation="conditioned on jointly matrix-eligible reads",
            ))
        for metric, numerator, denominator in (
            ("shared_fraction_of_open_eligible", counts.get("shared", 0), counts.get("open_hard_reads", 0)),
            ("shared_fraction_of_space_ranger_eligible", counts.get("shared", 0), counts.get("space_ranger_eligible_reads", 0)),
        ):
            rows.append(make_row(
                "read_concordance", "eligibility_coverage", metric,
                numerator / denominator if denominator else 0, umi_mode=mode,
                numerator=numerator, denominator=denominator, unit="fraction",
                interpretation="coverage, not biological accuracy",
            ))

        identity = shared[mode].get("identity", {})
        if identity.get("strict_shared_reads", 0) + identity.get("ambiguous_increment_reads", 0) != identity.get("shared_reads"):
            failures.append(f"{mode}: strict/ambiguous partition failed")
        flatten_numbers("", identity, rows, "shared_policy", "shared_partition", umi_mode=mode, unit="reads")
        for entry in load_tsv(shared_table_paths[mode]):
            for metric in ("coverage", "compatible_support", "conditional_fraction", "unconditional_fraction"):
                rows.append(make_row(
                    "shared_policy", entry["metric_type"], metric, entry[metric],
                    umi_mode=mode, policy=entry["policy"], subset=entry["subset"],
                    scale_um=entry["scale_um"], numerator=entry["covered_reads"],
                    denominator=entry["cohort_reads"], unit="fraction_or_expected_reads",
                    interpretation="strict/shared support reported separately from ambiguous increment",
                ))

        mcounts = membership[mode].get("counts", {})
        flatten_numbers("", mcounts, rows, "molecule_membership", "exact_member_sets", umi_mode=mode)
        for entry in load_tsv(matrix_paths[mode]):
            dims = {"umi_mode": mode, "policy": entry["policy"], "scale_um": entry["scale_um"]}
            for metric, value in entry.items():
                if metric in {"umi_mode", "policy", "scale_um"}:
                    continue
                rows.append(make_row(
                    "matrix_concordance", "union_axis", metric, value, **dims,
                    unit="raw_count_or_statistic",
                    interpretation=(
                        "secondary shape metric beside raw mass" if metric == "normalized_total_variation"
                        else "raw mass is not normalized" if "mass" in metric else ""
                    ),
                ))

    flatten_numbers("", hash_miss.get("counts", {}), rows, "hash_miss_audit", "hash_miss_counts")
    flatten_numbers("", hash_miss.get("assigned_probe_hamming", {}), rows, "hash_miss_audit", "assigned_probe_hamming")
    flatten_numbers("", hash_miss.get("alignment_fallback_space_ranger_hash_miss_audit", {}), rows, "hash_miss_audit", "alignment_fallback")
    flatten_numbers("", probe.get("counts", {}), rows, "probe_barcode_audit", "eligibility_partition")
    flatten_numbers("", probe.get("probe_counts", {}), rows, "probe_barcode_audit", "probe_hamming")
    flatten_numbers("", probe.get("barcode_counts", {}), rows, "probe_barcode_audit", "barcode_evidence")

    if any("jaccard" in str(row["metric"]).lower() for row in rows):
        failures.append("occupancy Jaccard appeared in the generated table")
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "visium_hd_processing.flex_100k_concordance_table.v1",
        "result_id": args.result_id,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "dataset": "H1-GMHFWPH/D1",
        "provenance_run_id": args.provenance_run_id,
        "generator": {"script": str(Path(__file__).resolve()), "commit": args.generator_commit, "argv": sys.argv[1:]},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in sorted(all_paths.items())},
        "output": {"path": str(args.out_tsv.resolve()), "sha256": sha256(args.out_tsv), "rows": len(rows)},
        "interpretation": {
            "space_ranger_role": "compatibility_not_truth",
            "raw_mass_normalized": False,
            "occupancy_jaccard_computed": False,
            "strict_shared_and_ambiguous_increment_separate": True,
            "normalized_tv_role": "secondary_shape_metric_beside_raw_mass",
        },
    }
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
