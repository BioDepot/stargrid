#!/usr/bin/env python3
"""Verify two independently sealed open primaries have identical payload hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_SCHEMA = "visium_hd_processing_recipes.open_primary_seal.v2"
EXPECTED_CONTRACT = "portable_scientific_payload_byte_sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-a", type=Path, required=True)
    parser.add_argument("--primary-b", type=Path, required=True)
    parser.add_argument("--primary-a-artifact-id", required=True)
    parser.add_argument("--primary-b-artifact-id", required=True)
    parser.add_argument("--provenance-run-id", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--result-id", default="hd_crc_flex_100k_determinism_v1")
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seal(root: Path, label: str) -> tuple[Path, dict]:
    path = root / "primary_seal.json"
    if not path.is_file():
        raise ValueError(f"{label}: missing primary seal: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"{label}: expected seal schema {EXPECTED_SCHEMA}, got {payload.get('schema')}")
    if payload.get("status") != "sealed":
        raise ValueError(f"{label}: primary is not sealed")
    if payload.get("vendor_artifacts_opened") is not False:
        raise ValueError(f"{label}: vendor artifacts were opened before sealing")
    if payload.get("declared_hash_contract") != EXPECTED_CONTRACT:
        raise ValueError(f"{label}: unexpected declared hash contract")
    hashes = payload.get("declared_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{label}: no declared hashes")
    return path, payload


def verify_files(root: Path, hashes: dict[str, str]) -> list[dict[str, str]]:
    failures = []
    for relative, expected in sorted(hashes.items()):
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "expected": expected, "observed": "missing"})
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    return failures


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    try:
        seal_a_path, seal_a = load_seal(args.primary_a, "primary_a")
        seal_b_path, seal_b = load_seal(args.primary_b, "primary_b")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    hashes_a = seal_a["declared_hashes"]
    hashes_b = seal_b["declared_hashes"]
    keys_a, keys_b = set(hashes_a), set(hashes_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    mismatched = sorted(key for key in keys_a & keys_b if hashes_a[key] != hashes_b[key])
    integrity_a = verify_files(args.primary_a, hashes_a)
    integrity_b = verify_files(args.primary_b, hashes_b)
    if seal_a.get("dataset") != seal_b.get("dataset") or seal_a.get("scale") != seal_b.get("scale"):
        failures.append("dataset or scale differs between primaries")
    if only_a or only_b:
        failures.append("declared file sets differ")
    if mismatched:
        failures.append("declared payload hashes differ")
    if integrity_a or integrity_b:
        failures.append("one or more sealed payloads fail current byte-integrity verification")

    metadata_a = seal_a.get("instance_metadata_hashes", {})
    metadata_b = seal_b.get("instance_metadata_hashes", {})
    result = {
        "schema": "visium_hd_processing.primary_determinism.v1",
        "result_id": args.result_id,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "provenance_run_id": args.provenance_run_id,
        "generator": {
            "script": str(Path(__file__).resolve()),
            "commit": args.generator_commit,
            "argv": sys.argv[1:],
        },
        "input_artifacts": [
            {"artifact_id": args.primary_a_artifact_id, "path": str(args.primary_a.resolve()),
             "seal_sha256": sha256(seal_a_path)},
            {"artifact_id": args.primary_b_artifact_id, "path": str(args.primary_b.resolve()),
             "seal_sha256": sha256(seal_b_path)},
        ],
        "dataset": seal_a.get("dataset"),
        "scale": seal_a.get("scale"),
        "declared_hash_contract": EXPECTED_CONTRACT,
        "declared_file_count_a": len(hashes_a),
        "declared_file_count_b": len(hashes_b),
        "declared_only_a": only_a,
        "declared_only_b": only_b,
        "declared_hash_mismatches": mismatched,
        "payload_integrity_failures_a": integrity_a,
        "payload_integrity_failures_b": integrity_b,
        "declared_hashes_identical": hashes_a == hashes_b,
        "instance_metadata_file_count_a": len(metadata_a),
        "instance_metadata_file_count_b": len(metadata_b),
        "instance_metadata_hashes_identical": metadata_a == metadata_b,
        "instance_metadata_is_cross_run_acceptance_target": False,
        "vendor_artifacts_opened_before_seal": False,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
