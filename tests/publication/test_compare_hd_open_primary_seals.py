from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/publication/compare_hd_open_primary_seals.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_primary(root: Path, content: bytes, metadata_value: str) -> None:
    root.mkdir()
    (root / "payload.tsv").write_bytes(content)
    (root / "run.json").write_text(metadata_value, encoding="utf-8")
    seal = {
        "schema": "visium_hd_processing_recipes.open_primary_seal.v2",
        "status": "sealed",
        "dataset": "fixture/D1",
        "scale": "tiny",
        "declared_hash_contract": "portable_scientific_payload_byte_sha256",
        "declared_hashes": {"payload.tsv": digest(root / "payload.tsv")},
        "instance_metadata_hashes": {"run.json": digest(root / "run.json")},
        "vendor_artifacts_opened": False,
    }
    (root / "primary_seal.json").write_text(json.dumps(seal), encoding="utf-8")


def run_compare(tmp_path: Path, content_b: bytes = b"same\n") -> tuple[subprocess.CompletedProcess, dict]:
    primary_a, primary_b = tmp_path / "a", tmp_path / "b"
    make_primary(primary_a, b"same\n", "path=a\n")
    make_primary(primary_b, content_b, "path=b\n")
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--primary-a", str(primary_a), "--primary-b", str(primary_b),
            "--primary-a-artifact-id", "a", "--primary-b-artifact-id", "b",
            "--provenance-run-id", "fixture_run", "--generator-commit", "abc123",
            "--out-json", str(output),
        ],
        text=True, capture_output=True,
    )
    return completed, json.loads(output.read_text())


def test_accepts_identical_payloads_with_distinct_instance_metadata(tmp_path: Path):
    completed, result = run_compare(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert result["status"] == "pass"
    assert result["declared_hashes_identical"] is True
    assert result["instance_metadata_hashes_identical"] is False
    assert result["instance_metadata_is_cross_run_acceptance_target"] is False


def test_rejects_payload_hash_difference(tmp_path: Path):
    completed, result = run_compare(tmp_path, content_b=b"different\n")
    assert completed.returncode != 0
    assert result["status"] == "fail"
    assert result["declared_hash_mismatches"] == ["payload.tsv"]
