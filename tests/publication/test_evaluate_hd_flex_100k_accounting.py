from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/publication/evaluate_hd_flex_100k_accounting.py"
GENERIC_SCRIPT = ROOT / "scripts/publication/evaluate_hd_flex_open_accounting.py"


def dump(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accounting_conserves_mass_and_treats_sentinel_as_diagnostic(tmp_path: Path):
    preflight = {"status": "pass", "raw_universe": {"total_read_pairs": 100000, "unique_read_names": 100000}}
    empty_lineage = {"prohibited_fields_used": {"vendor": False}, "counts": {}}
    hash_summary = {**empty_lineage, "counts": {"keep_h0": 90, "keep_h1": 2, "deny_probe_ambiguous": 1, "miss": 7}}
    feature_summary = {**empty_lineage, "counts": {"alignment_fallback": 4, "feature_reads": 96}}
    join = {**empty_lineage, "counts": {"candidate_rows_without_feature": 3}}
    resolver = {"parameters": {"compatibility_ledger_enabled": False}, "counts": {"candidate_reads": 91, "read_cliques": 80, "gated_assigned": 72, "gated_deferred": 8}}
    products = {policy: {"scale_mass_conserved": True, **{scale: {"mass": 10} for scale in ("square_002um", "square_008um", "square_016um")}} for policy in ("strict", "postcollapse_soft", "postcollapse_hard", "gated_hard")}
    materialized = {"products": products}
    audit = {"status": "pass", "failures": []}
    determinism = {"status": "pass", "declared_hashes_identical": True, "vendor_artifacts_opened_before_seal": False}
    paths = {
        "preflight": dump(tmp_path / "preflight.json", preflight),
        "hash-summary": dump(tmp_path / "hash.json", hash_summary),
        "feature-summary": dump(tmp_path / "feature.json", feature_summary),
        "join-summary": dump(tmp_path / "join.json", join),
        "resolver-summary": dump(tmp_path / "resolver.json", resolver),
        "materialized-1mm-summary": dump(tmp_path / "m1.json", materialized),
        "materialized-exact-summary": dump(tmp_path / "me.json", materialized),
        "audit-1mm": dump(tmp_path / "a1.json", audit),
        "audit-exact": dump(tmp_path / "ae.json", audit),
        "determinism": dump(tmp_path / "det.json", determinism),
        "sentinel-1mm-summary": dump(tmp_path / "s1.json", materialized),
        "sentinel-exact-summary": dump(tmp_path / "se.json", materialized),
    }
    out_json, out_tsv = tmp_path / "out.json", tmp_path / "out.tsv"
    argv = [sys.executable, str(SCRIPT)]
    for name, path in paths.items():
        argv.extend((f"--{name}", str(path)))
    argv.extend(("--provenance-run-id", "fixture", "--generator-commit", "abc", "--out-json", str(out_json), "--out-tsv", str(out_tsv)))
    completed = subprocess.run(argv, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(out_json.read_text())
    assert result["status"] == "pass"
    assert result["vendor_artifacts_opened"] is False
    assert result["material_historical_difference_observed"] is False
    assert len(result["historical_sentinel_differences"]) == 8
    assert out_tsv.read_text().count("raw_molecule_mass") == 8


def test_generic_accounting_accepts_declared_full_read_universe(tmp_path: Path):
    expected = 212_554_625
    preflight = {"status": "pass", "raw_universe": {"total_read_pairs": expected, "unique_read_names": expected}}
    empty_lineage = {"prohibited_fields_used": {"vendor": False}, "counts": {}}
    hash_summary = {**empty_lineage, "counts": {"keep_h0": 90, "keep_h1": 2, "deny_probe_ambiguous": 1, "miss": 7}}
    feature_summary = {**empty_lineage, "counts": {"alignment_fallback": 4, "feature_reads": 96}}
    join = {**empty_lineage, "counts": {"candidate_rows_without_feature": 3}}
    resolver = {"parameters": {"compatibility_ledger_enabled": False}, "counts": {"candidate_reads": 91, "read_cliques": 80, "gated_assigned": 72, "gated_deferred": 8}}
    products = {policy: {"scale_mass_conserved": True, **{scale: {"mass": 10} for scale in ("square_002um", "square_008um", "square_016um")}} for policy in ("strict", "postcollapse_soft", "postcollapse_hard", "gated_hard")}
    materialized = {"products": products}
    audit = {"status": "pass", "failures": []}
    determinism = {"status": "pass", "declared_hashes_identical": True, "vendor_artifacts_opened_before_seal": False}
    paths = {
        "preflight": dump(tmp_path / "preflight.json", preflight),
        "hash-summary": dump(tmp_path / "hash.json", hash_summary),
        "feature-summary": dump(tmp_path / "feature.json", feature_summary),
        "join-summary": dump(tmp_path / "join.json", join),
        "resolver-summary": dump(tmp_path / "resolver.json", resolver),
        "materialized-1mm-summary": dump(tmp_path / "m1.json", materialized),
        "materialized-exact-summary": dump(tmp_path / "me.json", materialized),
        "audit-1mm": dump(tmp_path / "a1.json", audit),
        "audit-exact": dump(tmp_path / "ae.json", audit),
        "determinism": dump(tmp_path / "det.json", determinism),
    }
    out_json, out_tsv = tmp_path / "out.json", tmp_path / "out.tsv"
    argv = [sys.executable, str(GENERIC_SCRIPT)]
    for name, path in paths.items():
        argv.extend((f"--{name}", str(path)))
    argv.extend(("--expected-read-pairs", str(expected), "--provenance-run-id", "full-fixture", "--generator-commit", "abc", "--out-json", str(out_json), "--out-tsv", str(out_tsv)))
    completed = subprocess.run(argv, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(out_json.read_text())
    assert result["schema"] == "visium_hd_processing.flex_open_accounting.v1"
    assert result["status"] == "pass"
    assert result["vendor_artifacts_opened"] is False
    assert result["rows"][0]["value"] == expected
    assert all(row["denominator"] == expected for row in result["rows"] if row["section"] == "open_accounting")
