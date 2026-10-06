from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/publication/build_hd_flex_100k_concordance_table.py"


def dump(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def table(path: Path, fields: tuple[str, ...], rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return path


def test_builds_long_table_with_required_interpretation(tmp_path: Path):
    accounting = dump(tmp_path / "accounting.json", {"status": "pass", "vendor_artifacts_opened": False, "rows": [{"section": "policy_mass", "metric": "strict.raw_molecule_mass", "value": 8, "unit": "molecules", "denominator": "not normalized", "note": ""}]})
    vendor = dump(tmp_path / "vendor.json", {"counts": {"primary_reads": 100000, "matrix_eligible_reads": 8}})
    hash_miss = dump(tmp_path / "hash.json", {"status": "pass", "counts": {"star_hash_miss_reads": 2}, "assigned_probe_hamming": {"H2": 1}, "alignment_fallback_space_ranger_hash_miss_audit": {"recovered_reads": 1}})
    probe = dump(tmp_path / "probe.json", {"status": "pass", "counts": {"shared_eligible_reads": 8}, "probe_counts": {"shared": {"hamming_0": 8}}, "barcode_counts": {"star_only": {"reads": 1}}})
    args = [sys.executable, str(SCRIPT), "--accounting-json", str(accounting), "--vendor-ledger-summary", str(vendor), "--hash-miss-summary", str(hash_miss), "--probe-audit-summary", str(probe)]
    for mode in ("1mm_cr", "exact"):
        read = dump(tmp_path / f"read_{mode}.json", {"counts": {"shared": 8, "open_only": 1, "space_ranger_only": 1, "neither": 99990, "open_hard_reads": 9, "space_ranger_eligible_reads": 9}, "shared_metrics": {"gene": 8, "corrected_umi": 8}})
        shared = dump(tmp_path / f"shared_{mode}.json", {"identity": {"shared_reads": 8, "strict_shared_reads": 6, "ambiguous_increment_reads": 2}})
        membership = dump(tmp_path / f"membership_{mode}.json", {"counts": {"exact_member_set_matches": 7, "space_ranger_molecules": 8}})
        shared_tsv = table(tmp_path / f"shared_{mode}.tsv", ("subset", "policy", "metric_type", "scale_um", "cohort_reads", "covered_reads", "coverage", "compatible_support", "conditional_fraction", "unconditional_fraction"), [{"subset": "ambiguous_increment", "policy": "soft_expected", "metric_type": "posterior_support", "scale_um": 2, "cohort_reads": 2, "covered_reads": 2, "coverage": 1, "compatible_support": 1.5, "conditional_fraction": .75, "unconditional_fraction": .75}])
        matrix_tsv = table(tmp_path / f"matrix_{mode}.tsv", ("umi_mode", "policy", "scale_um", "open_raw_mass", "space_ranger_raw_mass", "normalized_total_variation"), [{"umi_mode": mode, "policy": "strict", "scale_um": 2, "open_raw_mass": 8, "space_ranger_raw_mass": 7, "normalized_total_variation": .1}])
        args.extend(("--read-summary", f"{mode}={read}", "--shared-summary", f"{mode}={shared}", "--shared-table", f"{mode}={shared_tsv}", "--membership-summary", f"{mode}={membership}", "--matrix-table", f"{mode}={matrix_tsv}"))
    out_tsv, out_json = tmp_path / "out.tsv", tmp_path / "out.json"
    args.extend(("--provenance-run-id", "fixture", "--generator-commit", "abc", "--out-tsv", str(out_tsv), "--out-json", str(out_json)))
    completed = subprocess.run(args, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(out_json.read_text())
    assert result["status"] == "pass"
    assert result["interpretation"]["occupancy_jaccard_computed"] is False
    text = out_tsv.read_text()
    assert "shared_fraction_of_space_ranger_eligible" in text
    assert "normalized_total_variation" in text
    assert "ambiguous_increment" in text
