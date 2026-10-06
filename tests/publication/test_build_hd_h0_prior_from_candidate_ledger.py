from __future__ import annotations

import csv
import gzip
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "build_hd_h0_prior_from_candidate_ledger.py"


def test_counts_only_unique_minimum_tier_zero_reads(tmp_path: Path) -> None:
    ledger = tmp_path / "candidates.tsv.gz"
    with gzip.open(ledger, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("read_id", "candidate_count", "min_tier", "row2", "col2"), delimiter="\t")
        writer.writeheader()
        writer.writerows([
            {"read_id": "keep", "candidate_count": 1, "min_tier": 0, "row2": 1, "col2": 0},
            {"read_id": "tier1", "candidate_count": 1, "min_tier": 1, "row2": 0, "col2": 1},
            {"read_id": "ambig", "candidate_count": 2, "min_tier": 0, "row2": 0, "col2": 0},
            {"read_id": "ambig", "candidate_count": 2, "min_tier": 0, "row2": 1, "col2": 1},
        ])
    bc1, bc2 = tmp_path / "bc1.txt", tmp_path / "bc2.txt"
    bc1.write_text("AAAA\nCCCC\n"); bc2.write_text("GGGG\nTTTT\n")
    out, summary = tmp_path / "prior.tsv", tmp_path / "summary.json"
    subprocess.run([
        sys.executable, str(SCRIPT), "--candidate-ledger", str(ledger),
        "--bc1-oligos", str(bc1), "--bc2-oligos", str(bc2),
        "--out-tsv", str(out), "--summary-json", str(summary),
    ], check=True)
    with out.open() as handle:
        rows = {(row["barcode_half"], int(row["oligo_index"])): int(row["exact_h0_read_count"]) for row in csv.DictReader(handle, delimiter="\t")}
    assert rows == {("BC1", 0): 1, ("BC1", 1): 0, ("BC2", 0): 0, ("BC2", 1): 1}
