from __future__ import annotations

import csv
import gzip
import json
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "extract_hd_star_hash_feature_table.py"


def _pack(sequence: str) -> tuple[int, int]:
    lo = hi = 0
    for base in sequence:
        code = {"A": 0, "C": 1, "G": 2, "T": 3}[base]
        hi = ((hi << 2) & ((1 << 64) - 1)) | (lo >> 62)
        lo = ((lo << 2) & ((1 << 64) - 1)) | code
    return lo, hi


def _fastq(path: Path, mate: int, sequences: list[tuple[str, str]]) -> None:
    with gzip.open(path, "wt") as handle:
        for read_id, sequence in sequences:
            handle.write(f"@{read_id}\n{sequence}\n+\n{'I' * len(sequence)}\n")


def test_replays_h0_h1_and_deny_without_probe_csv(tmp_path: Path) -> None:
    r1, r2 = tmp_path / "r1.fastq.gz", tmp_path / "r2.fastq.gz"
    read_ids = ["h0", "h1", "deny", "miss"]
    _fastq(r1, 1, [(name, "ACGTACGTA" + "C" * 60) for name in read_ids])
    seqs = ["A" * 50, "C" * 50, "G" * 50, "T" * 50]
    _fastq(r2, 2, list(zip(read_ids, seqs, strict=True)))
    feature_ids = tmp_path / "features.txt"
    feature_ids.write_text("geneA\ngeneB\n")
    cache = tmp_path / "cache.bin"
    rows = [
        (*_pack("A" * 50), 1, 0, 0, 1),
        (*_pack("C" * 50), 2, 1, 0, 0),
        (*_pack("G" * 50), 0, 2, 1, 0),
    ]
    rows.sort(key=lambda row: (row[1], row[0], row[5]))
    with cache.open("wb") as handle:
        handle.write(struct.pack("<8sHHIQ", b"FH01SEQ1", 2, 50, 24, len(rows)))
        for row in rows:
            handle.write(struct.pack("<QQIBBH", *row))
    feature_table = tmp_path / "features.tsv.gz"
    ledger = tmp_path / "ledger.tsv.gz"
    summary = tmp_path / "summary.json"
    subprocess.run([
        sys.executable, str(SCRIPT), "--r1-fastq", str(r1), "--r2-fastq", str(r2),
        "--hash-cache", str(cache), "--feature-id-list", str(feature_ids),
        "--out-feature-tsv", str(feature_table), "--out-read-ledger", str(ledger),
        "--summary-json", str(summary),
    ], check=True)
    with gzip.open(feature_table, "rt") as handle:
        rows_out = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["read_id"], row["feature_id"], row["hash_tier"]) for row in rows_out] == [
        ("h0", "geneA", "H0"), ("h1", "geneB", "H1")
    ]
    payload = json.loads(summary.read_text())
    assert payload["counts"] == {
        "deny_probe_ambiguous": 1, "feature_reads": 2, "keep_h0": 1,
        "keep_h1": 1, "miss": 1, "read_pairs": 4,
    }
    assert payload["prohibited_fields_used"]["probe_csv"] is False
