from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "audit_hd_flex_star_hash_misses.py"
SPEC = importlib.util.spec_from_file_location("hash_miss_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_distances_distinguish_substitution_from_single_base_shift() -> None:
    probe = "ACGT" * 12 + "AC"
    substitution = "G" + probe[1:]
    shifted = probe[1:] + "G"
    assert MODULE.hamming(probe, substitution) == 1
    assert MODULE.levenshtein(probe, substitution) == 1
    assert MODULE.hamming(probe, shifted) > 2
    assert MODULE.levenshtein(probe, shifted) == 2


def test_cache_encoding_and_molecule_info_umi_decoding_are_stable() -> None:
    sequence = "ACGT" * 12 + "AC"
    assert MODULE.encode_50(sequence) == MODULE.encode_50(sequence)
    assert MODULE.encode_50(sequence[:10] + "N" + sequence[11:]) is None
    value = 0
    for base in "ACGTACGTA":
        value = (value << 2) | "ACGT".index(base)
    assert MODULE.decode_packed_umi(value, 9) == "ACGTACGTA"


def test_fastq_read_name_normalization() -> None:
    assert MODULE.fastq_id("@read/2 extra\n") == "read"
    assert MODULE.fastq_id("@instrument:1:2 2:N:0:1\n") == "instrument:1:2"


def test_complete_alignment_evidence_distinguishes_secondary_and_ambiguity() -> None:
    secondary_unique = {
        "records": 3, "mapped_records": 3, "primary_genes": set(),
        "secondary_genes": {"g1"}, "overlap_genes": {"g1"},
    }
    assert MODULE.classify_star_alignment_evidence(secondary_unique, "g1") == (
        "unique_gene_all_alignments", "secondary_only",
    )
    ambiguous = {
        "records": 2, "mapped_records": 2, "primary_genes": set(),
        "secondary_genes": set(), "overlap_genes": {"g1", "g2"},
    }
    assert MODULE.classify_star_alignment_evidence(ambiguous, "g1") == (
        "multigene_overlap_includes_vendor_gene", "ambiguous_overlap",
    )
    no_overlap = {
        "records": 1, "mapped_records": 1, "primary_genes": set(),
        "secondary_genes": set(), "overlap_genes": set(),
    }
    assert MODULE.classify_star_alignment_evidence(no_overlap, "g1") == (
        "no_gene_overlap", "none",
    )
