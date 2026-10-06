from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "audit_hd_flex_probe_barcode_disagreements.py"
SPEC = importlib.util.spec_from_file_location("probe_barcode_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_probe_index_reports_complete_hamming_zero_one_two_and_gene_ties() -> None:
    a = "A" * 50
    probes = [
        MODULE.Probe(a, "geneA", "pA"),
        MODULE.Probe("C" + a[1:], "geneB", "pB"),
        MODULE.Probe("C" + a[1:49] + "C", "geneC", "pC"),
    ]
    index = MODULE.ProbeIndex(probes)
    exact = index.match(a)
    assert exact.distance == 0 and exact.genes == ("geneA",)
    h1 = index.match("G" + a[1:])
    assert h1.distance == 1 and h1.genes == ("geneA", "geneB")
    h2 = index.match("G" + a[1:49] + "G")
    assert h2.distance == 2
    assert h2.genes == ("geneA", "geneB", "geneC")
    assert index.match("G" * 50).distance is None


def test_coordinate_normalization_accepts_padding_and_suffix() -> None:
    assert MODULE.parse_coordinate("s_002um_00012_00340-1") == (12, 340)
    assert MODULE.parse_coordinate("s_002um_12_340") == (12, 340)
    assert MODULE.parse_coordinate("ACGT") is None


def test_gzip_resolver_tables_are_accepted(tmp_path: Path) -> None:
    molecules = tmp_path / "star_molecules.tsv.gz"
    with gzip.open(molecules, "wt", encoding="utf-8", newline="") as handle:
        handle.write("umi_mode\tproduct\tfeature_id\tcorrected_umi\tunit_2um\tmember_read_ids\n")
        handle.write("1mm_cr\tpostcollapse_hard\tgeneA\tAAAA\ts_002um_1_2\tread1\n")
    assert MODULE.load_open_reads(molecules, "1mm_cr") == {
        "read1": {"gene": "geneA", "umi": "AAAA", "coordinate": (1, 2)}
    }

    cliques = tmp_path / "candidate_cliques.tsv.gz"
    with gzip.open(cliques, "wt", encoding="utf-8", newline="") as handle:
        handle.write("read_clique_id\tmember_read_ids\tcandidate\tposterior\n")
        handle.write("c1\tread1\ts_002um_1_2\t1\n")
    assert MODULE.load_cliques(cliques)["read1"] == {
        "candidates": [((1, 2), 1.0)], "reads": ["read1"],
    }


def test_complete_feature_lineage_includes_hash_and_alignment_keep(tmp_path: Path) -> None:
    ledger = tmp_path / "feature_lineage.tsv.gz"
    with gzip.open(ledger, "wt", encoding="utf-8", newline="") as handle:
        handle.write("read_id\taction\tfeature_source\tfeature_id\n")
        handle.write("hash\tkeep\thash_H0\tgeneA\n")
        handle.write("fallback\tkeep\talignment_fallback\tgeneB\n")
        handle.write("miss\tmiss\talignment_fallback\t\n")
        handle.write("deny\tdeny_probe_ambiguous\thash_H1_deny\t\n")
    assert MODULE.load_star_feature_ledger(ledger) == {"hash", "fallback"}
