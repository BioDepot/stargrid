from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "compare_hd_flex_star_hash_space_ranger.py"
SPEC = importlib.util.spec_from_file_location("hash_concordance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compare_separates_hash_feature_and_matrix_eligibility() -> None:
    hashes = {
        "shared": {"action": "keep", "hash_tier": "H0", "feature_id": "g1"},
        "vendor_only": {"action": "keep", "hash_tier": "H1", "feature_id": "g2"},
        "open_only": {"action": "keep", "hash_tier": "H0", "feature_id": "g3"},
        "neither": {"action": "miss", "hash_tier": "", "feature_id": ""},
    }
    open_reads = {
        "shared": {"gene": "g1", "umi": "A", "coordinate": (8, 8)},
        "open_only": {"gene": "g3", "umi": "C", "coordinate": (1, 1)},
    }
    vendor = {
        "shared": {"gene": "g1", "umi": "A", "coordinate": (9, 9), "eligible": True},
        "vendor_only": {"gene": "g2", "umi": "B", "coordinate": (2, 2), "eligible": True},
        "open_only": {"gene": "g3", "umi": "C", "coordinate": None, "eligible": False},
        "neither": {"gene": "", "umi": "", "coordinate": None, "eligible": False},
    }
    result = MODULE.compare(hashes, open_reads, vendor)
    assert result["counts"]["shared"] == 1
    assert result["counts"]["open_only"] == 1
    assert result["counts"]["space_ranger_only"] == 1
    assert result["shared_metrics"] == {
        "coordinate_16um": 1, "coordinate_2um": 0, "coordinate_8um": 1,
        "corrected_umi": 1, "gene": 1,
    }
    assert result["hash_tier_space_ranger_gene"] == {
        "H0": {"reads": 1, "gene_concordant": 1},
        "H1": {"reads": 1, "gene_concordant": 1},
    }


def test_compare_reports_alignment_fallback_lineage_separately() -> None:
    hashes = {
        "hash": {"action": "keep", "hash_tier": "H0", "feature_id": "g1"},
        "fallback": {"action": "miss", "hash_tier": "", "feature_id": ""},
    }
    open_reads = {
        "hash": {"gene": "g1", "umi": "A", "coordinate": (1, 1)},
        "fallback": {"gene": "g2", "umi": "B", "coordinate": (2, 2)},
    }
    vendor = {
        "hash": {"gene": "g1", "umi": "A", "coordinate": (1, 1), "eligible": True},
        "fallback": {"gene": "g2", "umi": "B", "coordinate": (2, 2), "eligible": True},
    }
    lineage = {
        "hash": {"action": "keep", "feature_source": "hash_H0", "feature_id": "g1"},
        "fallback": {"action": "keep", "feature_source": "alignment_fallback", "feature_id": "g2"},
    }
    result = MODULE.compare(hashes, open_reads, vendor, lineage)
    assert result["feature_source_space_ranger_gene"] == {
        "alignment_fallback": {"reads": 1, "gene_concordant": 1},
        "hash_H0": {"reads": 1, "gene_concordant": 1},
    }
    assert result["feature_source_open_shared_gene"] == result["feature_source_space_ranger_gene"]
