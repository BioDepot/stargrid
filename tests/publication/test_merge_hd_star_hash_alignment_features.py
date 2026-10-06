from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "merge_hd_star_hash_alignment_features.py"
SPEC = importlib.util.spec_from_file_location("hybrid_features", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hash_keep_wins_miss_falls_through_and_deny_stays_denied() -> None:
    ledger = {
        "h0": {"action": "keep", "feature_id": "g1", "hash_tier": "H0"},
        "miss": {"action": "miss", "feature_id": "", "hash_tier": ""},
        "deny": {"action": "deny_probe_ambiguous", "feature_id": "", "hash_tier": "H1"},
    }
    hashes = {"h0": ("g1", "AAAA", "H0")}
    alignments = {"h0": ("g1", "AAAA"), "miss": ("g2", "CCCC"), "deny": ("g3", "GGGG")}
    assert MODULE.merge_assignments(ledger, hashes, alignments) == {
        "h0": ("g1", "AAAA", "hash_H0", "H0"),
        "miss": ("g2", "CCCC", "alignment_fallback", ""),
    }


def test_hash_alignment_disagreement_is_a_stop_condition() -> None:
    ledger = {"read": {"action": "keep", "feature_id": "g1", "hash_tier": "H1"}}
    with pytest.raises(ValueError, match="hash and alignment feature disagree"):
        MODULE.merge_assignments(
            ledger, {"read": ("g1", "AAAA", "H1")}, {"read": ("g2", "AAAA")},
        )


def test_secondary_and_supplementary_alignments_are_diagnostic_only() -> None:
    primary = SimpleNamespace(
        is_unmapped=False, is_secondary=False, is_supplementary=False,
    )
    secondary = SimpleNamespace(
        is_unmapped=False, is_secondary=True, is_supplementary=False,
    )
    supplementary = SimpleNamespace(
        is_unmapped=False, is_secondary=False, is_supplementary=True,
    )
    unmapped = SimpleNamespace(
        is_unmapped=True, is_secondary=False, is_supplementary=False,
    )
    assert MODULE.alignment_record_is_feature_evidence(primary)
    assert not MODULE.alignment_record_is_feature_evidence(secondary)
    assert not MODULE.alignment_record_is_feature_evidence(supplementary)
    assert not MODULE.alignment_record_is_feature_evidence(unmapped)


def test_reported_alignment_set_completeness_detects_capped_bam() -> None:
    assert MODULE.reported_alignment_set_is_complete(0, set())
    assert MODULE.reported_alignment_set_is_complete(1, {1})
    assert MODULE.reported_alignment_set_is_complete(4, {4})
    assert not MODULE.reported_alignment_set_is_complete(1, {4})
    assert not MODULE.reported_alignment_set_is_complete(4, {3, 4})
    assert not MODULE.reported_alignment_set_is_complete(1, set())
