from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "scripts" / "publication" / "compare_hd_flex_molecule_membership.py"
SPEC = importlib.util.spec_from_file_location("molecule_concordance", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def molecule(identifier: str, reads: tuple[str, ...], unit: str = "s_002um_0_0"):
    return MODULE.MoleculeAssignment(identifier, "g", "A", unit, reads)


def test_summary_preserves_mass_and_matches_exact_member_sets() -> None:
    star = [molecule("a", ("r1", "r2")), molecule("b", ("r3",))]
    vendor = [molecule("x", ("r1", "r2")), molecule("y", ("r3", "r4"))]
    summary, components, exact = MODULE.summarize(star, vendor)
    assert summary["open_molecules"] == 2
    assert summary["space_ranger_molecules"] == 2
    assert summary["raw_molecule_mass_difference"] == 0
    assert summary["exact_member_set_matches"] == 1
    assert summary["exact_identity"]["feature"] == 1
    assert len(components) == 2
    assert len(exact) == 1
