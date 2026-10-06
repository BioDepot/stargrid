import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "score_hd_flex_full_compartments.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("score_hd_flex_full_compartments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compatibility_decomposition_preserves_raw_mass():
    star = {
        ("g1", "s_002um_0_0"): 3.0,
        ("g2", "s_002um_0_1"): 1.0,
    }
    vendor = {
        ("g1", "s_002um_0_0"): 2.0,
        ("g3", "s_002um_1_0"): 4.0,
    }
    shared, star_only, vendor_only = MODULE.compatibility_fields(
        star, vendor, width=2, height=2,
    )
    assert shared.tolist() == [2.0, 0.0, 0.0, 0.0]
    assert star_only.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert vendor_only.tolist() == [0.0, 0.0, 4.0, 0.0]
    assert shared.sum() + star_only.sum() == sum(star.values())
    assert shared.sum() + vendor_only.sum() == sum(vendor.values())


def test_increment_is_feature_bin_specific_and_scores_raw_compartments():
    strict = {("g1", "s_002um_0_0"): 1.0}
    policy = {
        ("g1", "s_002um_0_0"): 1.0,
        ("g2", "s_002um_1_1"): 2.0,
    }
    increment = MODULE.increment_field(policy, strict, width=2, height=2)
    assert increment.tolist() == [0.0, 0.0, 0.0, 2.0]
    states = np.array([0, 1, 2, 2], dtype=np.uint8)
    row = MODULE.score_field(
        "increment", "ambiguous_increment_over_strict", "hard",
        increment, states, width=2, height=2,
    )
    assert row["raw_molecule_mass"] == 2.0
    assert row["in_cell_mass_fraction"] == 1.0
    assert row["in_nucleus_mass_fraction"] == 1.0
    assert row["extracellular_mass_fraction"] == 0.0
    assert row["equal_mass_normalization_applied"] is False


def test_increment_rejects_loss_of_strict_mass():
    strict = {("g1", "s_002um_0_0"): 1.0}
    policy = {("g1", "s_002um_0_0"): 0.5}
    try:
        MODULE.increment_field(policy, strict, width=1, height=1)
    except ValueError as error:
        assert "strict mass is not a subset" in str(error)
    else:
        raise AssertionError("strict mass loss was accepted")
