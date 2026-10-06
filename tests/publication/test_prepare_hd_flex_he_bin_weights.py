import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "prepare_hd_flex_he_bin_weights.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_hd_flex_he_bin_weights", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compartment_weights_reconcile_and_preserve_hierarchy():
    weights = MODULE.compartment_weights(
        np.array([0, 1, 2, 2, 0, 1], dtype=np.uint8),
        np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
        np.array([4, 2], dtype=np.int64),
    )
    np.testing.assert_allclose(weights["cell_fraction"], [0.75, 0.5])
    np.testing.assert_allclose(weights["nucleus_fraction"], [0.5, 0.0])
    np.testing.assert_allclose(weights["extracellular_fraction"], [0.25, 0.5])
    assert weights["he_supported_children"].tolist() == [4, 2]


def test_compartment_weights_reject_nucleus_outside_cell_encoding():
    try:
        MODULE.compartment_weights(
            np.array([2], dtype=np.uint8),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
    except ValueError as exc:
        assert "invalid coarse-bin" in str(exc)
    else:
        raise AssertionError("zero-child coarse bin was accepted")
