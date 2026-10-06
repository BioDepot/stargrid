import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "analyze_flex_mapq_increment_localization.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "analyze_flex_mapq_increment_localization", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_align_split_deciles_and_concentration():
    aligned = MODULE.align_totals(
        ["g2", "g1"], np.array([3.0, 2.0]), {"g1": 0, "g2": 1, "g3": 2},
    )
    assert aligned.tolist() == [2.0, 3.0, 0.0]
    positive, negative = MODULE.split_delta(np.array([2.0, -3.0, 0.0]))
    assert positive.tolist() == [2.0, 0.0, 0.0]
    assert negative.tolist() == [0.0, 3.0, 0.0]
    deciles = MODULE.rank_deciles(
        np.arange(20, dtype=float), np.ones(20, dtype=bool),
    )
    assert np.bincount(deciles, minlength=10).tolist() == [2] * 10
    assert MODULE.concentration(np.array([1.0, 2.0, 7.0]), 1) == 0.7


def test_decile_rows_reconcile_signed_delta():
    rows = MODULE.decile_rows(
        "gene",
        np.arange(1, 11, dtype=float),
        np.arange(2, 12, dtype=float),
        np.ones(10, dtype=bool),
    )
    assert len(rows) == 10
    assert sum(row["signed_delta_mass"] for row in rows) == 10.0
