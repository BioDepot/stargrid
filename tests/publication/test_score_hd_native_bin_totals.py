import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "publication" / "score_hd_native_bin_totals.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("score_hd_native_bin_totals", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_field_restricts_to_native_indices_and_preserves_full_mass(tmp_path):
    path = tmp_path / "field.tsv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("unit_2um\tmolecule_mass\n")
        handle.write("s_002um_1_1-1\t3.5\n")
        handle.write("s_002um_0_0-1\t2\n")
    field, mass, occupied = MODULE.load_field(
        path, np.array([0, 3], dtype=np.int64), width=2, height=2,
    )
    assert field.tolist() == [2.0, 3.5]
    assert mass == 5.5
    assert occupied == 2


def test_signed_spatial_delta_reports_soft_redistribution():
    strict = np.array([2.0, 0.0, 1.0])
    soft = np.array([1.0, 2.5, 1.0])
    positive, removed = MODULE.split_delta(soft, strict)
    assert positive.tolist() == [0.0, 2.5, 0.0]
    assert removed.tolist() == [1.0, 0.0, 0.0]
    assert positive.sum() - removed.sum() == soft.sum() - strict.sum()
