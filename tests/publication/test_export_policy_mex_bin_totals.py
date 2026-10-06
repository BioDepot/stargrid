import gzip
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "publication" / "export_policy_mex_bin_totals.py"
SPEC = importlib.util.spec_from_file_location("export_policy_mex_bin_totals", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_field_writer_is_deterministic_and_preserves_raw_mass(tmp_path):
    first, second = tmp_path / "a.tsv.gz", tmp_path / "b.tsv.gz"
    rows = [("s_002um_00000_00000-1", 2.0), ("zero", 0.0), ("s_002um_00000_00001-1", 1.5)]
    assert MODULE.write_field(first, rows) == (2, 3.5)
    assert MODULE.write_field(second, rows) == (2, 3.5)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt") as handle:
        assert handle.read().splitlines() == [
            "unit_2um\tmolecule_mass",
            "s_002um_00000_00000-1\t2",
            "s_002um_00000_00001-1\t1.5",
        ]
