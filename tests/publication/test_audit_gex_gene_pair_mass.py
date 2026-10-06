import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/publication/audit_gex_gene_pair_mass.py"
SPEC = importlib.util.spec_from_file_location("audit_gex_gene_pair_mass", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_mex_totals_streams_only_requested_gene_rows(tmp_path: Path):
    root = tmp_path / "mex"
    root.mkdir()
    (root / "features.tsv").write_text(
        "ENSG1.9\tGENE1\tGene Expression\n"
        "ENSG2\tGENE2\tGene Expression\n"
        "ENSG3\tGENE3\tGene Expression\n"
    )
    (root / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "% fixture\n"
        "3 2 4\n"
        "1 1 2\n"
        "1 2 3\n"
        "2 1 7\n"
        "3 2 11\n"
    )
    totals, paths = MODULE.load_mex_totals(root, ["ENSG1", "ENSG3"])
    assert totals == {"ENSG1": 5.0, "ENSG3": 11.0}
    assert paths == {
        "features": root / "features.tsv",
        "matrix": root / "matrix.mtx",
    }


def test_assignment_and_gene_parsing_are_explicit():
    assert MODULE.parse_assignment("hard=/tmp/hard") == ("hard", Path("/tmp/hard"))
    assert MODULE.parse_gene("ENSG1=MECOM") == ("ENSG1", "MECOM")
    assert MODULE.parse_gene("ENSG2") == ("ENSG2", "ENSG2")
