import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "score_hd_flex_mapq_increment_morphology.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "score_hd_flex_mapq_increment_morphology", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sparse_mapq_delta_reconciles_positive_and_negative_mass():
    genomic = sparse.csr_matrix(np.array([
        [3.0, 1.0, 0.0],
        [0.0, 4.0, 2.0],
    ]))
    mapq_off = sparse.csr_matrix(np.array([
        [5.0, 0.0, 1.0],
        [0.0, 6.0, 1.0],
    ]))
    positive, negative = MODULE.split_sparse_delta(mapq_off, genomic)
    assert positive.toarray().tolist() == [[2.0, 0.0, 1.0], [0.0, 2.0, 0.0]]
    assert negative.toarray().tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert positive.sum() - negative.sum() == mapq_off.sum() - genomic.sum()


def test_target_feature_list_preserves_order_and_accepts_tsv(tmp_path):
    path = tmp_path / "features.txt"
    path.write_text("ENSG2\tignored\n\nENSG1\n", encoding="utf-8")
    assert MODULE.load_target_feature_list(path) == {"ENSG2": 0, "ENSG1": 1}


def test_target_feature_list_rejects_duplicates(tmp_path):
    path = tmp_path / "features.txt"
    path.write_text("ENSG1\nENSG1\tignored\n", encoding="utf-8")
    try:
        MODULE.load_target_feature_list(path)
    except ValueError as exc:
        assert "duplicate target feature identifiers" in str(exc)
    else:
        raise AssertionError("duplicate target feature list was accepted")
