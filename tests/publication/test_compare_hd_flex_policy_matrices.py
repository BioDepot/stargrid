import csv
import gzip
import importlib.util
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


SCRIPT = Path(__file__).parents[2] / "scripts" / "publication" / "compare_hd_flex_policy_matrices.py"
SPEC = importlib.util.spec_from_file_location("compare_hd_flex_policy_matrices", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compare_reports_raw_mass_and_shape_without_jaccard():
    open_values = {("g1", "b1"): 2.0, ("g2", "b2"): 1.0}
    vendor_values = {("g1", "b1"): 1.0, ("g2", "b2"): 1.0}
    result = MODULE.compare(open_values, vendor_values)
    assert result["open_raw_mass"] == 3.0
    assert result["space_ranger_raw_mass"] == 2.0
    assert result["signed_mass_difference"] == 1.0
    assert result["absolute_mass_difference"] == 1.0
    assert result["mass_ratio"] == 1.5
    assert result["union_axis_l1"] == 1.0
    assert result["exact_entries"] == 1
    assert result["exact_entry_fraction"] == 0.5
    assert "jaccard" not in " ".join(result).lower()


def test_mex_and_h5_loaders_normalize_coordinates_and_filter_target_axis(tmp_path):
    mex = tmp_path / "mex"
    mex.mkdir()
    for name, text in {
        "features.tsv.gz": "g1\tg1\tGene Expression\n",
        "barcodes.tsv.gz": "s_002um_00003_00004-1\n",
        "matrix.mtx.gz": (
            "%%MatrixMarket matrix coordinate integer general\n% fixture\n1 1 1\n1 1 2\n"
        ),
    }.items():
        with gzip.open(mex / name, "wt", encoding="utf-8") as handle:
            handle.write(text)
    assert MODULE.load_mex(mex, 2) == {("g1", "s_002um_3_4"): 2.0}

    h5_path = tmp_path / "vendor.h5"
    with h5py.File(h5_path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("shape", data=np.array([2, 1], dtype=np.int64))
        matrix.create_dataset("barcodes", data=np.array([b"s_002um_00003_00004-1"]))
        matrix.create_dataset("indptr", data=np.array([0, 2], dtype=np.int64))
        matrix.create_dataset("indices", data=np.array([0, 1], dtype=np.int64))
        matrix.create_dataset("data", data=np.array([2, 9], dtype=np.int64))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.array([b"g1", b"off_target"]))
        target_sets = features.create_group("target_sets")
        target_sets.create_dataset("panel", data=np.array([0], dtype=np.int64))
    values, genes = MODULE.load_space_ranger_h5(h5_path, 2)
    assert genes == {"g1"}
    assert values == {("g1", "s_002um_3_4"): 2.0}


def test_mex_loader_accepts_native_uncompressed_materializer_output(tmp_path):
    mex = tmp_path / "mex"
    mex.mkdir()
    (mex / "features.tsv").write_text("g1\tg1\tGene Expression\n")
    (mex / "barcodes.tsv").write_text("s_002um_3_4-1\n")
    (mex / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n"
        "% native fixture\n"
        "1 1 1\n"
        "1 1 0.75\n"
    )
    assert MODULE.load_mex(mex, 2) == {("g1", "s_002um_3_4"): 0.75}


def test_h5_loaders_use_gene_expression_axis_when_target_sets_are_absent(tmp_path):
    h5_path = tmp_path / "gex_vendor.h5"
    with h5py.File(h5_path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("shape", data=np.array([2, 1], dtype=np.int64))
        matrix.create_dataset("barcodes", data=np.array([b"s_002um_00000_00000-1"]))
        matrix.create_dataset("indptr", data=np.array([0, 2], dtype=np.int64))
        matrix.create_dataset("indices", data=np.array([0, 1], dtype=np.int64))
        matrix.create_dataset("data", data=np.array([3, 11], dtype=np.int64))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.array([b"g1", b"antibody1"]))
        features.create_dataset(
            "feature_type", data=np.array([b"Gene Expression", b"Antibody Capture"])
        )

    values, genes = MODULE.load_space_ranger_h5(h5_path, 2)
    assert genes == {"g1"}
    assert values == {("g1", "s_002um_0_0"): 3.0}

    matrix, index, width = MODULE.load_space_ranger_sparse(h5_path, 2)
    assert index == {"g1": 0}
    assert width == 1
    assert matrix.toarray().tolist() == [[3.0]]


def test_spearman_handles_ties_deterministically():
    left = {"a": 1.0, "b": 1.0, "c": 3.0}
    right = {"a": 2.0, "b": 2.0, "c": 4.0}
    assert MODULE.spearman(left, right) == 1.0


def test_sparse_comparison_matches_legacy_metrics():
    open_values = {("g1", "b1"): 2.0, ("g2", "b2"): 1.0}
    vendor_values = {("g1", "b1"): 1.0, ("g2", "b2"): 1.0}
    expected = MODULE.compare(open_values, vendor_values)
    observed = MODULE.compare_sparse(
        sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 1.0]])),
        sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])),
    )
    assert observed.keys() == expected.keys()
    for key in observed:
        assert np.isclose(observed[key], expected[key])
