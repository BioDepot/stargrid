import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import io as scipy_io
from scipy import sparse


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "score_hd_flex_star_sr_gene_bin_residuals.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "score_hd_flex_star_sr_gene_bin_residuals", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_gene_bin_residuals_reconcile_and_use_he_fraction_weights():
    star = sparse.csr_matrix(np.array([[3.0, 0.0], [0.0, 4.0]]))
    sr = sparse.csr_matrix(np.array([[1.0, 2.0], [1.0, 2.0]]))
    summary, rows = MODULE.analyse_method(
        dataset="fixture",
        method="hard",
        gene_ids=["g1", "g2"],
        gene_names=["G1", "G2"],
        star=star,
        space_ranger=sr,
        weights={
            "cell_fraction": np.array([1.0, 0.5]),
            "nucleus_fraction": np.array([0.0, 0.5]),
            "extracellular_fraction": np.array([0.0, 0.5]),
        },
    )
    assert summary["shared_gene_bin_mass"] == 3.0
    assert summary["star_positive_gene_bin_residual_mass"] == 4.0
    assert summary["space_ranger_positive_gene_bin_residual_mass"] == 3.0
    assert summary["signed_mass_difference"] == 1.0
    assert summary["star_positive_spatial_bin_residual_mass"] == 1.0
    assert summary["space_ranger_positive_spatial_bin_residual_mass"] == 0.0
    assert summary["feature_bin_to_spatial_bin_star_positive_residual_ratio"] == 4.0
    assert summary["star_positive_residual_cell_mass_fraction"] == 0.75
    assert summary["star_positive_residual_nucleus_mass_fraction"] == 0.25
    by_gene = {row["gene_id"]: row for row in rows}
    assert by_gene["g1"]["star_positive_gene_bin_residual_mass"] == 2.0
    assert by_gene["g1"]["star_positive_residual_cell_mass_fraction"] == 1.0
    assert by_gene["g2"]["star_positive_gene_bin_residual_mass"] == 2.0
    assert by_gene["g2"]["star_positive_residual_nucleus_mass_fraction"] == 0.5


def test_prior_reconciliation_detects_axis_mismatch():
    summary = {
        "star_mass_on_space_ranger_axis": 11.0,
        "space_ranger_mass": 10.0,
        "star_positive_spatial_bin_residual_mass": 2.0,
        "space_ranger_positive_spatial_bin_residual_mass": 1.0,
    }
    prior = {
        "star_mass_on_space_ranger_axis": "12",
        "space_ranger_mass": "10",
        "star_positive_spatial_bin_residual_mass": "2",
        "space_ranger_positive_spatial_bin_residual_mass": "1",
    }
    try:
        MODULE.assert_prior_reconciliation(summary, prior)
    except ValueError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("prior axis mismatch was accepted")


def test_star_mex_loader_maps_common_axis_and_reports_outside_mass(tmp_path):
    root = tmp_path / "mex"
    root.mkdir()
    (root / "features.tsv").write_text("g1\tG1\ng2\tG2\n")
    (root / "barcodes.tsv").write_text(
        "s_008um_00000_00000-1\n"
        "s_008um_00000_00001-1\n"
        "s_008um_00000_00002-1\n"
    )
    scipy_io.mmwrite(
        root / "matrix.mtx",
        sparse.coo_matrix(
            (
                np.array([1, 2, 3, 0, 4, 5], dtype=np.int64),
                (np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 1, 2])),
            ),
            shape=(2, 3),
        ),
    )
    matrix, metadata = MODULE.load_star_mex(
        root,
        sr_gene_ids=["g2", "g1"],
        sr_coordinates=np.array([[0, 0], [0, 1]], dtype=np.int32),
        scale_um=8,
    )
    np.testing.assert_array_equal(matrix.toarray(), [[0, 4], [1, 2]])
    assert metadata["star_mass_all_axes"] == 15.0
    assert metadata["star_mass_outside_space_ranger_axis"] == 8.0
    assert metadata["star_gene_bin_nnz_outside_space_ranger_axis"] == 2
    assert metadata["star_matrix_market_explicit_zero_entries"] == 1


def test_h5ad_loader_transposes_to_gene_by_barcode(tmp_path):
    path = tmp_path / "counts.h5ad"
    matrix = sparse.csr_matrix(np.array([[1, 0], [2, 3]], dtype=np.float32))
    string = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        x = handle.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = matrix.shape
        x.create_dataset("data", data=matrix.data)
        x.create_dataset("indices", data=matrix.indices)
        x.create_dataset("indptr", data=matrix.indptr)
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.asarray(
                ["s_008um_00000_00000-1", "s_008um_00000_00001-1"],
                dtype=object,
            ),
            dtype=string,
        )
        var = handle.create_group("var")
        var.create_dataset(
            "gene_ids", data=np.asarray(["g1.1", "g2.2"], dtype=object), dtype=string,
        )
        var.create_dataset(
            "_index", data=np.asarray(["G1", "G2"], dtype=object), dtype=string,
        )
    loaded = MODULE.load_space_ranger_h5ad(path, scale_um=8)
    assert loaded.gene_ids == ["g1", "g2"]
    np.testing.assert_array_equal(loaded.matrix.toarray(), [[1, 2], [0, 3]])
