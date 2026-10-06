import gzip
import importlib.util
from pathlib import Path

import h5py
import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "compare_hd_flex_h5ad_aggregates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_hd_flex_h5ad_aggregates", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def h5ad(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        x = handle.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = np.array([2, 2], dtype=np.int64)
        x.create_dataset("data", data=np.array([2, 1, 3], dtype=np.int32))
        x.create_dataset("indices", data=np.array([0, 1, 1], dtype=np.int32))
        x.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int64))
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index", data=np.array([b"s_008um_00000_00000-1", b"s_008um_00000_00001-1"])
        )
        var = handle.create_group("var")
        var.create_dataset("gene_ids", data=np.array([b"g1", b"g2"]))


def mex(path: Path) -> None:
    path.mkdir()
    (path / "features.tsv").write_text(
        "g1\tg1\tGene Expression\n"
        "g2\tg2\tGene Expression\n",
        encoding="utf-8",
    )
    (path / "barcodes.tsv").write_text(
        "s_008um_0_0-1\n"
        "s_008um_0_2-1\n",
        encoding="utf-8",
    )
    with gzip.open(path / "matrix.mtx.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "%%MatrixMarket matrix coordinate integer general\n"
            "% fixture\n"
            "2 2 2\n"
            "1 1 2\n"
            "2 2 4\n"
        )


def test_load_h5ad_totals_streams_csr(tmp_path):
    path = tmp_path / "sr.h5ad"
    h5ad(path)
    features, barcodes, genes, bins, nnz = MODULE.load_h5ad_totals(
        path, row_chunk=1
    )
    assert features == ["g1", "g2"]
    assert barcodes == [
        "s_008um_00000_00000-1",
        "s_008um_00000_00001-1",
    ]
    assert genes.tolist() == [2.0, 4.0]
    assert bins.tolist() == [3.0, 3.0]
    assert nnz == 3


def test_compare_reports_shared_and_outside_barcode_mass(tmp_path):
    h5ad_path = tmp_path / "sr.h5ad"
    mex_path = tmp_path / "mex"
    h5ad(h5ad_path)
    mex(mex_path)
    features, barcodes, genes, bins, _ = MODULE.load_h5ad_totals(
        h5ad_path, row_chunk=2
    )
    row, inputs = MODULE.compare_method(
        "hard", mex_path, 8, features, barcodes, genes, bins
    )
    assert len(inputs) == 3
    assert row["star_raw_mass"] == 6.0
    assert row["space_ranger_raw_mass"] == 6.0
    assert row["star_mass_on_space_ranger_barcodes"] == 2.0
    assert row["star_mass_outside_space_ranger_barcodes"] == 4.0
    assert row["star_barcodes_outside_space_ranger_object"] == 1
