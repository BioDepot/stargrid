import sys
import csv
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
from compare_hd_flex_cross_reference_h5ad import compare_aggregates


def example():
    star = (["g2", "g3", "g4"], ["s_008um_0_0-1", "s_008um_0_2-1"],
            np.array([6., 9., 5.]), np.array([12., 8.]), 5, [])
    vendor = (["g1", "g2", "g3"], ["s_008um_00000_00000-1", "s_008um_00000_00001-1"],
              np.array([2., 4., 8.]), np.array([5., 9.]), 4)
    return star, vendor


def test_unmatched_reference_mass_is_preserved_and_disclosed():
    star, vendor = example()
    row, axes = compare_aggregates("hard", 8, star, vendor)
    assert row["star_raw_mass"] == 20
    assert row["space_ranger_raw_mass"] == 14
    assert row["star_mass_on_shared_features"] == 15
    assert row["space_ranger_mass_on_shared_features"] == 12
    assert row["star_mass_on_features_absent_from_space_ranger"] == 5
    assert row["absent_feature_space_ranger_mass"] == 2
    assert row["star_mass_on_space_ranger_barcodes"] == 12
    assert row["star_mass_outside_space_ranger_barcodes"] == 8
    assert axes["star_only_feature_ids"] == ["g4"]
    assert axes["space_ranger_only_feature_ids"] == ["g1"]
    assert row["gene_total_pearson"] == pytest.approx(np.corrcoef([0,6,9,5], [2,4,8,0])[0,1])
    assert row["spatial_bin_total_pearson_on_space_ranger_axis"] == pytest.approx(np.corrcoef([12,0], [5,9])[0,1])


def test_reference_axis_order_does_not_change_metrics():
    star, vendor = example()
    expected = compare_aggregates("hard", 8, star, vendor)
    reordered = (list(reversed(star[0])), star[1], star[2][::-1], *star[3:])
    assert compare_aggregates("hard", 8, reordered, vendor) == expected


def test_shared_axis_has_no_structural_mass():
    star, vendor = example()
    vendor = (star[0], *vendor[1:])
    row, axes = compare_aggregates("hard", 8, star, vendor)
    assert row["star_mass_on_features_absent_from_space_ranger"] == 0
    assert row["absent_feature_space_ranger_mass"] == 0
    assert row["gene_total_pearson"] == row["gene_total_pearson_on_shared_features"]
    assert axes["star_only_feature_ids"] == axes["space_ranger_only_feature_ids"] == []


def test_cli_preserves_unmatched_counts_from_h5ad_and_mex(tmp_path):
    vendor = tmp_path / "vendor.h5ad"
    with h5py.File(vendor, "w") as handle:
        x = handle.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = [2, 3]
        x.create_dataset("data", data=[2, 3, 1, 8])
        x.create_dataset("indices", data=[0, 1, 1, 2])
        x.create_dataset("indptr", data=[0, 2, 4])
        handle.create_dataset("var/gene_ids", data=[b"g1.1", b"g2.1", b"g3.1"])
        handle.create_dataset("obs/_index", data=[b"s_008um_00000_00000-1", b"s_008um_00000_00001-1"])
    mex = tmp_path / "mex"
    mex.mkdir()
    (mex / "features.tsv").write_text("g2.9\tG2\tGene Expression\ng3.9\tG3\tGene Expression\ng4.9\tG4\tGene Expression\n")
    (mex / "barcodes.tsv").write_text("s_008um_0_0-1\ns_008um_0_2-1\n")
    (mex / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n3 2 6\n1 1 4\n2 1 6\n3 1 2\n1 2 2\n2 2 3\n3 2 3\n")
    out = tmp_path / "result"
    source = Path(__file__).resolve().parents[2] / "scripts/publication/compare_hd_flex_cross_reference_h5ad.py"
    subprocess.run([sys.executable, str(source), "--vendor-h5ad", str(vendor),
                    "--method", f"hard={mex}", "--out-dir", str(out)],
                   check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2"))
    row = next(csv.DictReader((out / "aggregate_concordance.tsv").open(), delimiter="\t"))
    assert float(row["star_raw_mass"]) == 20
    assert float(row["star_mass_on_features_absent_from_space_ranger"]) == 5
    assert float(row["absent_feature_space_ranger_mass"]) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["feature_axis_audit"]["hard"]["star_only_feature_ids"] == ["g4"]
