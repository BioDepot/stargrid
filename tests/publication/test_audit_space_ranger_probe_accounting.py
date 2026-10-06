import csv
import importlib.util
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "audit_space_ranger_probe_accounting.py"
)
SPEC = importlib.util.spec_from_file_location("probe_accounting", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_panel(path: Path) -> None:
    path.write_text(
        "#panel_name=fixture\n"
        "gene_id,probe_seq,probe_id,included,region\n"
        "g1,AAAA,g1|G1|a,TRUE,spliced\n"
        "g2,CCCC,g2|G2|b,TRUE,unspliced\n"
        "g3,GGGG,g3|G3|c,FALSE,unspliced\n",
        encoding="utf-8",
    )


def write_h5ad(path: Path) -> None:
    data = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[2, 1], [0, 3]], dtype=np.int32)),
        obs=pd.DataFrame(index=["s_008um_0_0-1", "s_008um_0_1-1"]),
        var=pd.DataFrame({"gene_ids": ["g1", "g2"]}, index=["G1", "G2"]),
    )
    data.write_h5ad(path)


def write_raw_probe_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("data", data=np.asarray([2, 3, 5], dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray([0, 1, 2], dtype=np.int64))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([b"p1", b"p2", b"p3"]))
        features.create_dataset(
            "filtered_probes", data=np.asarray([True, False, False])
        )
        target = features.create_group("target_sets")
        target.create_dataset("fixture", data=np.asarray([0, 1], dtype=np.uint32))


def test_panel_axis_and_raw_diagnostic_mass_are_distinguished(tmp_path):
    panel_path = tmp_path / "panel.csv"
    h5ad_path = tmp_path / "sr.h5ad"
    raw_path = tmp_path / "raw_probe.h5"
    write_panel(panel_path)
    write_h5ad(h5ad_path)
    write_raw_probe_h5(raw_path)

    panel = MODULE.load_filtered_probe_panel(panel_path)
    genes = set(panel.pop("eligible_genes"))
    assert genes == {"g1", "g2"}
    assert panel["excluded_gene_count"] == 1

    h5ad = MODULE.audit_h5ad(h5ad_path, genes)
    assert h5ad["axis_exact"] is True
    assert h5ad["raw_mass"] == 6

    raw = MODULE.audit_raw_probe_h5(raw_path)
    assert raw["total_raw_mass"] == 10
    assert raw["categories"]["filtered_probe_reference"]["raw_mass"] == 5
    assert (
        raw["categories"]["diagnostic_outside_filtered_probe_reference"]["raw_mass"]
        == 5
    )
    assert raw["categories"]["passed_gdna_filter"]["raw_mass"] == 2
    assert raw["categories"]["failed_gdna_filter"]["raw_mass"] == 8
    joint = raw["inclusion_by_gdna_qc"]
    assert joint["accepted_reference_passed_gdna_qc"]["raw_mass"] == 2
    assert joint["accepted_reference_failed_gdna_qc"]["raw_mass"] == 3
    assert joint["excluded_diagnostic_passed_gdna_qc"]["raw_mass"] == 0
    assert joint["excluded_diagnostic_failed_gdna_qc"]["raw_mass"] == 5
