from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/publication/score_hd_spatch_codex.py"


def write_probe_csv(path: Path) -> None:
    path.write_text(
        "#probe_set_file_format=3.0\n"
        "gene_id,probe_seq,probe_id,included,region,gene_name\n"
        "ENSG_CD8A," + "A" * 50 + ",p1,TRUE,spliced,CD8A\n"
        "ENSG_MS4A1," + "C" * 50 + ",p2,TRUE,spliced,MS4A1\n",
        encoding="utf-8",
    )


def write_mex(path: Path) -> None:
    path.mkdir()
    (path / "features.tsv").write_text(
        "ENSG_CD8A\tENSG_CD8A\tGene Expression\n"
        "ENSG_MS4A1\tENSG_MS4A1\tGene Expression\n"
        "ENSG_EXTRA\tENSG_EXTRA\tGene Expression\n",
        encoding="utf-8",
    )
    (path / "barcodes.tsv").write_text(
        "s_008um_0_0-1\n"
        "s_008um_0_1-1\n"
        "s_008um_1_0-1\n"
        "s_008um_1_1-1\n",
        encoding="utf-8",
    )
    matrix = sparse.coo_matrix(np.asarray([
        [0, 1, 2, 8],
        [8, 2, 1, 0],
        [3, 3, 3, 3],
    ], dtype=np.int64))
    mmwrite(path / "matrix.mtx", matrix, comment="synthetic")


def write_h5ads(rna_path: Path, codex_path: Path) -> None:
    barcodes = [
        "s_008um_0_0-1", "s_008um_0_1-1",
        "s_008um_1_0-1", "s_008um_1_1-1",
    ]
    spatial = np.asarray([[10, 20], [10, 28], [18, 20], [18, 28]], dtype=float)
    rna = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([
            [0, 8], [1, 2], [2, 1], [8, 0],
        ], dtype=np.int64)),
        obs=pd.DataFrame({"codex_common": [1, 1, 1, 1]}, index=barcodes),
        var=pd.DataFrame(
            {"gene_ids": ["ENSG_CD8A", "ENSG_MS4A1"]},
            index=["CD8A", "MS4A1"],
        ),
    )
    rna.obsm["spatial"] = spatial
    rna.write_h5ad(rna_path)

    proteins = [
        "CD8", "CD20", "CD3e", "CD56", "Pan-Cytokeratin", "CD4", "CD34", "SMA",
        "FOXP3", "CD163", "HLA-A", "CD11c", "MPO", "CD68", "HLA-DR", "IDO1",
    ]
    values = np.zeros((4, len(proteins)), dtype=float)
    values[:, 0] = [0, 1, 2, 8]
    values[:, 1] = [8, 2, 1, 0]
    codex = ad.AnnData(
        X=values,
        obs=pd.DataFrame({"codex_common": [1, 1, 1, 1]}, index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=proteins),
    )
    codex.obsm["spatial"] = spatial
    codex.write_h5ad(codex_path)


def run_score(tmp_path: Path, name: str) -> Path:
    rna = tmp_path / "rna.h5ad"
    codex = tmp_path / "codex.h5ad"
    probe = tmp_path / "probes.csv"
    mex = tmp_path / "mex"
    if not rna.exists():
        write_h5ads(rna, codex)
        write_probe_csv(probe)
        write_mex(mex)
    output = tmp_path / name
    subprocess.run([
        sys.executable, str(SCRIPT),
        "--sr-h5ad", str(rna),
        "--codex-h5ad", str(codex),
        "--probe-csv", str(probe),
        "--sr-filtered-probe-set", str(probe),
        "--method", f"star_2024a_hard={mex}",
        "--bin-size", "8",
        "--protein-positive-quantile", "0.5",
        "--out-dir", str(output),
    ], check=True)
    return output


def test_spatch_codex_generator_is_deterministic_and_mass_preserving(tmp_path: Path) -> None:
    first = run_score(tmp_path, "first")
    second = run_score(tmp_path, "second")
    for name in (
        "marker_metrics.tsv", "method_summary.tsv", "raw_mass.tsv",
        "marker_axis_coverage.tsv", "space_ranger_accounting.tsv",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    with (first / "raw_mass.tsv").open(newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert float(rows["space_ranger_2020a"]["full_raw_mass"]) == 22
    assert float(rows["star_2024a_hard"]["full_raw_mass"]) == 34
    assert float(rows["star_2024a_hard"]["space_ranger_2020a_shared_axis_mass"]) == 22

    with (first / "marker_metrics.tsv").open(newline="") as handle:
        metrics = list(csv.DictReader(handle, delimiter="\t"))
    cd8 = next(row for row in metrics if row["method"] == "star_2024a_hard" and row["protein"] == "CD8")
    cd20 = next(row for row in metrics if row["method"] == "star_2024a_hard" and row["protein"] == "CD20")
    hla_a = next(row for row in metrics if row["method"] == "star_2024a_hard" and row["protein"] == "HLA-A")
    assert float(cd8["roc_auc_top_protein_quantile"]) == 1.0
    assert float(cd20["roc_auc_top_protein_quantile"]) == 1.0
    assert cd8["reference_marker_available"] == "True"
    assert hla_a["reference_marker_available"] == "False"
    assert hla_a["roc_auc_top_protein_quantile"] == "NA"

    with (first / "method_summary.tsv").open(newline="") as handle:
        summaries = list(csv.DictReader(handle, delimiter="\t"))
    star_summary = next(row for row in summaries if row["method"] == "star_2024a_hard")
    assert int(star_summary["markers_scored"]) == 2

    with (first / "space_ranger_accounting.tsv").open(newline="") as handle:
        accounting = next(csv.DictReader(handle, delimiter="\t"))
    assert accounting["filtered_panel_axis_exact"] == "True"
    assert accounting["diagnostic_non_panel_gene_features"] == "0"
