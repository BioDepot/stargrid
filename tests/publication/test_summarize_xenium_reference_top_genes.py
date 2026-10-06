import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts/publication/summarize_xenium_reference_top_genes.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_xenium_reference_top_genes", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_top_gene_axis_is_shared_and_ranked_by_xenium_mass(tmp_path, monkeypatch):
    source = tmp_path / "metrics.tsv"
    rows = []
    for method, factor in (("space_ranger", 1), ("star_hard", 2)):
        for gene_id, name, xenium, visium, corr in (
            ("ENSG1", "A", 20, 3, 0.2),
            ("ENSG2", "B", 50, 5, 0.4),
            ("ENSG3", "C", 10, 7, 0.6),
        ):
            rows.append(
                {
                    "method": method,
                    "gene_id": gene_id,
                    "gene_name": name,
                    "xenium_counts": xenium,
                    "visium_counts": visium * factor,
                    "spatial_pearson_zero_if_undefined": corr,
                }
            )
    pd.DataFrame(rows).to_csv(source, sep="\t", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--reference-gene-set-metrics", str(source), "--top-n", "2", "--out-dir", str(out)],
    )
    MODULE.main()
    detail = pd.read_csv(out / "top_genes.tsv", sep="\t")
    assert detail["gene_id"].tolist() == ["ENSG2", "ENSG1"]
    assert detail["star_hard_visium_counts"].tolist() == [10.0, 6.0]
