from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts/publication/summarize_adjacent_xenium_top_genes.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_adjacent_xenium_top_genes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_top_genes_use_xenium_order_and_identical_axes(tmp_path: Path) -> None:
    rows = []
    for method, delta in [("space_ranger", 0.0), ("soft", 0.1)]:
        for gene, reference, correlation in [("g1", 10, 0.2), ("g2", 30, 0.4), ("g3", 20, 0.3)]:
            rows.append(
                {
                    "method": method,
                    "gene_id": gene,
                    "xenium_counts": reference,
                    "visium_counts": reference + 1,
                    "spatial_pearson": correlation + delta,
                }
            )
    metrics = tmp_path / "metrics.tsv"
    pd.DataFrame(rows).to_csv(metrics, sep="\t", index=False)
    diffexp = tmp_path / "diffexp.csv"
    pd.DataFrame(
        {"Feature ID": ["g1", "g2", "g3"], "Feature Name": ["A", "B", "C"]}
    ).to_csv(diffexp, index=False)

    top, summary = MODULE.summarize(metrics, diffexp, 2)

    assert [row["gene_name"] for row in top] == ["B", "C"]
    assert summary[0]["median_spatial_pearson"] == 0.35
    assert summary[1]["median_spatial_pearson"] == 0.45


def test_panel_selection_uses_declared_source_category(tmp_path: Path) -> None:
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "payload": {
                    "targets": [
                        {"source": {"category": "base"}, "type": {"descriptor": "gene", "data": {"name": "A"}}},
                        {"source": {"category": "current"}, "type": {"descriptor": "gene", "data": {"name": "B"}}},
                        {"source": {"category": "current"}, "type": {"descriptor": "gene", "data": {"name": "C"}}},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert MODULE.panel_gene_names(panel, "current") == {"B", "C"}
