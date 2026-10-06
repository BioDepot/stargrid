from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


SCRIPTS = Path(__file__).parents[2] / "scripts/publication"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "compare_ovarian_panel_in_xenium_cancer_cells.py"
SPEC = importlib.util.spec_from_file_location("compare_ovarian_panel_in_xenium_cancer_cells", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cancer_membership_requires_cluster_and_marker_evidence() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        )
    )
    observed = MODULE.cancer_cell_membership(
        matrix,
        ["MUC16", "PAX8", "EPCAM"],
        np.asarray([1, 1, 7, 1]),
        (1,),
        ("MUC16", "PAX8", "EPCAM"),
    )
    np.testing.assert_array_equal(observed, [True, True, False, False])


def test_tumor_panel_metrics_reward_reference_matching_signal() -> None:
    small = np.asarray(
        [
            [20, 18, 1, 2],
            [1, 2, 20, 18],
            [12, 11, 3, 2],
            [2, 2, 2, 2],
        ],
        dtype=float,
    )
    xenium = np.tile(small, 10)
    matching = xenium.copy()
    reversed_counts = np.tile(small[:, ::-1], 10)
    rich = np.tile(np.asarray([True, True, False, False]), 10)
    poor = ~rich
    summaries, rows = MODULE.evaluate_scope(
        "test",
        {"space_ranger": reversed_counts, "star": matching},
        xenium,
        ["g1", "g2", "g3", "g4"],
        ["G1", "G2", "G3", "G4"],
        np.asarray([True, True, True, False]),
        rich,
        poor,
        17,
        100,
    )
    by_method = {row["method"]: row for row in summaries}
    assert by_method["xenium"]["effect_pearson_vs_xenium"] == pytest.approx(1.0)
    assert by_method["star"]["effect_pearson_vs_xenium"] == pytest.approx(1.0)
    assert by_method["star"]["mean_reference_oriented_gene_auc"] == pytest.approx(1.0)
    assert by_method["space_ranger"]["mean_reference_oriented_gene_auc"] == pytest.approx(0.0)
    assert by_method["star"]["gene_auc_wins_vs_space_ranger"] == 3
    assert by_method["star"]["mean_positive_gene_auc_delta_vs_space_ranger"] == pytest.approx(1.0)
    assert by_method["star"]["raw_cancer_rich_panel_mass_ratio_vs_space_ranger"] == pytest.approx(
        64 / 46
    )
    assert len(rows) == 9


def test_comma_parsers_are_canonical() -> None:
    assert MODULE.comma_ints("3,1,3") == (1, 3)
    assert MODULE.comma_names("MUC16,PAX8,MUC16") == ("MUC16", "PAX8")


def test_full_reference_scope_can_report_a_previously_excluded_gene() -> None:
    small = np.asarray(
        [[8, 7, 0, 0], [1, 1, 2, 2], [2, 3, 1, 1]], dtype=float
    )
    xenium = np.tile(small, 10)
    rich = np.tile(np.asarray([True, True, False, False]), 10)
    poor = ~rich
    summaries, rows = MODULE.evaluate_scope(
        "full_reference_panel",
        {"space_ranger": xenium.copy(), "star": xenium.copy()},
        xenium,
        ["mecom", "other", "third"],
        ["MECOM", "OTHER", "THIRD"],
        np.asarray([True, True, True]),
        rich,
        poor,
        17,
        100,
    )
    assert {row["scope"] for row in summaries} == {"full_reference_panel"}
    assert {row["gene_name"] for row in rows} == {"MECOM", "OTHER", "THIRD"}
