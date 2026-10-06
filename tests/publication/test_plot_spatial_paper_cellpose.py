from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
import plot_spatial_paper_cellpose as figure


def test_mass_plot_keeps_negative_effect_visible():
    fig, ax = plt.subplots()
    data = pd.DataFrame({"star_percent_excess_over_space_ranger": [-3.0, 2.0]}, index=figure.DATASETS)
    figure.panel_excess(ax, data)
    assert ax.get_ylim()[0] < -3
    assert ax.get_ylim()[1] > 2
    assert any(t.get_text() == "-3.00%" for t in ax.texts)
    plt.close(fig)


def test_concordance_axis_does_not_clip_a_regression():
    fig, ax = plt.subplots()
    data = pd.DataFrame({"gene_total_pearson": [0.9, 0.95], "spatial_bin_total_pearson": [0.8, 0.85]}, index=figure.DATASETS)
    figure.panel_concordance(ax, data)
    assert ax.get_ylim()[0] < 0.8
    assert "8 µm spatial bins" in [t.get_text() for t in ax.get_legend().texts]
    plt.close(fig)


def test_duplicate_primary_rows_are_rejected(tmp_path):
    source = tmp_path / "table.tsv"
    source.write_text("dataset\tmethod\nCRC\thard\nCRC\thard\nSPATCH\thard\n")
    with pytest.raises(ValueError, match="duplicate"):
        figure.load(source)
