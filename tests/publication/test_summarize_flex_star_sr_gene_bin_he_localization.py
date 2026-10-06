import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "summarize_flex_star_sr_gene_bin_he_localization.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "summarize_flex_star_sr_gene_bin_he_localization", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(gene, residual, delta, sr):
    return {
        "method": "hard",
        "gene_id": gene,
        "star_positive_gene_bin_residual_mass": str(residual),
        "signed_star_minus_space_ranger_mass": str(delta),
        "space_ranger_mass": str(sr),
    }


def test_cross_slide_gene_consistency_reports_overlap_and_correlations():
    left = [row("g1", 10, 4, 100), row("g2", 2, 1, 200), row("g3", 1, -1, 100)]
    right = [row("g1", 20, 8, 200), row("g2", 4, 2, 400), row("g3", 0, -2, 200)]
    result = MODULE.cross_slide_row("hard", "A", left, "B", right)
    assert result["common_genes"] == 3
    assert result["genes_with_positive_net_delta_on_both_slides"] == 2
    assert result["top10_positive_residual_gene_overlap"] == 3
    assert result["signed_delta_rate_pearson_sr_mass_ge_100"] == 1.0
