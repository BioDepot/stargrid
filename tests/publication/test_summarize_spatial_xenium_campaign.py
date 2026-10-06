import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
import summarize_spatial_xenium_campaign as report


def test_exact_probability_and_changed_win_counts():
    assert report.exact_binomial_two_sided(16, 20) == pytest.approx(0.01181793212890625)
    assert report.exact_binomial_two_sided(10, 20) == 1
    assert report.exact_binomial_two_sided(4, 20) == report.exact_binomial_two_sided(16, 20)


def test_top20_reports_an_unfavorable_result_and_accepts_extra_audit_columns():
    rows = [dict(xenium_abundance_rank=i+1, gene_id=f"g{i}",
                 space_ranger_spatial_pearson=0.5, strict_spatial_pearson=0.4,
                 hard_spatial_pearson=0.3, soft_expected_spatial_pearson=0.6,
                 space_ranger_visium_counts=10, strict_visium_counts=8,
                 hard_visium_counts=12) for i in range(20)]
    summaries = [dict(method=name, top_gene_count=20, median_spatial_pearson=corr,
                      summed_visium_counts=mass) for name, corr, mass in
                 [("space_ranger", 0.5, 200), ("strict", 0.4, 160), ("hard", 0.3, 240)]]
    result = report.summarize_top20(rows, summaries)
    assert result["comparisons"]["hard"]["wins_vs_space_ranger"] == 0
    assert result["comparisons"]["hard"]["losses_vs_space_ranger"] == 20
    assert result["comparisons"]["hard"]["mean_spatial_pearson_delta_vs_space_ranger"] == pytest.approx(-0.2)
    rows[0]["gene_id"] = rows[1]["gene_id"]
    with pytest.raises(ValueError, match="duplicate"):
        report.summarize_top20(rows, summaries)


def cancer_payload():
    return dict(status="pass", parameters={"patch_um": 128}, axes={},
                input_sha256={"vendor.h5": "correct"}, methods=[
        dict(method=policy, scope=scope,
             mean_oriented_gene_auc_delta_vs_space_ranger=0.02,
             mean_oriented_gene_auc_delta_vs_space_ranger_ci95_low=0.01,
             mean_oriented_gene_auc_delta_vs_space_ranger_ci95_high=0.03)
        for policy in report.POLICIES
        for scope in ("full_reference_panel", "exclude_mecom", "label_leakage_controlled")])


def test_non_null_intervals_are_reported_instead_of_rejected():
    payload = cancer_payload()
    result = report.summarize_cancer(payload, "correct")
    assert result["policy_scope_interval_count"] == 6
    assert result["intervals_spanning_zero"] == 0
    assert not result["all_policy_scope_bootstrap_intervals_span_zero"]
    with pytest.raises(ValueError, match="different vendor"):
        report.summarize_cancer(payload, "old-exon-only")
    duplicate = copy.deepcopy(payload)
    duplicate["methods"][-1] = duplicate["methods"][0]
    with pytest.raises(ValueError, match="duplicate"):
        report.summarize_cancer(duplicate, "correct")
