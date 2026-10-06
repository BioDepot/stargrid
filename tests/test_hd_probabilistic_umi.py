from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from star_spatial.hd_probabilistic_umi import (
    CandidateRead, build_read_cliques, collapse_clique_to_unit,
    corrected_umi_maps, factorized_h0_read_log_prior, gated_hard_calls,
    mass_summary, strict_counts, weighted_occupancies,
)


def read(read_id: str, candidates: dict[str, float], umi: str = "AAAA") -> CandidateRead:
    return CandidateRead(read_id, "gene", umi, candidates)


def test_read_clique_requires_global_candidate_intersection_and_blocks_overlap_chain() -> None:
    cliques = build_read_cliques([
        read("a", {"s_002um_0_0": 0, "s_002um_0_1": 0}),
        read("b", {"s_002um_0_1": 0, "s_002um_0_2": 0}),
        read("c", {"s_002um_0_2": 0, "s_002um_0_3": 0}),
    ])
    assert len(cliques) == 2
    assert all(clique.candidates for clique in cliques)
    assert sorted(len(clique.read_ids) for clique in cliques) == [1, 2]


def test_read_clique_is_permutation_deterministic_and_normalized() -> None:
    rows = [read("b", {"s_002um_0_0": -1, "s_002um_0_1": 0}),
            read("a", {"s_002um_0_0": 0, "s_002um_0_1": 0})]
    left = build_read_cliques(rows, temperature=2)
    right = build_read_cliques(reversed(rows), temperature=2)
    assert left == right
    assert math.fsum(left[0].posterior) == pytest.approx(1)
    assert left[0].probabilities()["s_002um_0_1"] > 0.5


def test_raw_read_frequency_prior_is_applied_once_per_pcr_clique() -> None:
    candidates = {"s_002um_0_0": 0.0, "s_002um_0_1": 0.0}
    cliques = build_read_cliques(
        [read("pcr1", candidates), read("pcr2", candidates)],
        log_read_prior={"s_002um_0_0": math.log(9), "s_002um_0_1": math.log(1)},
    )
    assert len(cliques) == 1
    assert cliques[0].probabilities()["s_002um_0_0"] == pytest.approx(0.9)
    assert cliques[0].log_read_priors == pytest.approx((math.log(9), math.log(1)))


def test_phred_likelihoods_accumulate_but_read_prior_does_not_repeat() -> None:
    candidates = {"s_002um_0_0": math.log(2), "s_002um_0_1": 0.0}
    clique = build_read_cliques(
        [read("pcr1", candidates), read("pcr2", candidates)],
        log_read_prior={"s_002um_0_0": 0.0, "s_002um_0_1": math.log(4)},
    )[0]
    # Two 2:1 likelihood ratios and one 1:4 prior ratio cancel exactly.
    assert clique.posterior == pytest.approx((0.5, 0.5))


def test_factorized_prior_uses_raw_exact_h0_read_counts() -> None:
    counts = {("BC1", 3): 8, ("BC2", 7): 3}
    assert factorized_h0_read_log_prior(3, 7, counts, alpha=1) == pytest.approx(
        math.log(9) + math.log(4)
    )


def test_pcr_reads_contribute_one_read_clique_mass_not_one_umi_each() -> None:
    cliques = build_read_cliques([
        read("pcr1", {"s_002um_1_1": 0}), read("pcr2", {"s_002um_1_1": 0})
    ])
    assert len(cliques) == 1
    assert strict_counts(cliques) == {("gene", "s_002um_1_1"): 1}
    assert mass_summary(cliques, weighted_occupancies(cliques))["posterior_mass"] == pytest.approx(1)


def test_weighted_duplicate_occupancy_is_bounded_and_not_renormalized() -> None:
    cliques = build_read_cliques([
        read("one", {"s_002um_0_0": 0, "s_002um_0_1": 0}, "AAAA"),
        read("two", {"s_002um_0_0": 0, "s_002um_0_1": 0}, "AAAA"),
    ])
    # Different read IDs with identical feature/UMI/intersection form one read clique.
    rows = weighted_occupancies(cliques)
    assert [row.occupancy for row in rows] == pytest.approx([0.5, 0.5])
    assert mass_summary(cliques, rows)["deduplicated_mass"] == pytest.approx(0)


def test_separate_read_cliques_in_same_umi_clique_use_one_minus_product() -> None:
    cliques = build_read_cliques([
        read("one", {"s_002um_0_0": 0, "s_002um_0_1": 0}, "AAAA"),
        read("two", {"s_002um_0_0": 0, "s_002um_0_2": 0}, "AAAT"),
    ])
    row = next(row for row in weighted_occupancies(cliques, umi_mode="1mm_cr") if row.candidate == "s_002um_0_0")
    assert row.occupancy == pytest.approx(1 - (1 - 0.5) * (1 - 0.5))


def test_directional_1mm_correction_uses_deterministic_dominant_root() -> None:
    cliques = build_read_cliques([
        read("root", {"s_002um_0_0": 0}, "AAAA"),
        read("child", {"s_002um_0_0": 0}, "AAAT"),
    ])
    mapping = corrected_umi_maps(cliques, mode="1mm_cr")
    assert mapping[("gene", "s_002um_0_0", "AAAA")] == "AAAA"
    assert mapping[("gene", "s_002um_0_0", "AAAT")] == "AAAA"


def test_same_parent_collapse_conserves_mass() -> None:
    clique = build_read_cliques([read("x", {"s_002um_0_0": 0, "s_002um_1_1": 0})])[0]
    assert collapse_clique_to_unit(clique, 8) == {"s_008um_0_0": pytest.approx(1)}


def test_hard_assignment_obeys_posterior_and_margin_gates() -> None:
    cliques = build_read_cliques([read("x", {"s_002um_0_0": 0, "s_002um_0_1": -5})])
    assert gated_hard_calls(cliques, min_posterior=.95, min_margin=.9)[0].status == "assigned"
    assert gated_hard_calls(cliques, min_posterior=.999, min_margin=.999)[0].status == "deferred"


def test_reference_cli_materializes_strict_soft_and_gated_outputs(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.tsv"
    candidates.write_text(
        "read_id\tgene\tumi\tcandidate_row2\tcandidate_col2\tlog_sequence_likelihood\n"
        "r1\tgene\tAAAA\t0\t0\t0\n"
        "r1\tgene\tAAAA\t0\t1\t-5\n"
        "r2\t\tCCCC\t0\t0\t0\n"
    )
    prior = tmp_path / "h0_prior.tsv"
    prior.write_text(
        "barcode_half\toligo_index\texact_h0_read_count\n"
        "BC1\t0\t9\n"
        "BC1\t1\t1\n"
        "BC2\t0\t10\n"
    )
    out = tmp_path / "out"
    subprocess.run([
        sys.executable, "scripts/run_hd_probabilistic_umi_oracle.py",
        "--candidate-reads", str(candidates), "--h0-read-prior", str(prior),
        "--out-dir", str(out),
    ], check=True, env={**os.environ, "PYTHONPATH": "src"})
    expected = {
        "candidate_molecule_ledger.tsv", "weighted_umi_occupancy.tsv",
        "strict_2um.tsv", "soft_expected_2um.tsv", "gated_hard_2um.tsv",
        "resolved_runtime_config.json", "probabilistic_resolver_summary.json",
    }
    assert {path.name for path in out.iterdir()} == expected
    ledger_header = (out / "candidate_molecule_ledger.tsv").read_text().splitlines()[0]
    assert "log_sequence_likelihood_sum" in ledger_header
    assert "log_exact_h0_read_prior" in ledger_header
    config = __import__("json").loads((out / "resolved_runtime_config.json").read_text())
    assert config["h0_read_prior_units"] == "raw_exact_h0_reads_including_pcr_amplification"
    assert config["h0_read_prior_application"] == "once_per_read_clique_shared_latent_coordinate"
    summary = __import__("json").loads(
        (out / "probabilistic_resolver_summary.json").read_text()
    )
    assert summary["candidate_rows_seen"] == 3
    assert summary["candidate_row_exclusion_reasons"] == {"missing_feature": 1}
    assert summary["read_ids_excluded_non_countable"] == 1
