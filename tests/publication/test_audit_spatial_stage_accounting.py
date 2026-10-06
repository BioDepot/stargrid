import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
from audit_spatial_stage_accounting import DECODING_FIELDS, FLEX_FIELDS, audit
from summarize_sealed_spatial_primary import sha256


def fixture(root, **changes):
    spatial = root / "star/SpatialFlex.out"
    spatial.mkdir(parents=True)
    (spatial / "RUN_COMPLETE").touch()
    values = {key: "100" for key in DECODING_FIELDS + FLEX_FIELDS}
    values.update(schema="star_suite.spatial_flex_integrated.v1", source_revision="revision",
                  feature_assigned_reads="90")
    values.update(changes)
    summary = spatial / "run_summary.tsv"
    summary.write_text("".join(f"{k}\t{v}\n" for k, v in values.items()))
    seal = {"status": "sealed", "star_commit": "revision", "native_summary_sha256": sha256(summary)}
    for name in ("PRIMARY_SEAL.json", "RECIPE_COMPLETE.json"):
        (root / name).write_text(json.dumps(seal))
    return root


def test_nonexact_assignment_change_is_not_a_decoder_change(tmp_path):
    baseline = fixture(tmp_path / "old")
    primary = fixture(tmp_path / "new", feature_assigned_reads="80")
    result = audit(baseline, primary)
    assert result["passed"]
    assert result["inputs"]["primary"]["summary_sha256"]


def test_decoding_drift_fails_with_the_exact_field(tmp_path):
    result = audit(fixture(tmp_path / "old"), fixture(tmp_path / "new", reads_with_candidates="99"))
    assert not result["passed"]
    assert result["mismatches"] == {"reads_with_candidates": {"baseline": "100", "primary": "99"}}


def test_modified_summary_is_not_accepted_as_evidence(tmp_path):
    baseline, primary = fixture(tmp_path / "old"), fixture(tmp_path / "new")
    with (primary / "star/SpatialFlex.out/run_summary.tsv").open("a") as handle:
        handle.write("extra\t1\n")
    with pytest.raises(ValueError, match="hash drift"):
        audit(baseline, primary)
