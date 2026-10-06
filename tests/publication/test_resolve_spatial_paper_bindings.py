import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
from resolve_spatial_paper_bindings import resolve
from summarize_sealed_spatial_primary import sha256


def fixture(tmp_path):
    root = tmp_path / "analyses"
    step = root / "counts"
    step.mkdir(parents=True)
    source = step / "result.tsv"
    source.write_text("policy\tcount\nhard\t120\n")
    record = {"status": "complete", "exit_code": 0, "result_id": "counts_result",
              "output_directory": str(source), "outputs": {str(source): sha256(source)}}
    (step / "COMPLETE.json").write_text(json.dumps(record))
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"cases": [{"primary_run_id": "20260915_v195_crc"}]}))
    base = {"release": "v1.9.5", "sources": {}, "macros": {}}
    remaining = {
        "sources": {"counts": {"analysis_step": "counts", "relative_path": "", "format": "tsv"},
                    "timing": {"expected_cases": ["crc"]}},
        "macros": {"mass": {"value": {"source": "counts", "path": [0, "count"]},
                            "display": {"kind": "integer"}}},
        "row_label_checks": {"counts": {"0": {"policy": "hard"}}},
    }
    return base, remaining, root, timing, sha256(timing)


def test_resolves_file_result_and_pins_acceptance(tmp_path):
    result = resolve(*fixture(tmp_path))
    assert result["sources"]["counts"]["result_id"] == "counts_result"
    assert result["sources"]["counts"]["complete_record_sha256"]
    assert result["binding_resolution"]["scientific_analysis_executed"] is False


@pytest.mark.parametrize("defect", ["source", "status", "row", "timing", "collision"])
def test_rejects_unaccepted_or_misbound_evidence(tmp_path, defect):
    args = fixture(tmp_path)
    base, remaining, root, timing, _ = args
    if defect == "source":
        (root / "counts/result.tsv").write_text("policy\tcount\nhard\t121\n")
    elif defect == "status":
        p = root / "counts/COMPLETE.json"
        record = json.loads(p.read_text()); record["status"] = "failed"
        p.write_text(json.dumps(record))
    elif defect == "row":
        remaining["row_label_checks"]["counts"]["0"]["policy"] = "strict"
    elif defect == "timing":
        remaining["sources"]["timing"]["expected_cases"] = ["spatch"]
    else:
        base["sources"]["counts"] = {}
    with pytest.raises(ValueError):
        resolve(*args)
