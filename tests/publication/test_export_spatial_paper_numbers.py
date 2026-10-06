import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
from export_spatial_paper_numbers import display, displayed_number, evaluate, export
from summarize_sealed_spatial_primary import sha256


def fixture(tmp_path):
    main, numbers, source = [tmp_path / p for p in ("main.tex", "numbers.tex", "result.json")]
    main.write_text(r"\input{numbers}\gainAlias{}")
    numbers.write_text(r"\newcommand{\gain}{99}\let\gainAlias=\gain\newcommand{\unused}{123}")
    source.write_text(json.dumps({"hard": 120, "vendor": 100}))
    spec = {"status": "regenerated", "result_id": "counts", "provenance_run_id": "primary",
            "scope": "same gene and barcode axes", "display": {"kind": "decimal", "places": 1},
            "value": {"op": "product", "args": [100, {"op": "ratio", "args": [
                {"op": "difference", "args": [{"source": "counts", "path": ["hard"]}, {"source": "counts", "path": ["vendor"]}]},
                {"source": "counts", "path": ["vendor"]}]}]}}
    manifest = {"release": "v1.9.5", "sources": {"counts": {"path": str(source), "sha256": sha256(source)}},
                "macros": {"gain": spec, "gainAlias": {"status": "regenerated"}}}
    return main, numbers, source, manifest


def test_source_calculation_alias_and_unused_preservation(tmp_path):
    main, numbers, _, manifest = fixture(tmp_path)
    result = export(manifest, main, numbers, tmp_path / "out")
    assert result["publication_ready"]
    assert result["macros"]["gain"]["new_display"] == "20.0"
    assert result["macros"]["gainAlias"]["resolved_display"] == "20.0"
    assert result["macros"]["gain"]["absolute_change_from_old_display"] == -79
    assert result["macros"]["gainAlias"]["relative_change_percent_from_old_display"] == pytest.approx(-7900 / 99)
    assert result["macros"]["gainAlias"]["provenance_run_id"] == "primary"
    assert result["macros"]["unused"]["status"] == "historical_unused"
    assert r"\let\gainAlias=\gain" in (tmp_path / "out/numbers.tex").read_text()
    assert numbers.read_text().startswith(r"\newcommand{\gain}{99}")
    assert "% result_id: counts" in (tmp_path / "out/numbers.tex").read_text()


def test_new_macro_requires_explicit_binding_and_evidence(tmp_path):
    main,numbers,source,manifest=fixture(tmp_path)
    spec=dict(manifest['macros']['gain'])
    manifest['macros']['newPlacement']=spec
    with pytest.raises(ValueError,match='unknown macros'):
        export(manifest,main,numbers,tmp_path/'out')
    spec['new_macro']=True
    result=export(manifest,main,numbers,tmp_path/'out')
    assert result['macros']['newPlacement']['old_display']==''
    assert result['macros']['newPlacement']['new_display']=='20.0'
    assert r'\newcommand{\newPlacement}{20.0}' in (tmp_path/'out/numbers.tex').read_text()
    spec['value']=20
    with pytest.raises(ValueError,match='no evidence source'):
        export(manifest,main,numbers,tmp_path/'no_evidence')


def test_unmapped_live_macro_and_input_drift_fail(tmp_path):
    main, numbers, source, manifest = fixture(tmp_path)
    del manifest["macros"]["gain"]
    with pytest.raises(ValueError, match="unmapped live"):
        export(manifest, main, numbers, tmp_path / "out")
    source.write_text("{}")
    with pytest.raises(ValueError, match="hash drift"):
        export(manifest, main, numbers, tmp_path / "out")


def test_pending_decision_is_never_publication_ready(tmp_path):
    main, numbers, _, manifest = fixture(tmp_path)
    manifest["macros"] = {name: {"status": "pending_decision", "reason": "scope decision"}
                          for name in ("gain", "gainAlias")}
    result = export(manifest, main, numbers, tmp_path / "out")
    assert not result["publication_ready"]
    assert result["macros"]["gain"]["new_display"] == "99"


def test_format_does_not_hide_fractional_expected_mass():
    with pytest.raises(ValueError, match="fractional mass"):
        display(10.5, {"kind": "integer"})
    assert display(3670.1, {"kind": "duration"}) == "61:10"
    assert display(0.000122, {"kind": "scientific", "places": 2}) == r"1.22\times10^{-4}"


def test_historical_display_parsing_and_zero_denominator(tmp_path):
    assert displayed_number(r"1.22\times10^{-4}") == pytest.approx(0.000122)
    assert displayed_number(r"1\,h\,2\,min\,40\,s") == 3760
    assert displayed_number("1,234") == 1234
    assert displayed_number("-2.5") == -2.5
    assert displayed_number(r"\textsc{hard}") is None
    main, numbers, _, manifest = fixture(tmp_path)
    numbers.write_text(numbers.read_text().replace("{99}", "{0}"))
    result = export(manifest, main, numbers, tmp_path / "out")
    row = result["macros"]["gain"]
    assert row["absolute_change_from_old_display"] == 20
    assert row["relative_change_percent_from_old_display"] is None


def test_metadata_stays_derived_from_evidence():
    source = {"archive": {"assays": {"one": {}, "two": {}}}}
    expression = {"op": "length", "args": [{"source": "archive", "path": ["assays"]}]}
    assert evaluate(expression, source) == 2
    with pytest.raises(ValueError, match="length requires"):
        evaluate({"op": "length", "args": ["two"]}, source)
    assert display("4.1.1", {"kind": "text", "prefix": "Cellpose "}) == "Cellpose 4.1.1"


def test_raw_historical_changes_are_distinct_from_rounded_display(tmp_path):
    main, numbers, source, manifest = fixture(tmp_path)
    source.write_text(json.dumps({"hard": 120, "vendor": 100, "old_gain": 98.75}))
    manifest["sources"]["counts"]["sha256"] = sha256(source)
    manifest["macros"]["gain"]["old_evidence"] = {
        "result_id": "historical_counts", "provenance_run_id": "original_primary",
        "value": {"source": "counts", "path": ["old_gain"]}}
    result = export(manifest, main, numbers, tmp_path / "out")
    for name in ("gain", "gainAlias"):
        row = result["macros"][name]
        assert row["old_raw_value"] == 98.75
        assert row["absolute_change"] == -78.75
        assert row["absolute_change_from_old_display"] == -79
        assert row["old_provenance_run_id"] == "original_primary"
    with (tmp_path / "out/changes.tsv").open() as handle:
        rows = {row["macro"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert rows["gain"]["old_raw_value"] == "98.75"
    assert float(rows["gain"]["new_raw_value"]) == 20
    assert rows["gain"]["old_result_id"] == "historical_counts"


def test_missing_historical_evidence_is_explicit(tmp_path):
    main, numbers, _, manifest = fixture(tmp_path)
    result = export(manifest, main, numbers, tmp_path / "out")
    row = result["macros"]["gain"]
    assert row["old_raw_value"] is None
    assert "displayed value only" in row["old_evidence_status"]
    assert "absolute_change" not in row


def test_historical_raw_values_require_recorded_evidence(tmp_path):
    main, numbers, _, manifest = fixture(tmp_path)
    manifest["macros"]["gain"]["old_evidence"] = {
        "result_id": "historical", "provenance_run_id": "historical", "value": 99}
    with pytest.raises(ValueError, match="historical raw value has no evidence"):
        export(manifest, main, numbers, tmp_path / "out")


def test_forward_aliases_and_chains_follow_their_tex_targets(tmp_path):
    main, numbers, _, manifest = fixture(tmp_path)
    main.write_text(r"\input{numbers}\aAlias{}")
    numbers.write_text(r"\newcommand{\zValue}{99}\let\bAlias=\zValue\let\aAlias=\bAlias"
                      r"\newcommand{\zUnused}{123}\let\aUnused=\zUnused")
    manifest["macros"] = {"zValue": manifest["macros"]["gain"],
                          "aAlias": {"status": "regenerated"},
                          "bAlias": {"status": "regenerated"}}
    export(manifest, main, numbers, tmp_path / "out")
    tex = (tmp_path / "out/numbers.tex").read_text()
    assert tex.index(r"\newcommand{\zValue}") < tex.index(r"\let\bAlias") < tex.index(r"\let\aAlias")
    assert tex.index(r"\newcommand{\zUnused}") < tex.index(r"\let\aUnused")
