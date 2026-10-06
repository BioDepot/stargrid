from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/publication/summarize_hd_flex_100k_biology_sanity.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(value: float, eligibility: str) -> dict:
    return {
        "total_molecule_mass": 100 + value,
        "occupied_bins": 10 + value,
        "classification": {
            "in_cell": {"roc_auc": value, "average_precision": value},
            "in_nucleus": {"roc_auc": value, "average_precision": value},
        },
        "compartment_support": {"extracellular": {"mass_fraction": value}},
        "distribution_fit": {
            "in_cell": {"normalized_total_variation": value},
            "in_nucleus": {"normalized_total_variation": value},
        },
        "parent_image_support": {
            "all_parents": {"cell_fraction_pearson": value, "nucleus_fraction_pearson": value},
            "cell_supported_fine_scale": {
                "global_supported_mass_fraction": value,
                "parents_with_image_empty_unique_max_fraction": value,
            },
            "nucleus_supported_fine_scale": {
                "global_supported_mass_fraction": value,
                "parents_with_image_empty_unique_max_fraction": value,
            },
        },
    }


def write_method_metrics(path: Path, methods: dict[str, dict]) -> None:
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "eligibility_surface"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {"method": name, "eligibility_surface": value["eligibility_surface"]}
            for name, value in methods.items()
        )


def write_increment_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    star_molecules = tmp_path / "star_molecules.tsv.gz"
    with gzip.open(star_molecules, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("product", "unit_2um"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows((
            {"product": "strict", "unit_2um": "s_002um_0_0"},
            {"product": "postcollapse_hard", "unit_2um": "s_002um_0_0"},
            {"product": "postcollapse_hard", "unit_2um": "s_002um_0_1"},
            {"product": "gated_hard", "unit_2um": "s_002um_0_0"},
            {"product": "gated_hard", "unit_2um": "s_002um_0_1"},
        ))
    soft_expected = tmp_path / "soft_expected_2um.tsv.gz"
    with gzip.open(soft_expected, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("unit_2um", "expected_count"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows((
            {"unit_2um": "s_002um_0_0", "expected_count": "1"},
            {"unit_2um": "s_002um_0_1", "expected_count": "1"},
        ))
    barcode_mappings = tmp_path / "barcode_mappings.parquet"
    pq.write_table(
        pa.table({
            "square_002um": [
                "s_002um_0_0",
                "s_002um_0_1",
                "s_002um_1_0",
                "s_002um_1_1",
            ],
            "in_cell": [False, True, True, False],
            "in_nucleus": [False, True, False, False],
        }),
        barcode_mappings,
    )
    return star_molecules, soft_expected, barcode_mappings


def run(
    source: Path,
    metrics: Path,
    star_molecules: Path,
    soft_expected: Path,
    barcode_mappings: Path,
    out: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-summary",
            str(source),
            "--expected-source-sha256",
            sha256(source),
            "--source-method-metrics",
            str(metrics),
            "--expected-method-metrics-sha256",
            sha256(metrics),
            "--star-molecules",
            str(star_molecules),
            "--expected-star-molecules-sha256",
            sha256(star_molecules),
            "--soft-expected",
            str(soft_expected),
            "--expected-soft-expected-sha256",
            sha256(soft_expected),
            "--barcode-mappings",
            str(barcode_mappings),
            "--expected-barcode-mappings-sha256",
            sha256(barcode_mappings),
            "--width",
            "2",
            "--height",
            "2",
            "--result-id",
            "hd_crc_flex_100k_frozen_biology_sanity_v1",
            "--generator-commit",
            "a" * 40,
            "--provenance-run-id",
            "20260721_human_crc_flex_100k_publication_reboot_v2",
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )


def test_summarizes_vendor_embedded_100k_biology_deterministically(tmp_path: Path) -> None:
    method_specs = {
        "whole_strict": (0.50, "whole_prediction"),
        "whole_postcollapse_soft": (0.65, "whole_prediction"),
        "whole_postcollapse_hard": (0.70, "whole_prediction"),
        "whole_gated_hard": (0.60, "whole_prediction"),
        "whole_space_ranger": (0.55, "whole_prediction"),
        "matched_postcollapse_hard": (0.70, "exact_member_set"),
        "matched_space_ranger": (0.55, "exact_member_set"),
    }
    methods = {name: method(value, eligibility) for name, (value, eligibility) in method_specs.items()}
    methods["whole_strict"]["total_molecule_mass"] = 1
    methods["whole_postcollapse_soft"]["total_molecule_mass"] = 2
    methods["whole_postcollapse_hard"]["total_molecule_mass"] = 2
    methods["whole_gated_hard"]["total_molecule_mass"] = 2
    methods["matched_space_ranger"]["total_molecule_mass"] = methods["matched_postcollapse_hard"]["total_molecule_mass"]
    source = tmp_path / "source.json"
    metrics = tmp_path / "method_metrics.tsv"
    write_method_metrics(metrics, {
        name: {"eligibility_surface": eligibility}
        for name, (_, eligibility) in method_specs.items()
    })
    source.write_text(json.dumps({
        "schema": "star_spatial.hd.natural_100k_matched_sanity.v1",
        "slide": "H1-GMHFWPH",
        "area": "D1",
        "audit_role": "structural_and_numerical_sanity_gate_not_scientific_winner",
        "image_metrics": methods,
    }, sort_keys=True) + "\n")
    star_molecules, soft_expected, barcode_mappings = write_increment_inputs(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run(source, metrics, star_molecules, soft_expected, barcode_mappings, out_a)
    run(source, metrics, star_molecules, soft_expected, barcode_mappings, out_b)
    assert (out_a / "ambiguous_increment_biology.tsv").read_bytes() == (out_b / "ambiguous_increment_biology.tsv").read_bytes()
    assert (out_a / "star_vs_space_ranger.tsv").read_bytes() == (out_b / "star_vs_space_ranger.tsv").read_bytes()
    assert (out_a / "policy_vs_strict.tsv").read_bytes() == (out_b / "policy_vs_strict.tsv").read_bytes()
    assert (out_a / "RESULTS.md").read_bytes() == (out_b / "RESULTS.md").read_bytes()
    summary = json.loads((out_a / "summary.json").read_text())
    assert summary["artifact_role"] == "sanity_check_only"
    assert summary["star_space_ranger_rows"] == 75
    assert summary["policy_strict_rows"] == 45
    assert not summary["equal_mass_normalization_applied"]
    assert not summary["occupancy_jaccard_reported"]
    assert summary["ambiguous_increment_fields_scored"] == [
        "postcollapse_soft",
        "postcollapse_hard",
        "gated_hard",
    ]
    with (out_a / "ambiguous_increment_biology.tsv").open(newline="") as handle:
        increment_rows = list(csv.DictReader(handle, delimiter="\t"))
    hard_increment = next(row for row in increment_rows if row["field"] == "postcollapse_hard")
    assert float(hard_increment["raw_molecule_mass"]) == 1.0
    assert float(hard_increment["in_cell_mass_fraction"]) == 1.0
    assert float(hard_increment["in_nucleus_mass_fraction"]) == 1.0
    assert float(hard_increment["extracellular_mass_fraction"]) == 0.0
    strict_context = next(row for row in increment_rows if row["field"] == "strict")
    assert float(strict_context["extracellular_mass_fraction"]) == 1.0
    with (out_a / "star_vs_space_ranger.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matched_auc = next(
        row for row in rows
        if row["comparison"] == "exact_member_set_hard_vs_sr" and row["metric"] == "in_cell_roc_auc"
    )
    assert abs(float(matched_auc["difference"]) - 0.15) < 1e-12
    assert matched_auc["eligibility_surface_matched"] == "True"
    whole_auc = next(
        row for row in rows
        if row["comparison"] == "whole_field_hard_vs_sr" and row["metric"] == "in_cell_roc_auc"
    )
    assert whole_auc["eligibility_surface_matched"] == "False"
    assert "vendor-embedded" in (out_a / "LIMITATIONS.md").read_text()
    result_text = (out_a / "RESULTS.md").read_text()
    assert "Biological support for additional predictions" in result_text
    assert "Equal-mass exact-member-set comparison" in result_text
    assert "cannot select or tune a policy" in result_text


def test_rejects_wrong_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-summary",
            str(source),
            "--expected-source-sha256",
            "0" * 64,
            "--source-method-metrics",
            str(source),
            "--expected-method-metrics-sha256",
            sha256(source),
            "--star-molecules",
            str(source),
            "--expected-star-molecules-sha256",
            sha256(source),
            "--soft-expected",
            str(source),
            "--expected-soft-expected-sha256",
            sha256(source),
            "--barcode-mappings",
            str(source),
            "--expected-barcode-mappings-sha256",
            sha256(source),
            "--result-id",
            "hd_crc_flex_100k_frozen_biology_sanity_v1",
            "--generator-commit",
            "a" * 40,
            "--provenance-run-id",
            "20260721_human_crc_flex_100k_publication_reboot_v2",
            "--out-dir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode != 0
    assert "source summary hash mismatch" in completed.stdout
