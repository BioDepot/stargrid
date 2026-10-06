import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/publication/compare_spatch_mapq_codex_results.py"


def write_summary(path: Path, prefix: str, offset: float) -> None:
    fields = [
        "method", "bin_size_um", "markers_scored", "macro_roc_auc",
        "median_roc_auc", "direct_panel_rna_mass", "method_evaluation_mass",
    ]
    rows = [{
        "method": "space_ranger_2020a",
        "bin_size_um": 100,
        "markers_scored": 1,
        "macro_roc_auc": 0.6,
        "median_roc_auc": 0.6,
        "direct_panel_rna_mass": 10,
        "method_evaluation_mass": 100,
    }]
    for policy in ("strict", "soft_expected", "hard", "gated_hard"):
        rows.append({
            "method": f"{prefix}{policy}",
            "bin_size_um": 100,
            "markers_scored": 1,
            "macro_roc_auc": 0.61 + offset,
            "median_roc_auc": 0.61 + offset,
            "direct_panel_rna_mass": 11,
            "method_evaluation_mass": 110 + offset,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markers(path: Path, prefix: str, offset: float) -> None:
    fields = [
        "method", "bin_size_um", "protein", "reference_marker_available",
        "roc_auc_top_protein_quantile",
    ]
    rows = [{
        "method": "space_ranger_2020a",
        "bin_size_um": 100,
        "protein": "CD8",
        "reference_marker_available": True,
        "roc_auc_top_protein_quantile": 0.6,
    }]
    for policy in ("strict", "soft_expected", "hard", "gated_hard"):
        rows.append({
            "method": f"{prefix}{policy}",
            "bin_size_um": 100,
            "protein": "CD8",
            "reference_marker_available": True,
            "roc_auc_top_protein_quantile": 0.61 + offset,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_compare_spatch_mapq_codex_results(tmp_path):
    genomic_summary = tmp_path / "genomic_summary.tsv"
    genomic_markers = tmp_path / "genomic_markers.tsv"
    off_summary = tmp_path / "off_summary.tsv"
    off_markers = tmp_path / "off_markers.tsv"
    write_summary(genomic_summary, "star_2020a_", 0.0)
    write_markers(genomic_markers, "star_2020a_", 0.0)
    write_summary(off_summary, "star_2020a_mapq_off_", 0.01)
    write_markers(off_markers, "star_2020a_mapq_off_", 0.01)
    out = tmp_path / "out"
    subprocess.run([
        sys.executable,
        str(SCRIPT),
        "--genomic-method-summary",
        str(genomic_summary),
        "--genomic-marker-metrics",
        str(genomic_markers),
        "--mapq-off-method-summary",
        str(off_summary),
        "--mapq-off-marker-metrics",
        str(off_markers),
        "--out-dir",
        str(out),
    ], check=True)
    with (out / "macro_auc_comparison.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    hard = next(row for row in rows if row["policy"] == "hard")
    assert abs(float(hard["mapq_off_minus_genomic_macro_roc_auc"]) - 0.01) < 1e-12
