#!/usr/bin/env python3
"""Synthesize count gain and shared-versus-excess control distributions.

Whole-slide biological count totals are reported separately from every control
set.  Each control table compares the shared SR/STAR field with the positive
STAR residual at that control's explicitly named evaluation unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crc-count-summary", type=Path, required=True)
    parser.add_argument("--spatch-count-summary", type=Path, required=True)
    parser.add_argument("--gex-prepared-summary", type=Path, required=True)
    parser.add_argument("--crc-he-fields", type=Path, required=True)
    parser.add_argument("--spatch-he-fields", type=Path, required=True)
    parser.add_argument("--gex-he-fields", type=Path, required=True)
    broad = parser.add_mutually_exclusive_group(required=True)
    broad.add_argument("--xenium-broad-root", type=Path)
    broad.add_argument("--xenium-broad-input", action="append", metavar="SCALE=DIR",
                       help="Explicit evaluated scales; avoids requiring unrequested historical scales.")
    parser.add_argument("--xenium-cancer-summary", type=Path, required=True)
    parser.add_argument("--codex-summary", type=Path, required=True)
    parser.add_argument("--codex-markers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def clean_roundoff(value: float, tolerance: float = 1e-5) -> float:
    return 0.0 if abs(value) <= tolerance else value


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def count_rows(
    crc: dict[str, Any], spatch: dict[str, Any], gex: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset, assay, source in (
        ("CRC", "Flex", crc["methods"]),
        ("SPATCH", "Flex", spatch["methods"]),
    ):
        for method in ("hard", "soft_expected"):
            values = source[method]
            sr = float(values["space_ranger_mass"])
            star_full = float(values["star_mass_all_axes"])
            star_common = float(values["star_mass_on_space_ranger_axis"])
            rows.append({
                "dataset": dataset,
                "assay": assay,
                "method": method,
                "space_ranger_biological_mass": sr,
                "star_full_biological_mass": star_full,
                "net_full_star_minus_space_ranger": star_full - sr,
                "net_full_percent_over_space_ranger": 100.0 * (star_full - sr) / sr,
                "star_mass_on_space_ranger_barcode_axis": star_common,
                "net_common_axis_star_minus_space_ranger": star_common - sr,
                "net_common_axis_percent_over_space_ranger": (
                    100.0 * (star_common - sr) / sr
                ),
                "star_mass_outside_space_ranger_barcode_axis": clean_roundoff(
                    star_full - star_common
                ),
            })
    for method in ("hard", "soft_expected"):
        values = gex["methods"][method]
        sr = float(values["space_ranger_mass"])
        star_full = float(values["star_full_mass"])
        star_common = float(values["star_mass_on_space_ranger_axis"])
        rows.append({
            "dataset": "OVARIAN_GEX",
            "assay": "3prime_GEX",
            "method": method,
            "space_ranger_biological_mass": sr,
            "star_full_biological_mass": star_full,
            "net_full_star_minus_space_ranger": star_full - sr,
            "net_full_percent_over_space_ranger": 100.0 * (star_full - sr) / sr,
            "star_mass_on_space_ranger_barcode_axis": star_common,
            "net_common_axis_star_minus_space_ranger": star_common - sr,
            "net_common_axis_percent_over_space_ranger": (
                100.0 * (star_common - sr) / sr
            ),
            "star_mass_outside_space_ranger_barcode_axis": clean_roundoff(
                star_full - star_common
            ),
        })
    return rows


def he_rows(inputs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, path in inputs:
        fields = read_tsv(path)
        lookup = {(row["policy"], row["field_role"]): row for row in fields}
        for method in ("hard", "soft_expected"):
            for component, role in (
                ("shared", "shared_spatial_bin_mass"),
                ("star_excess", "star_positive_spatial_bin_residual"),
            ):
                row = lookup[(method, role)]
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "component": component,
                    "component_unit": "8um spatial bin",
                    "raw_component_mass_in_he_domain": row["raw_molecule_mass"],
                    "cell_mass_fraction": row["in_cell_mass_fraction"],
                    "nucleus_mass_fraction": row["in_nucleus_mass_fraction"],
                    "cell_area_normalized_enrichment": (
                        row["in_cell_area_normalized_enrichment"]
                    ),
                    "nucleus_area_normalized_enrichment": (
                        row["in_nucleus_area_normalized_enrichment"]
                    ),
                    "cell_roc_auc": row["in_cell_roc_auc"],
                    "nucleus_roc_auc": row["in_nucleus_roc_auc"],
                    "cell_fraction_pearson_16um": row["cell_fraction_pearson_16um"],
                    "nucleus_fraction_pearson_16um": (
                        row["nucleus_fraction_pearson_16um"]
                    ),
                })
    return rows


def broad_inputs(root: Path | None, explicit: list[str] | None) -> dict[int, Path]:
    if explicit:
        result = {}
        for item in explicit:
            scale_text, path = item.split("=", 1)
            scale = int(scale_text)
            if scale <= 0 or scale in result or not path:
                raise ValueError("invalid or duplicate Xenium scale")
            result[scale] = Path(path)
        return result
    if root is None:
        raise ValueError("Xenium broad inputs are required")
    return {scale: root / f"{scale}um_broad" for scale in (64, 128, 256)}


def xenium_broad_rows(root: Path | dict[int, Path]) -> list[dict[str, Any]]:
    rows = []
    paths = root if isinstance(root, dict) else broad_inputs(root, None)
    for scale, directory in sorted(paths.items()):
        source = read_tsv(directory / "matrix_delta_summary.tsv")
        lookup = {row["policy"]: row for row in source}
        for method in ("hard", "soft_expected"):
            row = lookup[method]
            for component, prefix, mass_key in (
                ("shared", "matrix_intersection", "cohort_matrix_intersection_mass"),
                ("star_excess", "positive_delta", "cohort_positive_delta_mass"),
            ):
                rows.append({
                    "dataset": "OVARIAN_GEX",
                    "method": method,
                    "component": component,
                    "patch_scale_um": scale,
                    "component_unit": "shared gene by registered patch",
                    "raw_component_mass_in_xenium_cohort": row[mass_key],
                    "gene_total_log1p_pearson_vs_xenium": (
                        row[f"{prefix}_gene_total_log1p_pearson"]
                    ),
                    "gene_total_spearman_vs_xenium": (
                        row[f"{prefix}_gene_total_spearman"]
                    ),
                    "median_gene_spatial_pearson_vs_xenium": (
                        row[f"{prefix}_median_gene_spatial_pearson"]
                    ),
                    "median_patch_profile_pearson_vs_xenium": (
                        row[f"{prefix}_median_patch_profile_pearson"]
                    ),
                    "macro_cluster_fraction_pearson_vs_xenium": (
                        row[f"{prefix}_macro_cluster_fraction_pearson"]
                    ),
                    "macro_cluster_dominant_roc_auc_vs_xenium": (
                        row[f"{prefix}_macro_cluster_dominant_roc_auc"]
                    ),
                })
    return rows


def xenium_cancer_rows(path: Path) -> list[dict[str, Any]]:
    source = read_tsv(path)
    rows = []
    for scope in (
        "full_reference_panel",
        "exclude_mecom",
        "label_leakage_controlled",
    ):
        for method in ("hard", "soft_expected"):
            for component in ("shared", "star_excess"):
                label = f"{method}_{component}"
                selected = [
                    row for row in source
                    if row["scope"] == scope and row["method"] == label
                ]
                if len(selected) != 1:
                    raise ValueError(f"missing Xenium cancer row: {scope}/{label}")
                row = selected[0]
                rows.append({
                    "dataset": "OVARIAN_GEX",
                    "method": method,
                    "component": component,
                    "scope": scope,
                    "component_unit": "shared gene by registered 128um patch",
                    "raw_cancer_rich_mass": row["raw_cancer_rich_panel_mass"],
                    "raw_cancer_poor_mass": row["raw_cancer_poor_panel_mass"],
                    "cancer_rich_to_poor_mass_per_patch": (
                        row["raw_cancer_rich_to_poor_mass_per_patch"]
                    ),
                    "effect_pearson_vs_xenium": row["effect_pearson_vs_xenium"],
                    "effect_spearman_vs_xenium": row["effect_spearman_vs_xenium"],
                    "effect_direction_agreement_vs_xenium": (
                        row["effect_direction_agreement_vs_xenium"]
                    ),
                    "mean_reference_oriented_gene_auc": (
                        row["mean_reference_oriented_gene_auc"]
                    ),
                    "mean_xenium_positive_gene_auc": (
                        row["mean_xenium_positive_gene_auc"]
                    ),
                    "positive_panel_score_auc": row["positive_panel_score_auc"],
                })
    return rows


def codex_rows(summary_path: Path, marker_path: Path) -> list[dict[str, Any]]:
    summaries = read_tsv(summary_path)
    markers = read_tsv(marker_path)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in markers:
        grouped[(row["method"], int(row["bin_size_um"]))].append(row)
    rows = []
    lookup = {
        (row["method"], int(row["bin_size_um"])): row for row in summaries
    }
    for size in (100, 200, 300, 400, 500):
        for method in ("hard", "soft_expected"):
            for component in ("shared", "star_excess"):
                label = f"{method}_{component}"
                row = lookup[(label, size)]
                metric_rows = grouped[(label, size)]
                rows.append({
                    "dataset": "SPATCH",
                    "method": method,
                    "component": component,
                    "grid_size_um": size,
                    "component_unit": "CODEX marker by evaluation grid",
                    "raw_direct_marker_component_mass": row["direct_panel_rna_mass"],
                    "macro_marker_roc_auc": row["macro_roc_auc"],
                    "median_marker_roc_auc": row["median_roc_auc"],
                    "mean_marker_pearson_raw": finite_mean([
                        float(value["pearson_raw"])
                        for value in metric_rows if value["pearson_raw"] != "NA"
                    ]),
                    "mean_marker_spearman_raw": finite_mean([
                        float(value["spearman_raw"])
                        for value in metric_rows if value["spearman_raw"] != "NA"
                    ]),
                    "mean_rna_detection_fraction": finite_mean([
                        float(value["rna_detection_fraction"])
                        for value in metric_rows
                    ]),
                    "mean_protein_mass_covered_by_rna": finite_mean([
                        float(value["protein_mass_covered_by_rna"])
                        for value in metric_rows
                        if value["protein_mass_covered_by_rna"] != "NA"
                    ]),
                })
    return rows


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    xenium_paths = broad_inputs(args.xenium_broad_root, args.xenium_broad_input)
    input_paths = [
        args.crc_count_summary,
        args.spatch_count_summary,
        args.gex_prepared_summary,
        args.crc_he_fields,
        args.spatch_he_fields,
        args.gex_he_fields,
        args.xenium_cancer_summary,
        args.codex_summary,
        args.codex_markers,
        *(directory / "matrix_delta_summary.tsv" for directory in xenium_paths.values()),
    ]
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    crc = json.loads(args.crc_count_summary.read_text(encoding="utf-8"))
    spatch = json.loads(args.spatch_count_summary.read_text(encoding="utf-8"))
    gex = json.loads(args.gex_prepared_summary.read_text(encoding="utf-8"))
    outputs = {
        "count_gain.tsv": count_rows(crc, spatch, gex),
        "he_shared_excess.tsv": he_rows([
            ("CRC", args.crc_he_fields),
            ("SPATCH", args.spatch_he_fields),
            ("OVARIAN_GEX", args.gex_he_fields),
        ]),
        "xenium_broad_shared_excess.tsv": xenium_broad_rows(
            xenium_paths
        ),
        "xenium_cancer_shared_excess.tsv": xenium_cancer_rows(
            args.xenium_cancer_summary
        ),
        "codex_shared_excess.tsv": codex_rows(
            args.codex_summary, args.codex_markers
        ),
    }
    args.out_dir.mkdir(parents=True)
    for name, rows in outputs.items():
        write_rows(args.out_dir / name, rows)
    summary = {
        "schema": "visium_hd_processing.three_slide_shared_excess_controls.v1",
        "count_definition": (
            "Whole-slide biological counts; net=STAR-SR and percent=100*net/SR. "
            "Full and matched Space Ranger barcode-axis values are separate."
        ),
        "control_definition": (
            "Within each control's declared evaluation unit, shared=min(STAR,SR) "
            "and STAR excess=max(STAR-SR,0). Control coverage is not count mass."
        ),
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_paths
        },
        "outputs": {},
        "invariants": {
            "whole_slide_counts_separate_from_control_domains": True,
            "hard_and_soft_reported_separately": True,
            "raw_mass_not_normalized": True,
            "net_percentage_uses_space_ranger_once_as_denominator": True,
            "controls_not_used_for_assignment": True,
        },
    }
    for name in outputs:
        path = args.out_dir / name
        summary["outputs"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [
                *(args.out_dir / name for name in outputs),
                summary_path,
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
