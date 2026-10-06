#!/usr/bin/env python3
"""Summarize the frozen vendor-embedded Flex 100K biology sanity check."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from star_spatial.hd_natural_image_oracle import binary_density_metrics, distribution_distance


EXPECTED_SCHEMA = "star_spatial.hd.natural_100k_matched_sanity.v1"
EXPECTED_SLIDE = "H1-GMHFWPH"
EXPECTED_AREA = "D1"
UNIT_PATTERN = re.compile(r"^s_002um_(\d+)_(\d+)(?:-1)?$")

STAR_SR_COMPARISONS = (
    ("exact_member_set_hard_vs_sr", "exact_member_set", "matched_postcollapse_hard", "matched_space_ranger"),
    ("whole_field_strict_vs_sr", "whole_field", "whole_strict", "whole_space_ranger"),
    ("whole_field_soft_vs_sr", "whole_field", "whole_postcollapse_soft", "whole_space_ranger"),
    ("whole_field_hard_vs_sr", "whole_field", "whole_postcollapse_hard", "whole_space_ranger"),
    ("whole_field_gated_hard_vs_sr", "whole_field", "whole_gated_hard", "whole_space_ranger"),
)

POLICY_STRICT_COMPARISONS = (
    ("soft_vs_strict", "whole_postcollapse_soft", "whole_strict"),
    ("hard_vs_strict", "whole_postcollapse_hard", "whole_strict"),
    ("gated_hard_vs_strict", "whole_gated_hard", "whole_strict"),
)

METRICS = (
    ("raw_molecule_mass", ("total_molecule_mass",), "report_only"),
    ("occupied_2um_bins", ("occupied_bins",), "report_only"),
    ("in_cell_roc_auc", ("classification", "in_cell", "roc_auc"), "higher"),
    ("in_cell_average_precision", ("classification", "in_cell", "average_precision"), "higher"),
    ("in_nucleus_roc_auc", ("classification", "in_nucleus", "roc_auc"), "higher"),
    ("in_nucleus_average_precision", ("classification", "in_nucleus", "average_precision"), "higher"),
    ("extracellular_mass_fraction", ("compartment_support", "extracellular", "mass_fraction"), "lower"),
    ("in_cell_normalized_tv", ("distribution_fit", "in_cell", "normalized_total_variation"), "lower_secondary"),
    ("in_nucleus_normalized_tv", ("distribution_fit", "in_nucleus", "normalized_total_variation"), "lower_secondary"),
    ("cell_fraction_pearson", ("parent_image_support", "all_parents", "cell_fraction_pearson"), "higher"),
    ("nucleus_fraction_pearson", ("parent_image_support", "all_parents", "nucleus_fraction_pearson"), "higher"),
    ("cell_supported_mass_fraction", ("parent_image_support", "cell_supported_fine_scale", "global_supported_mass_fraction"), "higher"),
    ("nucleus_supported_mass_fraction", ("parent_image_support", "nucleus_supported_fine_scale", "global_supported_mass_fraction"), "higher"),
    ("cell_image_empty_unique_max_fraction", ("parent_image_support", "cell_supported_fine_scale", "parents_with_image_empty_unique_max_fraction"), "lower"),
    ("nucleus_image_empty_unique_max_fraction", ("parent_image_support", "nucleus_supported_fine_scale", "parents_with_image_empty_unique_max_fraction"), "lower"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-method-metrics", type=Path, required=True)
    parser.add_argument("--expected-method-metrics-sha256", required=True)
    parser.add_argument("--star-molecules", type=Path, required=True)
    parser.add_argument("--expected-star-molecules-sha256", required=True)
    parser.add_argument("--soft-expected", type=Path, required=True)
    parser.add_argument("--expected-soft-expected-sha256", required=True)
    parser.add_argument("--barcode-mappings", type=Path, required=True)
    parser.add_argument("--expected-barcode-mappings-sha256", required=True)
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--result-id", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--provenance-run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nested_value(document: dict, path: tuple[str, ...]) -> float:
    value = document
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric at {'.'.join(path)}")
    return result


def preferred_difference(difference: float, direction: str) -> float | str:
    if direction == "higher":
        return difference
    if direction in {"lower", "lower_secondary"}:
        return -difference
    return ""


def comparison_rows(
    methods: dict,
    eligibility_surfaces: dict[str, str],
    specifications: tuple[tuple[str, str, str, str], ...],
) -> list[dict]:
    rows = []
    for comparison, cohort, star_name, sr_name in specifications:
        star = methods[star_name]
        space_ranger = methods[sr_name]
        declared_surface_equal = eligibility_surfaces[star_name] == eligibility_surfaces[sr_name]
        matched_surface = cohort == "exact_member_set" and declared_surface_equal
        if cohort == "exact_member_set" and not declared_surface_equal:
            raise ValueError("exact-member-set STAR/SR eligibility surfaces differ")
        for metric, path, direction in METRICS:
            star_value = nested_value(star, path)
            sr_value = nested_value(space_ranger, path)
            difference = star_value - sr_value
            rows.append({
                "comparison": comparison,
                "cohort": cohort,
                "star_method": star_name,
                "space_ranger_method": sr_name,
                "eligibility_surface_matched": matched_surface,
                "metric": metric,
                "preferred_direction": direction,
                "star_value": star_value,
                "space_ranger_value": sr_value,
                "difference": difference,
                "preferred_signed_difference": preferred_difference(difference, direction),
                "equal_mass_normalization_applied": False,
            })
    return rows


def policy_rows(methods: dict) -> list[dict]:
    rows = []
    for comparison, policy_name, strict_name in POLICY_STRICT_COMPARISONS:
        policy = methods[policy_name]
        strict = methods[strict_name]
        for metric, path, direction in METRICS:
            policy_value = nested_value(policy, path)
            strict_value = nested_value(strict, path)
            difference = policy_value - strict_value
            rows.append({
                "comparison": comparison,
                "policy_method": policy_name,
                "strict_method": strict_name,
                "metric": metric,
                "preferred_direction": direction,
                "policy_value": policy_value,
                "strict_value": strict_value,
                "difference": difference,
                "preferred_signed_difference": preferred_difference(difference, direction),
                "interpretation": "ambiguous_increment_over_strict" if metric == "raw_molecule_mass" else "whole_field_metric_difference",
                "equal_mass_normalization_applied": False,
            })
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unit_index(unit: str, width: int, height: int) -> int:
    match = UNIT_PATTERN.fullmatch(unit)
    if match is None:
        raise ValueError(f"malformed 2 um unit: {unit}")
    row, column = int(match.group(1)), int(match.group(2))
    if row >= height or column >= width:
        raise ValueError(f"2 um unit outside declared grid: {unit}")
    return row * width + column


def load_star_fields(path: Path, width: int, height: int) -> dict[str, np.ndarray]:
    names = ("strict", "postcollapse_hard", "gated_hard")
    fields = {name: np.zeros(width * height, dtype=np.float64) for name in names}
    seen = {name: 0 for name in names}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"product", "unit_2um"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("STAR molecule table lacks product or unit_2um")
        for row in reader:
            product = row["product"]
            if product not in fields:
                continue
            fields[product][unit_index(row["unit_2um"], width, height)] += 1.0
            seen[product] += 1
    missing = [name for name, count in seen.items() if count == 0]
    if missing:
        raise ValueError(f"STAR molecule table lacks required products: {missing}")
    return fields


def load_soft_field(path: Path, width: int, height: int) -> np.ndarray:
    field = np.zeros(width * height, dtype=np.float64)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"unit_2um", "expected_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("soft expected-count table lacks unit_2um or expected_count")
        for row in reader:
            value = float(row["expected_count"])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid soft expected count: {value}")
            field[unit_index(row["unit_2um"], width, height)] += value
    if not np.any(field):
        raise ValueError("soft expected-count field is empty")
    return field


def load_compartments(path: Path, width: int, height: int) -> np.ndarray:
    states = np.zeros(width * height, dtype=np.uint8)
    offset = 0
    parquet = pq.ParquetFile(path)
    required = {"square_002um", "in_cell", "in_nucleus"}
    if not required.issubset(parquet.schema.names):
        raise ValueError("barcode mappings lack image compartment columns")
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=100_000):
        data = batch.to_pydict()
        for unit, in_cell, in_nucleus in zip(
            data["square_002um"], data["in_cell"], data["in_nucleus"], strict=True
        ):
            index = unit_index(unit, width, height)
            if index != offset:
                raise ValueError("barcode mappings are not a complete row-major grid")
            states[index] = 2 if in_nucleus else (1 if in_cell else 0)
            offset += 1
    if offset != width * height:
        raise ValueError(f"incomplete barcode mapping grid: {offset} != {width * height}")
    return states


def score_increment(name: str, role: str, field: np.ndarray, states: np.ndarray) -> dict:
    total = float(np.sum(field, dtype=np.float64))
    in_cell = states > 0
    in_nucleus = states == 2
    result = {
        "field": name,
        "field_role": role,
        "raw_molecule_mass": total,
        "occupied_2um_bins": int(np.count_nonzero(field)),
    }
    for label, mask in (
        ("extracellular", states == 0),
        ("in_cell", in_cell),
        ("in_nucleus", in_nucleus),
    ):
        mass = float(np.sum(field[mask], dtype=np.float64))
        area_fraction = float(np.mean(mask))
        fraction = mass / total if total else 0.0
        result[f"{label}_mass"] = mass
        result[f"{label}_mass_fraction"] = fraction
        result[f"{label}_area_fraction"] = area_fraction
        result[f"{label}_area_normalized_enrichment"] = fraction / area_fraction if area_fraction else 0.0
    cell_classification = binary_density_metrics(field, in_cell)
    nucleus_classification = binary_density_metrics(field, in_nucleus)
    result.update({
        "in_cell_roc_auc": cell_classification["roc_auc"],
        "in_cell_average_precision": cell_classification["average_precision"],
        "in_nucleus_roc_auc": nucleus_classification["roc_auc"],
        "in_nucleus_average_precision": nucleus_classification["average_precision"],
        "in_cell_normalized_tv_secondary": distribution_distance(
            field, in_cell.astype(np.float64)
        )["normalized_total_variation"],
        "in_nucleus_normalized_tv_secondary": distribution_distance(
            field, in_nucleus.astype(np.float64)
        )["normalized_total_variation"],
    })
    return result


def ambiguous_increment_rows(
    methods: dict,
    star_molecules: Path,
    soft_expected: Path,
    barcode_mappings: Path,
    width: int,
    height: int,
) -> list[dict]:
    fields = load_star_fields(star_molecules, width, height)
    fields["postcollapse_soft"] = load_soft_field(soft_expected, width, height)
    states = load_compartments(barcode_mappings, width, height)
    strict = fields["strict"]
    expected_masses = {
        "strict": nested_value(methods["whole_strict"], ("total_molecule_mass",)),
        "postcollapse_soft": nested_value(methods["whole_postcollapse_soft"], ("total_molecule_mass",)),
        "postcollapse_hard": nested_value(methods["whole_postcollapse_hard"], ("total_molecule_mass",)),
        "gated_hard": nested_value(methods["whole_gated_hard"], ("total_molecule_mass",)),
    }
    for name, field in fields.items():
        observed = float(np.sum(field, dtype=np.float64))
        if not math.isclose(observed, expected_masses[name], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{name} field mass differs from frozen summary: {observed} != {expected_masses[name]}")
    rows = [score_increment("strict", "strict_context", strict, states)]
    for name in ("postcollapse_soft", "postcollapse_hard", "gated_hard"):
        increment = fields[name] - strict
        minimum = float(np.min(increment))
        if minimum < -1e-12:
            raise ValueError(f"strict is not a field-level subset of {name}: minimum delta {minimum}")
        increment[increment < 0.0] = 0.0
        rows.append(score_increment(name, "ambiguous_increment_over_strict", increment, states))
    return rows


def write_results(path: Path, methods: dict, increment_rows: list[dict]) -> None:
    ordered = (
        ("Strict", "whole_strict"),
        ("Soft", "whole_postcollapse_soft"),
        ("Hard", "whole_postcollapse_hard"),
        ("Gated hard", "whole_gated_hard"),
        ("Space Ranger", "whole_space_ranger"),
    )
    strict_mass = nested_value(methods["whole_strict"], ("total_molecule_mass",))
    lines = [
        "# H1-GMHFWPH/D1 100K frozen biology sanity check",
        "",
        "> **Rider:** vendor-embedded, downsampled sanity check only. These values",
        "> cannot select or tune a policy and are not publication-primary biology.",
        "",
        "## Biological support for additional predictions",
        "",
        "The ambiguous increment is scored directly after subtracting the strict",
        "2 um field. Compartment enrichment is primary; AUC/AP and normalized TV",
        "are secondary because the 100K field is extremely sparse.",
        "",
        "| Field | Role | Raw mass | In-cell fraction | In-cell enrichment | Nucleus fraction | Nucleus enrichment | Extracellular fraction | Extracellular enrichment | In-cell AUC | Nucleus AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in increment_rows:
        lines.append(
            f"| {row['field']} | {row['field_role']} | {row['raw_molecule_mass']:.12g} | "
            f"{row['in_cell_mass_fraction']:.12g} | {row['in_cell_area_normalized_enrichment']:.12g} | "
            f"{row['in_nucleus_mass_fraction']:.12g} | {row['in_nucleus_area_normalized_enrichment']:.12g} | "
            f"{row['extracellular_mass_fraction']:.12g} | {row['extracellular_area_normalized_enrichment']:.12g} | "
            f"{row['in_cell_roc_auc']:.12g} | {row['in_nucleus_roc_auc']:.12g} |"
        )
    lines.extend((
        "",
        "## Whole-field frozen-image metrics",
        "",
        "Raw molecule mass is not normalized. Normalized TV is secondary and is",
        "available in the complete TSV rather than promoted in this compact table.",
        "",
        "| Method | Raw mass | Delta vs strict | In-cell ROC AUC | In-nucleus ROC AUC | Cell-fraction Pearson | Nucleus-fraction Pearson | Extracellular mass fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for label, name in ordered:
        value = methods[name]
        mass = nested_value(value, ("total_molecule_mass",))
        lines.append(
            f"| {label} | {mass:.12g} | {mass - strict_mass:+.12g} | "
            f"{nested_value(value, ('classification', 'in_cell', 'roc_auc')):.12g} | "
            f"{nested_value(value, ('classification', 'in_nucleus', 'roc_auc')):.12g} | "
            f"{nested_value(value, ('parent_image_support', 'all_parents', 'cell_fraction_pearson')):.12g} | "
            f"{nested_value(value, ('parent_image_support', 'all_parents', 'nucleus_fraction_pearson')):.12g} | "
            f"{nested_value(value, ('compartment_support', 'extracellular', 'mass_fraction')):.12g} |"
        )
    star = methods["matched_postcollapse_hard"]
    sr = methods["matched_space_ranger"]
    matched_mass = nested_value(star, ("total_molecule_mass",))
    if matched_mass != nested_value(sr, ("total_molecule_mass",)):
        raise ValueError("exact-member-set STAR/SR raw masses differ")
    lines.extend((
        "",
        "## Equal-mass exact-member-set comparison",
        "",
        f"Both STAR and Space Ranger contain {matched_mass:.12g} molecules. STAR-minus-SR deltas:",
        "",
        f"- in-cell ROC AUC: {nested_value(star, ('classification', 'in_cell', 'roc_auc')) - nested_value(sr, ('classification', 'in_cell', 'roc_auc')):+.12g}",
        f"- in-nucleus ROC AUC: {nested_value(star, ('classification', 'in_nucleus', 'roc_auc')) - nested_value(sr, ('classification', 'in_nucleus', 'roc_auc')):+.12g}",
        f"- extracellular mass fraction: {nested_value(star, ('compartment_support', 'extracellular', 'mass_fraction')) - nested_value(sr, ('compartment_support', 'extracellular', 'mass_fraction')):+.12g}",
        f"- cell-fraction Pearson: {nested_value(star, ('parent_image_support', 'all_parents', 'cell_fraction_pearson')) - nested_value(sr, ('parent_image_support', 'all_parents', 'cell_fraction_pearson')):+.12g}",
        f"- nucleus-fraction Pearson: {nested_value(star, ('parent_image_support', 'all_parents', 'nucleus_fraction_pearson')) - nested_value(sr, ('parent_image_support', 'all_parents', 'nucleus_fraction_pearson')):+.12g}",
        "",
        "The matched result is a coordinate-assignment sanity check, not evidence",
        "that Space Ranger is biological truth. Whole-field differences combine",
        "assignment behavior with unequal recovered molecule mass.",
        "",
    ))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.result_id != "hd_crc_flex_100k_frozen_biology_sanity_v1":
        raise ValueError(f"unexpected result ID: {args.result_id}")
    if re.fullmatch(r"[0-9a-f]{40}", args.generator_commit) is None:
        raise ValueError("generator commit must be a 40-character lowercase Git SHA")
    if args.provenance_run_id != "20260721_human_crc_flex_100k_publication_reboot_v2":
        raise ValueError(f"unexpected provenance run ID: {args.provenance_run_id}")
    observed_sha256 = sha256(args.source_summary)
    if observed_sha256 != args.expected_source_sha256:
        raise ValueError(
            f"source summary hash mismatch: {observed_sha256} != {args.expected_source_sha256}"
        )
    observed_metrics_sha256 = sha256(args.source_method_metrics)
    if observed_metrics_sha256 != args.expected_method_metrics_sha256:
        raise ValueError(
            "source method-metrics hash mismatch: "
            f"{observed_metrics_sha256} != {args.expected_method_metrics_sha256}"
        )
    extra_input_hashes = (
        (args.star_molecules, args.expected_star_molecules_sha256, "STAR molecule table"),
        (args.soft_expected, args.expected_soft_expected_sha256, "soft expected-count table"),
        (args.barcode_mappings, args.expected_barcode_mappings_sha256, "barcode mappings"),
    )
    observed_extra_hashes = {}
    for path, expected, label in extra_input_hashes:
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"{label} hash mismatch: {observed} != {expected}")
        observed_extra_hashes[str(path.resolve())] = observed
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if source.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unsupported source schema: {source.get('schema')!r}")
    if source.get("slide") != EXPECTED_SLIDE or source.get("area") != EXPECTED_AREA:
        raise ValueError("source is not the frozen H1-GMHFWPH/D1 100K artifact")
    if source.get("audit_role") != "structural_and_numerical_sanity_gate_not_scientific_winner":
        raise ValueError("source does not declare the required sanity-only audit role")
    methods = source.get("image_metrics", {})
    required = {
        method
        for _, _, star, sr in STAR_SR_COMPARISONS
        for method in (star, sr)
    } | {
        method
        for _, policy, strict in POLICY_STRICT_COMPARISONS
        for method in (policy, strict)
    }
    missing = sorted(required - methods.keys())
    if missing:
        raise ValueError(f"source is missing required biological fields: {missing}")
    with args.source_method_metrics.open("rt", encoding="utf-8", newline="") as handle:
        eligibility_surfaces = {
            row["method"]: row["eligibility_surface"]
            for row in csv.DictReader(handle, delimiter="\t")
        }
    missing_surfaces = sorted(required - eligibility_surfaces.keys())
    if missing_surfaces:
        raise ValueError(f"method metrics lack required eligibility surfaces: {missing_surfaces}")
    star_sr = comparison_rows(methods, eligibility_surfaces, STAR_SR_COMPARISONS)
    policies = policy_rows(methods)
    increments = ambiguous_increment_rows(
        methods,
        args.star_molecules,
        args.soft_expected,
        args.barcode_mappings,
        args.width,
        args.height,
    )
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    star_sr_path = args.out_dir / "star_vs_space_ranger.tsv"
    policy_path = args.out_dir / "policy_vs_strict.tsv"
    limitations_path = args.out_dir / "LIMITATIONS.md"
    results_path = args.out_dir / "RESULTS.md"
    increment_path = args.out_dir / "ambiguous_increment_biology.tsv"
    write_tsv(star_sr_path, star_sr)
    write_tsv(policy_path, policies)
    write_tsv(increment_path, increments)
    write_results(results_path, methods, increments)
    limitations_path.write_text(
        """# H1-GMHFWPH/D1 100K biological sanity-check rider

This analysis is permitted only as a **vendor-embedded, downsampled sanity
check**. It is not publication-primary biological evidence, cannot select or
tune a policy, and must be replaced by the full raw-FASTQ open primary.

The frozen field ancestry used Space Ranger-derived read eligibility, feature,
UMI, and barcode information. Space Ranger is a comparator, not biological
truth. Raw molecule mass is preserved; policies are not normalized to equal
mass. Normalized total variation is secondary, and occupancy Jaccard is not
reported.
""",
        encoding="utf-8",
    )
    output_paths = (increment_path, star_sr_path, policy_path, results_path, limitations_path)
    summary = {
        "schema": "star_spatial.hd.flex_100k_biology_sanity.v1",
        "status": "pass",
        "result_id": args.result_id,
        "generator": "scripts/publication/summarize_hd_flex_100k_biology_sanity.py",
        "generator_commit": args.generator_commit,
        "provenance_run_id": args.provenance_run_id,
        "slide": EXPECTED_SLIDE,
        "area": EXPECTED_AREA,
        "artifact_role": "sanity_check_only",
        "publication_primary_eligible": False,
        "trusted_for_biological_claims": False,
        "vendor_information_embedded": True,
        "downsampled_100k": True,
        "parameters_reestimated": False,
        "policy_selection_permitted": False,
        "equal_mass_normalization_applied": False,
        "occupancy_jaccard_reported": False,
        "normalized_tv_role": "secondary_shape_metric",
        "source_summary": str(args.source_summary.resolve()),
        "source_summary_sha256": observed_sha256,
        "source_method_metrics": str(args.source_method_metrics.resolve()),
        "source_method_metrics_sha256": observed_metrics_sha256,
        "increment_field_inputs": observed_extra_hashes,
        "ambiguous_increment_fields_scored": [
            row["field"] for row in increments if row["field_role"] == "ambiguous_increment_over_strict"
        ],
        "star_space_ranger_rows": len(star_sr),
        "policy_strict_rows": len(policies),
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in output_paths
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
