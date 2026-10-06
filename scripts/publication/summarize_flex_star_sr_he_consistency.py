#!/usr/bin/env python3
"""Build a matched CRC/SPATCH STAR-versus-Space-Ranger H&E table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_METHODS = {"hard", "soft_expected"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=COMPARISON_DIR")
    return label, Path(raw_path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(row: dict[str, str], name: str) -> float:
    return float(row[name])


def segmentation_signature(summary: dict[str, object]) -> dict[str, object]:
    contract = summary["input_contract"]
    method = summary["method"]
    return {
        "engine": method["engine"],
        "engine_version": method["engine_version"],
        "model": method["model"],
        "channel": method["channel"],
        "downsample": contract["downsample"],
        "nucleus_expansion_um": method["nucleus_expansion_um"],
        "flow_threshold": method["flow_threshold"],
        "cellprob_threshold": method["cellprob_threshold"],
        "min_size": method["min_size"],
        "max_size_fraction": method["max_size_fraction"],
        "tile_size": method["tile_size"],
        "overlap": method["overlap"],
        "cell_domain": method["cell_domain"],
    }


def load_dataset(
    label: str, root: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    summary_path = root / "summary.json"
    fields_path = root / "morphology_fields.tsv"
    reconciliation_path = root / "mass_reconciliation.tsv"
    for path in (summary_path, fields_path, reconciliation_path):
        if not path.is_file():
            raise ValueError(f"missing comparison artifact: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "visium_hd_processing.flex_star_sr_he_morphology.v1":
        raise ValueError(f"unexpected comparison schema: {summary_path}")
    if int(summary["comparison_scale_um"]) != 8:
        raise ValueError("matched synthesis requires 8-um comparison fields")
    fields = {row["field"]: row for row in read_tsv(fields_path)}
    reconciliations = {row["method"]: row for row in read_tsv(reconciliation_path)}
    if set(reconciliations) != EXPECTED_METHODS:
        raise ValueError("comparison must contain hard and soft_expected STAR methods")
    sr = fields.get("space_ranger_common_axis")
    if sr is None:
        raise ValueError("comparison lacks Space Ranger whole field")
    rows: list[dict[str, object]] = []
    for method in sorted(EXPECTED_METHODS):
        star = fields[f"{method}_star_common_axis"]
        star_only = fields[f"{method}_star_positive_spatial_bin_residual"]
        sr_only = fields[f"{method}_space_ranger_positive_spatial_bin_residual"]
        mass = reconciliations[method]
        row: dict[str, object] = {
            "dataset": label,
            "method": method,
            "comparison_scale_um": 8,
            "comparison_axis": "space_ranger_barcode_axis",
            "space_ranger_mass": number(mass, "space_ranger_mass"),
            "star_mass_on_common_axis": number(
                mass, "star_mass_on_space_ranger_axis",
            ),
            "star_percent_excess_over_space_ranger": number(
                mass, "percent_mass_difference_over_space_ranger_on_common_axis",
            ),
            "star_full_capture_mass": number(mass, "star_full_mass"),
            "star_mass_outside_space_ranger_axis": number(
                mass, "star_mass_outside_space_ranger_axis",
            ),
            "star_mass_fraction_outside_space_ranger_axis": (
                number(mass, "star_mass_outside_space_ranger_axis")
                / number(mass, "star_full_mass")
            ),
            "spatial_bin_total_pearson": number(
                mass, "spatial_bin_total_pearson_on_common_axis",
            ),
            "spatial_bin_spearman": number(
                mass, "spatial_bin_spearman_on_common_axis",
            ),
            "gene_total_pearson": number(mass, "gene_total_pearson"),
            "gene_total_spearman_common_detected": number(
                mass, "gene_total_spearman_common_detected",
            ),
            "space_ranger_cell_mass_fraction": number(sr, "in_cell_mass_fraction"),
            "star_cell_mass_fraction": number(star, "in_cell_mass_fraction"),
            "star_minus_space_ranger_cell_fraction_pp": 100.0 * (
                number(star, "in_cell_mass_fraction")
                - number(sr, "in_cell_mass_fraction")
            ),
            "space_ranger_nucleus_mass_fraction": number(
                sr, "in_nucleus_mass_fraction",
            ),
            "star_nucleus_mass_fraction": number(star, "in_nucleus_mass_fraction"),
            "star_minus_space_ranger_nucleus_fraction_pp": 100.0 * (
                number(star, "in_nucleus_mass_fraction")
                - number(sr, "in_nucleus_mass_fraction")
            ),
        }
        for compartment, prefix in (("cell", "in_cell"), ("nucleus", "in_nucleus")):
            row.update({
                f"space_ranger_{compartment}_roc_auc": number(sr, f"{prefix}_roc_auc"),
                f"star_{compartment}_roc_auc": number(star, f"{prefix}_roc_auc"),
                f"star_minus_space_ranger_{compartment}_roc_auc": (
                    number(star, f"{prefix}_roc_auc")
                    - number(sr, f"{prefix}_roc_auc")
                ),
                f"star_only_{compartment}_roc_auc": number(
                    star_only, f"{prefix}_roc_auc",
                ),
                f"space_ranger_only_{compartment}_roc_auc": number(
                    sr_only, f"{prefix}_roc_auc",
                ),
                f"star_only_minus_space_ranger_only_{compartment}_roc_auc": (
                    number(star_only, f"{prefix}_roc_auc")
                    - number(sr_only, f"{prefix}_roc_auc")
                ),
                f"star_minus_space_ranger_{compartment}_fraction_spearman_16um": (
                    number(star, f"{compartment}_fraction_spearman_16um")
                    - number(sr, f"{compartment}_fraction_spearman_16um")
                ),
                f"star_only_minus_space_ranger_only_{compartment}_fraction_spearman_16um": (
                    number(star_only, f"{compartment}_fraction_spearman_16um")
                    - number(sr_only, f"{compartment}_fraction_spearman_16um")
                ),
            })
        row.update({
            "star_positive_spatial_bin_residual_mass": number(
                mass, "star_positive_spatial_bin_residual_mass",
            ),
            "space_ranger_positive_spatial_bin_residual_mass": number(
                mass, "space_ranger_positive_spatial_bin_residual_mass",
            ),
            "star_to_space_ranger_positive_spatial_residual_mass_ratio": number(
                mass, "star_to_space_ranger_positive_spatial_residual_mass_ratio",
            ),
        })
        rows.append(row)
    cellpose_summary_path = Path(summary["inputs"]["cellpose_summary"]["path"])
    cellpose = json.loads(cellpose_summary_path.read_text(encoding="utf-8"))
    inputs = {
        "comparison_summary": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
        "morphology_fields": {"path": str(fields_path.resolve()), "sha256": sha256(fields_path)},
        "mass_reconciliation": {
            "path": str(reconciliation_path.resolve()),
            "sha256": sha256(reconciliation_path),
        },
        "cellpose_summary": {
            "path": str(cellpose_summary_path.resolve()),
            "sha256": sha256(cellpose_summary_path),
        },
    }
    return rows, inputs, segmentation_signature(cellpose)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    labels = [label for label, _ in args.dataset]
    if len(labels) != len(set(labels)):
        raise SystemExit("duplicate dataset label")
    rows: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    signatures: dict[str, dict[str, object]] = {}
    for label, root in args.dataset:
        dataset_rows, dataset_inputs, signature = load_dataset(label, root)
        rows.extend(dataset_rows)
        inputs[label] = dataset_inputs
        signatures[label] = signature
    if set(signatures) != {"CRC", "SPATCH"}:
        raise ValueError("matched synthesis requires CRC and SPATCH")
    if len({json.dumps(value, sort_keys=True) for value in signatures.values()}) != 1:
        raise ValueError("Cellpose method signatures differ between datasets")
    args.out_dir.mkdir(parents=True)
    table_path = args.out_dir / "flex_star_sr_he_consistency.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "visium_hd_processing.flex_star_sr_he_consistency.v1",
        "result_id": "hd_flex_full_star_sr_he_cellpose_two_slide_v1",
        "interpretation": (
            "STAR-versus-Space-Ranger compatibility comparison on each Space Ranger "
            "barcode axis using matched independent H&E/Cellpose structure masks."
        ),
        "inputs": inputs,
        "shared_cellpose_method_signature": next(iter(signatures.values())),
        "outputs": {
            table_path.name: {
                "bytes": table_path.stat().st_size,
                "sha256": sha256(table_path),
            },
        },
        "invariants": {
            "prior_mapq_mode_analysis_preserved_separately": True,
            "primary_comparator_is_space_ranger": True,
            "same_cellpose_method_signature_required": True,
            "raw_mass_not_normalized": True,
            "spatial_bin_residual_not_feature_bin_residual": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        f"{sha256(table_path)}  {table_path.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
