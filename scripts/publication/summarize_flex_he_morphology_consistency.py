#!/usr/bin/env python3
"""Build a matched two-slide H&E morphology consistency table for Flex."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELD_SUFFIXES = {
    "genomic": "genomic_whole",
    "mapq_off": "mapq_off_whole",
    "positive": "mapq_off_positive_feature_bin_increment",
}
SEGMENTATION_METHOD_KEYS = (
    "model",
    "flow_threshold",
    "cellprob_threshold",
    "min_size",
    "max_size_fraction",
    "nucleus_expansion_um",
    "tile_size",
    "overlap",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty consistency table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("dataset must be LABEL=COMPARISON_DIR")
    return label, Path(path)


def segmentation_signature(summary: dict[str, object]) -> dict[str, object]:
    method = summary["method"]
    contract = summary["input_contract"]
    return {
        **{key: method[key] for key in SEGMENTATION_METHOD_KEYS},
        "downsample": contract["downsample"],
        "engine": method["engine"],
        "engine_version": method["engine_version"],
        "channel": method["channel"],
        "cell_domain": method["cell_domain"],
    }


def summarize_dataset(label: str, root: Path) -> tuple[
    list[dict[str, object]], dict[str, object], dict[str, object],
]:
    fields_path = root / "morphology_fields.tsv"
    reconciliation_path = root / "delta_reconciliation.tsv"
    analysis_summary_path = root / "summary.json"
    for path in (fields_path, reconciliation_path, analysis_summary_path):
        if not path.is_file():
            raise ValueError(f"missing morphology input: {path}")

    field_rows = {row["field"]: row for row in read_tsv(fields_path)}
    reconciliation_rows = {
        row["policy"]: row for row in read_tsv(reconciliation_path)
    }
    analysis_summary = json.loads(analysis_summary_path.read_text(encoding="utf-8"))
    cellpose_summary_path = Path(
        analysis_summary["inputs"]["cellpose_summary"]["path"]
    )
    cellpose_summary = json.loads(
        cellpose_summary_path.read_text(encoding="utf-8")
    )

    rows: list[dict[str, object]] = []
    for policy, reconciliation in reconciliation_rows.items():
        selected = {
            role: field_rows[f"{policy}_{suffix}"]
            for role, suffix in FIELD_SUFFIXES.items()
        }
        genomic = selected["genomic"]
        mapq_off = selected["mapq_off"]
        positive = selected["positive"]

        def number(row: dict[str, str], key: str) -> float:
            return float(row[key])

        row = {
            "dataset": label,
            "policy": policy,
            "genomic_mass": number(reconciliation, "genomic_mass"),
            "mapq_off_mass": number(reconciliation, "mapq_off_mass"),
            "net_increment_mass": number(reconciliation, "net_increment_mass"),
            "percent_net_increment_over_genomic": number(
                reconciliation, "percent_net_increment_over_genomic",
            ),
            "cell_area_fraction": number(genomic, "in_cell_area_fraction"),
            "nucleus_area_fraction": number(
                genomic, "in_nucleus_area_fraction",
            ),
        }
        for compartment, prefix in (("cell", "in_cell"), ("nucleus", "in_nucleus")):
            genomic_fraction = number(genomic, f"{prefix}_mass_fraction")
            mapq_off_fraction = number(mapq_off, f"{prefix}_mass_fraction")
            positive_fraction = number(positive, f"{prefix}_mass_fraction")
            row.update({
                f"genomic_{compartment}_mass_fraction": genomic_fraction,
                f"mapq_off_{compartment}_mass_fraction": mapq_off_fraction,
                f"positive_increment_{compartment}_mass_fraction": positive_fraction,
                f"positive_minus_genomic_{compartment}_fraction_pp": (
                    100.0 * (positive_fraction - genomic_fraction)
                ),
                f"mapq_off_minus_genomic_{compartment}_fraction_pp": (
                    100.0 * (mapq_off_fraction - genomic_fraction)
                ),
                f"genomic_{compartment}_area_normalized_enrichment": number(
                    genomic, f"{prefix}_area_normalized_enrichment",
                ),
                f"positive_increment_{compartment}_area_normalized_enrichment": (
                    number(positive, f"{prefix}_area_normalized_enrichment")
                ),
                f"genomic_{compartment}_roc_auc": number(
                    genomic, f"{prefix}_roc_auc",
                ),
                f"mapq_off_{compartment}_roc_auc": number(
                    mapq_off, f"{prefix}_roc_auc",
                ),
                f"mapq_off_minus_genomic_{compartment}_roc_auc": (
                    number(mapq_off, f"{prefix}_roc_auc")
                    - number(genomic, f"{prefix}_roc_auc")
                ),
                f"positive_increment_{compartment}_roc_auc": number(
                    positive, f"{prefix}_roc_auc",
                ),
                f"genomic_{compartment}_average_precision": number(
                    genomic, f"{prefix}_average_precision",
                ),
                f"mapq_off_minus_genomic_{compartment}_average_precision": (
                    number(mapq_off, f"{prefix}_average_precision")
                    - number(genomic, f"{prefix}_average_precision")
                ),
                f"genomic_{compartment}_fraction_spearman_16um": number(
                    genomic, f"{compartment}_fraction_spearman_16um",
                ),
                f"mapq_off_minus_genomic_{compartment}_fraction_spearman_16um": (
                    number(mapq_off, f"{compartment}_fraction_spearman_16um")
                    - number(genomic, f"{compartment}_fraction_spearman_16um")
                ),
            })
        rows.append(row)

    inputs = {
        "comparison_summary": {
            "path": str(analysis_summary_path.resolve()),
            "sha256": sha256(analysis_summary_path),
        },
        "morphology_fields": {
            "path": str(fields_path.resolve()),
            "sha256": sha256(fields_path),
        },
        "delta_reconciliation": {
            "path": str(reconciliation_path.resolve()),
            "sha256": sha256(reconciliation_path),
        },
        "cellpose_summary": {
            "path": str(cellpose_summary_path.resolve()),
            "sha256": sha256(cellpose_summary_path),
        },
    }
    return rows, inputs, segmentation_signature(cellpose_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="Dataset label and morphology comparison directory as LABEL=DIR.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.dataset) < 2:
        raise SystemExit("at least two --dataset inputs are required")
    if len({label for label, _ in args.dataset}) != len(args.dataset):
        raise SystemExit("dataset labels must be unique")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")

    rows: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    signatures: dict[str, dict[str, object]] = {}
    for label, root in args.dataset:
        dataset_rows, dataset_inputs, signature = summarize_dataset(label, root)
        rows.extend(dataset_rows)
        inputs[label] = dataset_inputs
        signatures[label] = signature
    if len({json.dumps(value, sort_keys=True) for value in signatures.values()}) != 1:
        raise ValueError("Cellpose method signatures differ between datasets")

    args.out_dir.mkdir(parents=True)
    table_path = args.out_dir / "flex_he_morphology_consistency.tsv"
    write_tsv(table_path, rows)
    summary = {
        "schema": "visium_hd_processing.flex_he_morphology_consistency.v1",
        "interpretation": (
            "Matched independent H&E/Cellpose localization of the existing "
            "genomic-to-MAPQ-off contrast. Raw mass is not normalized; small "
            "metric signs do not select a mode."
        ),
        "datasets": [label for label, _ in args.dataset],
        "policies": sorted({str(row["policy"]) for row in rows}),
        "shared_segmentation_method": next(iter(signatures.values())),
        "inputs": inputs,
        "outputs": {
            table_path.name: {
                "bytes": table_path.stat().st_size,
                "sha256": sha256(table_path),
            },
        },
        "invariants": {
            "same_cellpose_method_signature_required": True,
            "image_or_segmentation_not_used_for_assignment": True,
            "positive_increment_compared_with_whole_genomic_field": True,
            "equal_mass_normalization_applied": False,
            "small_metric_sign_selects_policy": False,
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
