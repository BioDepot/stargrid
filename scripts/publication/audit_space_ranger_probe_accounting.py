#!/usr/bin/env python3
"""Audit accepted versus diagnostic Space Ranger probe accounting.

10x excludes ``included=false`` and deprecated probes from expression outputs
by default while retaining their counts in the raw probe-level artifact for
diagnostics, controls, and provenance.  H5 ``target_sets`` membership
identifies the accepted filtered probe reference.  The unfortunately named
raw-probe H5 ``filtered_probes`` field is a separate passed-gDNA-QC annotation;
it must never be substituted for the 10x ``included``/``target_sets`` default
off-target exclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_gene_id(value: str) -> str:
    return value.split(".", 1)[0]


def load_filtered_probe_panel(path: Path) -> dict[str, object]:
    """Return the gene set retained by Space Ranger probe filtering."""
    metadata: dict[str, str] = {}
    table_lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                key, separator, value = line[1:].strip().partition("=")
                if separator:
                    metadata[key] = value
            elif line.strip():
                table_lines.append(line)
    reader = csv.DictReader(table_lines)
    required = {"gene_id", "probe_id", "included"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"probe reference lacks columns: {sorted(required)}")

    rows_by_gene: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    input_rows = included_true_rows = included_false_rows = deprecated_rows = 0
    for row in reader:
        input_rows += 1
        gene = normalize_gene_id(row["gene_id"])
        raw_included = row["included"].strip().upper()
        if raw_included not in {"TRUE", "FALSE"}:
            raise ValueError(f"invalid probe included value: {row['included']!r}")
        included = raw_included == "TRUE"
        deprecated = (
            gene.startswith("DEPRECATED_")
            or row["probe_id"].startswith("DEPRECATED")
        )
        included_true_rows += int(included)
        included_false_rows += int(not included)
        deprecated_rows += int(deprecated)
        rows_by_gene[gene].append((included, deprecated))

    eligible_genes = {
        gene
        for gene, rows in rows_by_gene.items()
        if rows
        and any(included for included, _ in rows)
        and not any((not included) or deprecated for included, deprecated in rows)
    }
    eligible_probe_rows = sum(
        included and not deprecated and gene in eligible_genes
        for gene, rows in rows_by_gene.items()
        for included, deprecated in rows
    )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "metadata": metadata,
        "input_probe_rows": input_rows,
        "included_true_rows": included_true_rows,
        "included_false_rows": included_false_rows,
        "deprecated_rows": deprecated_rows,
        "input_gene_count": len(rows_by_gene),
        "eligible_probe_rows": eligible_probe_rows,
        "eligible_gene_count": len(eligible_genes),
        "excluded_gene_count": len(rows_by_gene) - len(eligible_genes),
        "eligible_genes": sorted(eligible_genes),
    }


def audit_h5ad(path: Path, eligible_genes: set[str]) -> dict[str, object]:
    """Audit the retained H5AD feature axis and raw mass without loading X."""
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        encoding = x.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding != "csr_matrix":
            raise ValueError(f"expected CSR H5AD X, observed {encoding!r}")
        gene_ids = [
            normalize_gene_id(value.decode() if isinstance(value, bytes) else str(value))
            for value in handle["var/gene_ids"][:]
        ]
        if len(gene_ids) != len(set(gene_ids)):
            raise ValueError("duplicate H5AD gene IDs")
        shape = tuple(int(value) for value in x.attrs["shape"])
        if shape[1] != len(gene_ids):
            raise ValueError("H5AD X and gene axis disagree")
        mass = 0.0
        data = x["data"]
        nnz = len(data)
        for start in range(0, len(data), 5_000_000):
            values = np.asarray(data[start : start + 5_000_000], dtype=np.float64)
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("invalid H5AD counts")
            mass += float(values.sum(dtype=np.float64))
    observed = set(gene_ids)
    missing = sorted(eligible_genes - observed)
    unexpected = sorted(observed - eligible_genes)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "observations": shape[0],
        "features": shape[1],
        "nnz": nnz,
        "raw_mass": mass,
        "axis_exact": not missing and not unexpected,
        "filtered_panel_genes_missing": missing,
        "diagnostic_non_panel_genes_present": unexpected,
    }


def audit_raw_probe_h5(path: Path) -> dict[str, object]:
    """Partition raw probe counts by 10x inclusion and orthogonal gDNA QC."""
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        features = matrix["features"]
        target_sets = features["target_sets"]
        if len(target_sets) != 1:
            raise ValueError("expected exactly one raw-probe target set")
        target_name = next(iter(target_sets))
        probe_count = len(features["id"])
        in_filtered_reference = np.zeros(probe_count, dtype=bool)
        in_filtered_reference[target_sets[target_name][:]] = True
        passed_gdna = np.asarray(features["filtered_probes"][:], dtype=bool)
        if len(passed_gdna) != probe_count:
            raise ValueError("raw-probe filtered_probes axis mismatch")

        categories = {
            "filtered_probe_reference": in_filtered_reference,
            "diagnostic_outside_filtered_probe_reference": ~in_filtered_reference,
            "passed_gdna_filter": passed_gdna,
            "failed_gdna_filter": ~passed_gdna,
        }
        joint_categories = {
            "accepted_reference_passed_gdna_qc": (
                in_filtered_reference & passed_gdna
            ),
            "accepted_reference_failed_gdna_qc": (
                in_filtered_reference & ~passed_gdna
            ),
            "excluded_diagnostic_passed_gdna_qc": (
                ~in_filtered_reference & passed_gdna
            ),
            "excluded_diagnostic_failed_gdna_qc": (
                ~in_filtered_reference & ~passed_gdna
            ),
        }
        masses = {key: 0.0 for key in categories}
        detected = {key: np.zeros(probe_count, dtype=bool) for key in categories}
        joint_masses = {key: 0.0 for key in joint_categories}
        joint_detected = {
            key: np.zeros(probe_count, dtype=bool) for key in joint_categories
        }
        indices, values = matrix["indices"], matrix["data"]
        for start in range(0, len(values), 5_000_000):
            stop = min(len(values), start + 5_000_000)
            index = np.asarray(indices[start:stop], dtype=np.int64)
            data = np.asarray(values[start:stop], dtype=np.float64)
            if np.any(~np.isfinite(data)) or np.any(data < 0):
                raise ValueError("invalid raw-probe counts")
            for key, probe_mask in categories.items():
                mask = probe_mask[index]
                masses[key] += float(data[mask].sum(dtype=np.float64))
                detected[key][index[mask]] = True
            for key, probe_mask in joint_categories.items():
                entry_mask = probe_mask[index]
                joint_masses[key] += float(data[entry_mask].sum(dtype=np.float64))
                joint_detected[key][index[entry_mask]] = True

    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "target_set_name": target_name,
        "probe_features": probe_count,
        "total_raw_mass": masses["filtered_probe_reference"]
        + masses["diagnostic_outside_filtered_probe_reference"],
        "categories": {
            key: {
                "probe_features": int(mask.sum()),
                "detected_probe_features": int(detected[key].sum()),
                "raw_mass": masses[key],
            }
            for key, mask in categories.items()
        },
        "inclusion_by_gdna_qc": {
            key: {
                "probe_features": int(mask.sum()),
                "detected_probe_features": int(joint_detected[key].sum()),
                "raw_mass": joint_masses[key],
            }
            for key, mask in joint_categories.items()
        },
    }


def parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in labels:
            raise ValueError(f"expected unique LABEL=PATH: {value}")
        labels.add(label)
        result.append((label, Path(raw_path)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-h5ad", type=Path, required=True)
    parser.add_argument("--filtered-probe-set", type=Path, required=True)
    parser.add_argument("--raw-probe-h5", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    for path in (args.sr_h5ad, args.filtered_probe_set):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    raw_inputs = parse_labeled_paths(args.raw_probe_h5)
    for _, path in raw_inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    panel = load_filtered_probe_panel(args.filtered_probe_set)
    eligible_genes = set(panel.pop("eligible_genes"))
    h5ad = audit_h5ad(args.sr_h5ad, eligible_genes)
    if not h5ad["axis_exact"]:
        raise SystemExit("Space Ranger H5AD does not match the filtered probe-panel gene axis")
    raw = {label: audit_raw_probe_h5(path) for label, path in raw_inputs}

    rows = [{
        "source": "published_space_ranger_h5ad",
        "category": "filtered_panel_gene_matrix",
        "features_or_probes": h5ad["features"],
        "raw_mass": h5ad["raw_mass"],
        "panel_inclusion_role": "accepted biological comparator",
    }]
    for label, result in raw.items():
        for category, values in result["categories"].items():
            rows.append({
                "source": label,
                "category": category,
                "features_or_probes": values["probe_features"],
                "raw_mass": values["raw_mass"],
                "panel_inclusion_role": (
                    "included by 10x filtered reference; accepted expression"
                    if category == "filtered_probe_reference"
                    else "excluded by 10x default probe filter; raw diagnostic/control only"
                    if category == "diagnostic_outside_filtered_probe_reference"
                    else (
                        "orthogonal gDNA-QC annotation; never substitute for "
                        "included/target_sets"
                    )
                ),
            })

    args.out_dir.mkdir(parents=True)
    table = args.out_dir / "space_ranger_probe_accounting.tsv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "visium_hd_processing.space_ranger_probe_accounting.v2",
        "panel_inclusion_contract": {
            "included_field_and_target_sets_define_panel_membership": True,
            "included_false_and_deprecated_probes_excluded_by_default": True,
            "diagnostic_non_panel_counts_excluded": True,
            "excluded_probe_counts_retained_in_raw_probe_artifact": True,
            "accepted_expression_category": "filtered_probe_reference",
            "filtered_probes_field_role": (
                "passed gDNA-QC annotation; not the included/target_sets "
                "off-target exclusion"
            ),
        },
        "filtered_probe_panel": panel,
        "published_h5ad": h5ad,
        "raw_probe_h5": raw,
        "output": {
            "path": str(table.resolve()),
            "bytes": table.stat().st_size,
            "sha256": sha256(table),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
