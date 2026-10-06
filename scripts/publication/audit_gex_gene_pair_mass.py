#!/usr/bin/env python3
"""Audit selected whole-slide gene totals in Space Ranger H5 and STAR MEX.

The Matrix Market reader streams the file and retains only requested feature
rows, so the audit does not materialize a full gene-by-bin matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def parse_assignment(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def parse_gene(value: str) -> tuple[str, str]:
    identifier, separator, name = value.partition("=")
    if not identifier:
        raise argparse.ArgumentTypeError("expected GENE_ID=GENE_NAME")
    return identifier, name if separator and name else identifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def load_sr_totals(path: Path, genes: list[str]) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        identifiers = [
            decode(value).split(".", 1)[0]
            for value in matrix["features"]["id"][:]
        ]
        indexes = {identifier: index for index, identifier in enumerate(identifiers)}
        missing = sorted(set(genes) - set(indexes))
        if missing:
            raise ValueError(f"genes absent from Space Ranger H5: {missing}")
        target_to_gene = {indexes[gene]: gene for gene in genes}
        totals = {gene: 0.0 for gene in genes}
        data = matrix["data"]
        feature_indices = matrix["indices"]
        for begin in range(0, len(data), 10_000_000):
            end = min(len(data), begin + 10_000_000)
            local_indices = np.asarray(feature_indices[begin:end], dtype=np.int64)
            local_data = np.asarray(data[begin:end], dtype=np.float64)
            for index, gene in target_to_gene.items():
                totals[gene] += float(local_data[local_indices == index].sum())
    return totals


def load_mex_totals(root: Path, genes: list[str]) -> tuple[dict[str, float], dict[str, Path]]:
    features_path = root / "features.tsv"
    matrix_path = root / "matrix.mtx"
    if not features_path.is_file() or not matrix_path.is_file():
        raise ValueError(f"incomplete MEX directory: {root}")
    row_to_gene: dict[int, str] = {}
    with features_path.open(encoding="utf-8") as handle:
        for row, line in enumerate(handle, start=1):
            identifier = line.rstrip("\n").split("\t", 1)[0].split(".", 1)[0]
            if identifier in genes:
                if identifier in row_to_gene.values():
                    raise ValueError(f"duplicate STAR feature: {identifier}")
                row_to_gene[row] = identifier
    missing = sorted(set(genes) - set(row_to_gene.values()))
    if missing:
        raise ValueError(f"genes absent from STAR MEX: {missing}")
    totals = {gene: 0.0 for gene in genes}
    with matrix_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            dimensions = line.split()
            if len(dimensions) != 3:
                raise ValueError("invalid Matrix Market dimensions")
            break
        for line in handle:
            row_text, _, value_text = line.split()
            gene = row_to_gene.get(int(row_text))
            if gene is not None:
                totals[gene] += float(value_text)
    return totals, {"features": features_path, "matrix": matrix_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-ranger-h5", type=Path, required=True)
    parser.add_argument("--star-method", type=parse_assignment, action="append", required=True)
    parser.add_argument("--gene", type=parse_gene, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    gene_names = dict(args.gene)
    genes = list(gene_names)
    if len(genes) != len(args.gene):
        raise SystemExit("duplicate --gene identifier")

    totals: dict[str, dict[str, float]] = {
        "space_ranger": load_sr_totals(args.space_ranger_h5, genes)
    }
    inputs = {
        str(args.space_ranger_h5.resolve()): sha256(args.space_ranger_h5),
    }
    for label, root in args.star_method:
        if label in totals:
            raise SystemExit(f"duplicate method: {label}")
        totals[label], paths = load_mex_totals(root, genes)
        for path in paths.values():
            inputs[str(path.resolve())] = sha256(path)

    sr_pair = sum(totals["space_ranger"].values())
    rows = []
    for method, values in totals.items():
        pair = sum(values.values())
        for gene in genes:
            sr_value = totals["space_ranger"][gene]
            value = values[gene]
            rows.append({
                "method": method,
                "gene_id": gene,
                "gene_name": gene_names[gene],
                "whole_slide_mass": value,
                "net_vs_space_ranger": value - sr_value,
                "percent_vs_space_ranger": (
                    100.0 * (value - sr_value) / sr_value if sr_value else ""
                ),
                "pair_mass": pair,
                "pair_mass_percent_vs_space_ranger": (
                    100.0 * (pair - sr_pair) / sr_pair if sr_pair else ""
                ),
                "gene_fraction_of_pair": value / pair if pair else "",
            })

    args.out_dir.mkdir(parents=True)
    table_path = args.out_dir / "gene_pair_mass.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "visium_hd_processing.gex_gene_pair_mass_audit.v1",
        "genes": gene_names,
        "inputs_sha256": inputs,
        "methods": totals,
        "invariants": {
            "whole_slide_raw_mass": True,
            "no_aligner_or_counter_rerun": True,
            "percent_denominator_is_space_ranger_once": True,
        },
        "output": str(table_path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
