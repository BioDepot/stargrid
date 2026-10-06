#!/usr/bin/env python3
"""Compare open and Space Ranger molecules by exact raw read membership."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from star_spatial.hd_matched_molecules import (
    MoleculeAssignment,
    exact_member_matches,
    match_membership_components,
)


COORDINATE = re.compile(r"^s_002um_(\d+)_(\d+)(?:-\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_unit(value: str) -> str:
    match = COORDINATE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid 2 um unit: {value}")
    return f"s_002um_{int(match.group(1))}_{int(match.group(2))}"


def stable_id(prefix: str, fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1f".join(fields).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def load_open(path: Path, umi_mode: str, product: str) -> list[MoleculeAssignment]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "umi_mode", "product", "molecule_id", "feature_id", "corrected_umi",
            "unit_2um", "member_read_ids", "tier",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("open molecule ledger has the wrong schema")
        for row in reader:
            if row["umi_mode"] != umi_mode or row["product"] != product:
                continue
            rows.append(MoleculeAssignment(
                row["molecule_id"], row["feature_id"], row["corrected_umi"],
                normalized_unit(row["unit_2um"]),
                tuple(sorted(set(row["member_read_ids"].split(";")))), row["tier"],
            ))
    return sorted(rows, key=lambda row: row.molecule_id)


def load_vendor(path: Path) -> list[MoleculeAssignment]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "eligible", "feature_id", "corrected_umi", "unit_2um"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("vendor read ledger has the wrong schema")
        for row in reader:
            if row["eligible"] not in {"1", "true", "True"}:
                continue
            key = (
                row["feature_id"], row["corrected_umi"], normalized_unit(row["unit_2um"]),
            )
            groups.setdefault(key, []).append(row["read_id"])
    rows = []
    for (gene, umi, unit), members in sorted(groups.items()):
        read_ids = tuple(sorted(set(members)))
        if len(read_ids) != len(members):
            raise ValueError(f"duplicate vendor family read: {(gene, umi, unit)}")
        rows.append(MoleculeAssignment(
            stable_id("srm", (gene, umi, unit, *read_ids)), gene, umi, unit, read_ids,
        ))
    return sorted(rows, key=lambda row: row.molecule_id)


def coordinate(value: str) -> tuple[int, int]:
    match = COORDINATE.fullmatch(value)
    if match is None:
        raise ValueError(value)
    return int(match.group(1)), int(match.group(2))


def summarize(
    star: list[MoleculeAssignment], vendor: list[MoleculeAssignment],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    components = match_membership_components(star, vendor)
    exact = exact_member_matches(star, vendor)
    classes = Counter(row.classification for row in components)
    metrics = Counter()
    exact_rows = []
    for left, right in exact:
        lc, rc = coordinate(left.candidate), coordinate(right.candidate)
        metrics["feature"] += left.feature_id == right.feature_id
        metrics["corrected_umi"] += left.umi == right.umi
        metrics["coordinate_2um"] += lc == rc
        metrics["coordinate_8um"] += (lc[0] // 4, lc[1] // 4) == (rc[0] // 4, rc[1] // 4)
        metrics["coordinate_16um"] += (lc[0] // 8, lc[1] // 8) == (rc[0] // 8, rc[1] // 8)
        exact_rows.append({
            "open_molecule_id": left.molecule_id,
            "space_ranger_molecule_id": right.molecule_id,
            "member_read_count": len(left.read_ids),
            "member_read_ids": ";".join(left.read_ids),
            "feature_concordant": int(left.feature_id == right.feature_id),
            "corrected_umi_concordant": int(left.umi == right.umi),
            "coordinate_2um_concordant": int(lc == rc),
            "coordinate_8um_concordant": int((lc[0] // 4, lc[1] // 4) == (rc[0] // 4, rc[1] // 4)),
            "coordinate_16um_concordant": int((lc[0] // 8, lc[1] // 8) == (rc[0] // 8, rc[1] // 8)),
        })
    component_rows = [{
        "component_id": row.component_id,
        "classification": row.classification,
        "open_molecule_ids": ";".join(row.star_ids),
        "space_ranger_molecule_ids": ";".join(row.sr_ids),
        "shared_read_count": len(row.shared_read_ids),
        "shared_read_ids": ";".join(row.shared_read_ids),
    } for row in components]
    summary = {
        "open_molecules": len(star),
        "space_ranger_molecules": len(vendor),
        "raw_molecule_mass_difference": len(star) - len(vendor),
        "open_member_reads": sum(len(row.read_ids) for row in star),
        "space_ranger_member_reads": sum(len(row.read_ids) for row in vendor),
        "exact_member_set_matches": len(exact),
        "exact_member_set_reads": sum(len(left.read_ids) for left, _ in exact),
        "exact_fraction_of_open": len(exact) / len(star) if star else 0.0,
        "exact_fraction_of_space_ranger": len(exact) / len(vendor) if vendor else 0.0,
        "component_classes": dict(sorted(classes.items())),
        "exact_identity": dict(sorted(metrics.items())),
    }
    return summary, component_rows, exact_rows


def write_rows(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0)
    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-molecules", type=Path, required=True)
    parser.add_argument("--vendor-read-ledger", type=Path, required=True)
    parser.add_argument("--umi-mode", default="1mm_cr")
    parser.add_argument("--product", default="postcollapse_hard")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.open_molecules, args.vendor_read_ledger)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    star = load_open(args.open_molecules, args.umi_mode, args.product)
    vendor = load_vendor(args.vendor_read_ledger)
    counts, component_rows, exact_rows = summarize(star, vendor)

    args.out_dir.mkdir(parents=True)
    components_path = args.out_dir / "membership_components.tsv.gz"
    exact_path = args.out_dir / "exact_member_matches.tsv.gz"
    write_rows(components_path, component_rows, (
        "component_id", "classification", "open_molecule_ids",
        "space_ranger_molecule_ids", "shared_read_count", "shared_read_ids",
    ))
    write_rows(exact_path, exact_rows, (
        "open_molecule_id", "space_ranger_molecule_id", "member_read_count",
        "member_read_ids", "feature_concordant", "corrected_umi_concordant",
        "coordinate_2um_concordant", "coordinate_8um_concordant",
        "coordinate_16um_concordant",
    ))
    summary = {
        "schema": "visium_hd_processing.flex_molecule_membership_concordance.v1",
        "method": {"umi_mode": args.umi_mode, "product": args.product},
        "counts": counts,
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {
            "components": {"path": str(components_path.resolve()), "sha256": sha256(components_path)},
            "exact_matches": {"path": str(exact_path.resolve()), "sha256": sha256(exact_path)},
        },
        "invariants": {
            "matching_uses_only_exact_raw_member_sets": True,
            "coordinate_not_used_to_define_match": True,
            "raw_mass_not_normalized": True,
            "occupancy_jaccard_not_computed": True,
            "vendor_not_used_for_open_assignment": True,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
