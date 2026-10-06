#!/usr/bin/env python3
"""Aggregate resolved full-slide shards and match natural molecule memberships."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from star_spatial.hd_matched_molecules import (
    MoleculeAssignment,
    compact_residual_components,
    exact_member_matches,
    match_membership_components,
)


METHODS = (
    "strict_1mm_cr", "postcollapse_hard_1mm_cr", "gated_hard_1mm_cr",
    "postcollapse_soft_1mm_cr", "strict_exact", "postcollapse_hard_exact",
    "gated_hard_exact", "postcollapse_soft_exact", "space_ranger",
    "exact_matched_star", "exact_matched_space_ranger",
)
COMPONENT_FIELDS = (
    "component_id", "classification", "star_molecule_count",
    "space_ranger_molecule_count", "shared_read_count", "star_member_read_count",
    "space_ranger_member_read_count", "star_molecule_ids", "space_ranger_molecule_ids",
    "shared_read_ids",
)
EXACT_FIELDS = (
    "star_molecule_id", "space_ranger_molecule_id", "feature_agree", "umi_agree",
    "member_read_count", "member_read_ids", "tier", "star_unit_2um",
    "space_ranger_unit_2um", "agree_2um", "agree_8um", "agree_16um",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-shard", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--slide", default="H1-VM2JXXK")
    parser.add_argument("--area", default="A1")
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--field-only", action="store_true",
        help="Aggregate STAR policy fields without materializing STAR/SR molecule membership components.",
    )
    parser.add_argument(
        "--omit-match-ledgers", action="store_true",
        help=(
            "Compute exact member-set matches and component/tier summaries, but omit "
            "the row-level membership and exact-match ledgers."
        ),
    )
    parser.add_argument("--partial-dense", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--omit-ledger-header", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


class GzipWriter:
    def __init__(self, path, fields, *, write_header=True):
        self.handle = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if write_header:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.handle.close()


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _coordinate(unit):
    _, row, col = unit.rsplit("_", 2)
    return int(row), int(col)


def _assignment(row, sr=False):
    return MoleculeAssignment(
        row["molecule_id"], row["feature_id"], row["corrected_umi"], row["unit_2um"],
        tuple(sorted(set(row["member_read_ids"].split(";")))), row["tier"],
    )


def _add_field(field, unit, value, width, height):
    row, col = _coordinate(unit)
    if not 0 <= row < height or not 0 <= col < width:
        raise ValueError(f"unit outside grid: {unit}")
    field[row * width + col] += value


def _field_name(row):
    return f"{row['product']}_{row['umi_mode']}"


def _write_field(path, values, width):
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("unit_2um", "expected_count"))
        for index in np.flatnonzero(values):
            row, col = divmod(int(index), width)
            writer.writerow((f"s_002um_{row}_{col}", repr(float(values[index]))))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_single(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shard in args.resolved_shard:
        required_names = ["star_molecules.tsv.gz", "soft_expected.tsv.gz"]
        if not args.field_only:
            required_names.append("space_ranger_molecules.tsv.gz")
        for name in required_names:
            if not (shard / name).is_file():
                raise SystemExit(f"missing shard product: {shard / name}")

    size = args.width * args.height
    fields = {name: np.zeros(size, dtype=np.float64) for name in METHODS}
    components_out = None if args.omit_match_ledgers else GzipWriter(
        args.out_dir / "membership_components.tsv.gz", COMPONENT_FIELDS,
        write_header=not args.omit_ledger_header,
    )
    exact_out = None if args.omit_match_ledgers else GzipWriter(
        args.out_dir / "exact_member_matches.tsv.gz", EXACT_FIELDS,
        write_header=not args.omit_ledger_header,
    )
    summary = Counter()
    component_counts = Counter()
    tier_counts = Counter()
    tier_agree_2 = Counter()
    tier_agree_8 = Counter()
    tier_agree_16 = Counter()

    try:
        for shard in sorted(args.resolved_shard):
            star_primary = []
            star_by_id = {}
            for row in _rows(shard / "star_molecules.tsv.gz"):
                method = _field_name(row)
                if method not in fields:
                    raise ValueError(f"unexpected STAR method: {method}")
                _add_field(fields[method], row["unit_2um"], 1.0, args.width, args.height)
                summary[f"molecules_{method}"] += 1
                if (
                    not args.field_only
                    and row["umi_mode"] == "1mm_cr"
                    and row["product"] == "postcollapse_hard"
                ):
                    assignment = _assignment(row)
                    star_primary.append(assignment)
                    if components_out is not None:
                        star_by_id[assignment.molecule_id] = assignment
            for row in _rows(shard / "soft_expected.tsv.gz"):
                method = f"postcollapse_soft_{row['umi_mode']}"
                value = float(row["expected_count"])
                _add_field(fields[method], row["unit_2um"], value, args.width, args.height)
                summary[f"soft_rows_{row['umi_mode']}"] += 1
            sr = []
            sr_by_id = {}
            if not args.field_only:
                for row in _rows(shard / "space_ranger_molecules.tsv.gz"):
                    assignment = _assignment(row, sr=True)
                    sr.append(assignment)
                    if components_out is not None:
                        sr_by_id[assignment.molecule_id] = assignment
                    _add_field(fields["space_ranger"], assignment.candidate, 1.0, args.width, args.height)
                    summary["molecules_space_ranger"] += 1

            if args.field_only:
                components, matches = [], []
            else:
                matches = exact_member_matches(star_primary, sr)
                if args.omit_match_ledgers:
                    components = compact_residual_components(
                        star_primary, sr, matches,
                    )
                    component_counts["exact_member_set"] += len(matches)
                else:
                    components = match_membership_components(star_primary, sr)
            for component in components:
                component_counts[component.classification] += 1
                if components_out is not None:
                    star_reads = {
                        read for identifier in component.star_ids
                        for read in star_by_id[identifier].read_ids
                    }
                    sr_reads = {
                        read for identifier in component.sr_ids
                        for read in sr_by_id[identifier].read_ids
                    }
                    components_out.write({
                        "component_id": component.component_id,
                        "classification": component.classification,
                        "star_molecule_count": len(component.star_ids),
                        "space_ranger_molecule_count": len(component.sr_ids),
                        "shared_read_count": len(component.shared_read_ids),
                        "star_member_read_count": len(star_reads),
                        "space_ranger_member_read_count": len(sr_reads),
                        "star_molecule_ids": ";".join(component.star_ids),
                        "space_ranger_molecule_ids": ";".join(component.sr_ids),
                        "shared_read_ids": ";".join(component.shared_read_ids),
                    })
            for left, right in matches:
                lr, lc = _coordinate(left.candidate)
                rr, rc = _coordinate(right.candidate)
                agree2 = (lr, lc) == (rr, rc)
                agree8 = (lr // 4, lc // 4) == (rr // 4, rc // 4)
                agree16 = (lr // 8, lc // 8) == (rr // 8, rc // 8)
                tier = left.tier
                tier_counts[tier] += 1
                tier_agree_2[tier] += agree2
                tier_agree_8[tier] += agree8
                tier_agree_16[tier] += agree16
                _add_field(fields["exact_matched_star"], left.candidate, 1.0, args.width, args.height)
                _add_field(fields["exact_matched_space_ranger"], right.candidate, 1.0, args.width, args.height)
                if exact_out is not None:
                    exact_out.write({
                        "star_molecule_id": left.molecule_id,
                        "space_ranger_molecule_id": right.molecule_id,
                        "feature_agree": int(left.feature_id == right.feature_id),
                        "umi_agree": int(left.umi == right.umi),
                        "member_read_count": len(left.read_ids),
                        "member_read_ids": ";".join(left.read_ids),
                        "tier": tier,
                        "star_unit_2um": left.candidate,
                        "space_ranger_unit_2um": right.candidate,
                        "agree_2um": int(agree2),
                        "agree_8um": int(agree8),
                        "agree_16um": int(agree16),
                    })
            summary["exact_member_matches"] += len(matches)
    finally:
        if components_out is not None:
            components_out.close()
        if exact_out is not None:
            exact_out.close()

    field_dir = args.out_dir / ("fields_dense" if args.partial_dense else "fields")
    field_dir.mkdir()
    field_paths = []
    for name, values in fields.items():
        path = field_dir / (f"{name}.npy" if args.partial_dense else f"{name}.tsv.gz")
        if args.partial_dense:
            np.save(path, values, allow_pickle=False)
        else:
            _write_field(path, values, args.width)
        field_paths.append(path)
        summary[f"field_mass_{name}"] = float(values.sum())
        summary[f"occupied_bins_{name}"] = int(np.count_nonzero(values))

    tiers = [f"H{index}" for index in range(5)]
    with (args.out_dir / "tier_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tier", "matched_molecules", "agree_2um_fraction", "agree_8um_fraction", "agree_16um_fraction"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for tier in tiers + ["all"]:
            total = sum(tier_counts.values()) if tier == "all" else tier_counts[tier]
            a2 = sum(tier_agree_2.values()) if tier == "all" else tier_agree_2[tier]
            a8 = sum(tier_agree_8.values()) if tier == "all" else tier_agree_8[tier]
            a16 = sum(tier_agree_16.values()) if tier == "all" else tier_agree_16[tier]
            writer.writerow({
                "tier": tier, "matched_molecules": total,
                "agree_2um_fraction": a2 / total if total else 0.0,
                "agree_8um_fraction": a8 / total if total else 0.0,
                "agree_16um_fraction": a16 / total if total else 0.0,
            })

    manifest = {
        "schema": "star_spatial.hd.full_natural_matched_molecules.v1",
        "slide": args.slide, "area": args.area,
        "match_key": "exact_sorted_raw_member_read_ids",
        "coordinate_used_to_define_match": False,
        "primary_star": "postcollapse_hard_1mm_cr",
        "primary_space_ranger": "space_ranger",
        "membership_audit_enabled": not args.field_only,
        "match_ledgers_materialized": not args.omit_match_ledgers,
        "compact_exact_match_prefilter": args.omit_match_ledgers,
        "component_counts": dict(sorted(component_counts.items())),
        "tier_raw": {
            tier: {
                "matched_molecules": tier_counts[tier],
                "agree_2um": tier_agree_2[tier],
                "agree_8um": tier_agree_8[tier],
                "agree_16um": tier_agree_16[tier],
            }
            for tier in sorted(tier_counts)
        },
        "counts": dict(sorted(summary.items())),
        "resolved_shards": [str(path) for path in sorted(args.resolved_shard)],
        "prohibited_prior_flags": {
            "spatial": False, "image": False, "expression": False,
            "cell_type": False, "neighborhood": False, "graph": False,
            "space_ranger_assignment": False,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    products = [args.out_dir / "tier_metrics.tsv", summary_path, *field_paths]
    if not args.omit_match_ledgers:
        products[:0] = [
            args.out_dir / "membership_components.tsv.gz",
            args.out_dir / "exact_member_matches.tsv.gz",
        ]
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(args.out_dir)}\n" for path in products),
        encoding="utf-8",
    )
    return 0


def _run_partial(command):
    subprocess.run(command, check=True)


def _write_tier_metrics(path, tier_counts, tier_agree_2, tier_agree_8, tier_agree_16):
    tiers = [f"H{index}" for index in range(5)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tier", "matched_molecules", "agree_2um_fraction", "agree_8um_fraction", "agree_16um_fraction"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for tier in tiers + ["all"]:
            total = sum(tier_counts.values()) if tier == "all" else tier_counts[tier]
            a2 = sum(tier_agree_2.values()) if tier == "all" else tier_agree_2[tier]
            a8 = sum(tier_agree_8.values()) if tier == "all" else tier_agree_8[tier]
            a16 = sum(tier_agree_16.values()) if tier == "all" else tier_agree_16[tier]
            writer.writerow({
                "tier": tier, "matched_molecules": total,
                "agree_2um_fraction": a2 / total if total else 0.0,
                "agree_8um_fraction": a8 / total if total else 0.0,
                "agree_16um_fraction": a16 / total if total else 0.0,
            })


def _run_parallel(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.resolved_shard)
    worker_count = min(args.workers, len(shards))
    chunk_size = (len(shards) + worker_count - 1) // worker_count
    groups = [shards[start:start + chunk_size] for start in range(0, len(shards), chunk_size)]
    partial_root = args.out_dir / "partials"
    partial_root.mkdir()
    partial_dirs = []
    commands = []
    for index, group in enumerate(groups):
        partial = partial_root / f"group-{index:03d}"
        partial_dirs.append(partial)
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--out-dir", str(partial), "--width", str(args.width),
            "--height", str(args.height), "--workers", "1", "--partial-dense",
        ]
        if args.field_only:
            command.append("--field-only")
        if args.omit_match_ledgers:
            command.append("--omit-match-ledgers")
        if index:
            command.append("--omit-ledger-header")
        for shard in group:
            command.extend(("--resolved-shard", str(shard)))
        commands.append(command)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_run_partial, command) for command in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if not args.omit_match_ledgers:
        for name in ("membership_components.tsv.gz", "exact_member_matches.tsv.gz"):
            with (args.out_dir / name).open("wb") as output:
                for partial in partial_dirs:
                    with (partial / name).open("rb") as source:
                        shutil.copyfileobj(source, output, length=16 << 20)

    summary = Counter()
    component_counts = Counter()
    tier_counts = Counter()
    tier_agree_2 = Counter()
    tier_agree_8 = Counter()
    tier_agree_16 = Counter()
    partial_manifests = []
    for partial in partial_dirs:
        manifest = json.loads((partial / "summary.json").read_text(encoding="utf-8"))
        partial_manifests.append(manifest)
        for key, value in manifest["counts"].items():
            if not key.startswith(("field_mass_", "occupied_bins_")):
                summary[key] += value
        component_counts.update(manifest["component_counts"])
        for tier, values in manifest.get("tier_raw", {}).items():
            tier_counts[tier] += values["matched_molecules"]
            tier_agree_2[tier] += values["agree_2um"]
            tier_agree_8[tier] += values["agree_8um"]
            tier_agree_16[tier] += values["agree_16um"]

    field_dir = args.out_dir / "fields"
    field_dir.mkdir()
    field_paths = []
    size = args.width * args.height
    for name in METHODS:
        values = np.zeros(size, dtype=np.float64)
        for partial in partial_dirs:
            values += np.load(partial / "fields_dense" / f"{name}.npy", mmap_mode="r")
        path = field_dir / f"{name}.tsv.gz"
        _write_field(path, values, args.width)
        field_paths.append(path)
        summary[f"field_mass_{name}"] = float(values.sum())
        summary[f"occupied_bins_{name}"] = int(np.count_nonzero(values))

    tier_path = args.out_dir / "tier_metrics.tsv"
    _write_tier_metrics(
        tier_path, tier_counts, tier_agree_2, tier_agree_8, tier_agree_16,
    )
    manifest = {
        "schema": "star_spatial.hd.full_natural_matched_molecules.v1",
        "slide": args.slide, "area": args.area,
        "match_key": "exact_sorted_raw_member_read_ids",
        "coordinate_used_to_define_match": False,
        "primary_star": "postcollapse_hard_1mm_cr",
        "primary_space_ranger": "space_ranger",
        "membership_audit_enabled": not args.field_only,
        "match_ledgers_materialized": not args.omit_match_ledgers,
        "compact_exact_match_prefilter": args.omit_match_ledgers,
        "audit_workers": worker_count,
        "component_counts": dict(sorted(component_counts.items())),
        "tier_raw": {
            tier: {
                "matched_molecules": tier_counts[tier],
                "agree_2um": tier_agree_2[tier],
                "agree_8um": tier_agree_8[tier],
                "agree_16um": tier_agree_16[tier],
            }
            for tier in sorted(tier_counts)
        },
        "counts": dict(sorted(summary.items())),
        "resolved_shards": [str(path) for path in shards],
        "prohibited_prior_flags": {
            "spatial": False, "image": False, "expression": False,
            "cell_type": False, "neighborhood": False, "graph": False,
            "space_ranger_assignment": False,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    products = [tier_path, summary_path, *field_paths]
    if not args.omit_match_ledgers:
        products[:0] = [
            args.out_dir / "membership_components.tsv.gz",
            args.out_dir / "exact_member_matches.tsv.gz",
        ]
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(args.out_dir)}\n" for path in products),
        encoding="utf-8",
    )
    return 0


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    if args.workers > 1 and not args.partial_dense:
        return _run_parallel(args)
    return _run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
