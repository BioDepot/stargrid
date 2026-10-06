#!/usr/bin/env python3
"""Audit aggregate and elementwise differences between sparse policy fields."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator


DEFAULT_POLICIES = ("hard", "soft_expected", "gated_hard")


class CompensatedSum:
    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def value(self) -> float:
        return self.total + self.correction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> Iterator[tuple[str, float]]:
    previous: str | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        unit_column = next((field for field in fields if "unit_2um" in field), None)
        if unit_column is None or "molecule_mass" not in fields:
            raise ValueError(f"invalid bin-total schema: {path}")
        for row in reader:
            unit = row[unit_column]
            value = float(row["molecule_mass"])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid sparse value for {unit} in {path}: {value}")
            if previous is not None and unit <= previous:
                raise ValueError(f"bin-total field is not strictly sorted: {path}")
            previous = unit
            yield unit, value


def compare(baseline_path: Path, policy_path: Path) -> dict[str, object]:
    baseline = iter(rows(baseline_path))
    policy = iter(rows(policy_path))
    left = next(baseline, None)
    right = next(policy, None)
    baseline_mass = CompensatedSum()
    policy_mass = CompensatedSum()
    positive_mass = CompensatedSum()
    negative_mass = CompensatedSum()
    union_bins = equal_bins = higher_bins = lower_bins = 0
    minimum_difference = 0.0
    maximum_difference = 0.0

    while left is not None or right is not None:
        if right is None or (left is not None and left[0] < right[0]):
            left_value, right_value = left[1], 0.0
            left = next(baseline, None)
        elif left is None or right[0] < left[0]:
            left_value, right_value = 0.0, right[1]
            right = next(policy, None)
        else:
            left_value, right_value = left[1], right[1]
            left = next(baseline, None)
            right = next(policy, None)

        union_bins += 1
        baseline_mass.add(left_value)
        policy_mass.add(right_value)
        difference = right_value - left_value
        minimum_difference = min(minimum_difference, difference)
        maximum_difference = max(maximum_difference, difference)
        if difference > 0.0:
            higher_bins += 1
            positive_mass.add(difference)
        elif difference < 0.0:
            lower_bins += 1
            negative_mass.add(-difference)
        else:
            equal_bins += 1

    baseline_total = baseline_mass.value()
    policy_total = policy_mass.value()
    return {
        "baseline_mass": baseline_total,
        "policy_mass": policy_total,
        "net_mass_difference_policy_minus_baseline": policy_total - baseline_total,
        "union_occupied_bins": union_bins,
        "equal_bins": equal_bins,
        "policy_higher_bins": higher_bins,
        "policy_lower_bins": lower_bins,
        "positive_difference_mass": positive_mass.value(),
        "negative_difference_mass_absolute": negative_mass.value(),
        "minimum_bin_difference": minimum_difference,
        "maximum_bin_difference": maximum_difference,
        "elementwise_superset": lower_bins == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="strict")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.baseline in args.policies:
        parser.error("baseline must not also be a comparison policy")

    baseline_path = args.fields_root / f"{args.baseline}.bin_totals.tsv.gz"
    paths = {
        policy: args.fields_root / f"{policy}.bin_totals.tsv.gz"
        for policy in args.policies
    }
    for path in (baseline_path, *paths.values()):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    report = {
        "schema": "visium_hd_processing.policy_bin_total_nesting.v1",
        "interpretation": (
            "A net mass difference is not an independent increment cohort when "
            "elementwise_superset is false."
        ),
        "baseline": args.baseline,
        "inputs": {
            args.baseline: {
                "path": str(baseline_path.resolve()),
                "bytes": baseline_path.stat().st_size,
                "sha256": sha256(baseline_path),
            },
            **{
                policy: {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for policy, path in paths.items()
            },
        },
        "comparisons": {
            policy: compare(baseline_path, path)
            for policy, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
