"""Strict-UMI broad spatial field for candidate-preserving Visium HD models."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .hd_assignment_ledger import bin_unit_id


BROAD_FIELD_SCHEMA = "star_spatial.hd.strict_broad_field_16um.v1"
BROAD_RESOLVER_SCHEMA = "star_spatial.hd.probabilistic_umi_broad16.v1"


@dataclass(frozen=True)
class StrictMolecule:
    strict_molecule_id: str
    feature_id: str
    umi: str
    row2: int
    col2: int
    raw_read_count: int
    split: str = ""

    def __post_init__(self) -> None:
        if not self.strict_molecule_id or not self.feature_id or not self.umi:
            raise ValueError("strict molecule ID, feature ID, and UMI are required")
        if self.row2 < 0 or self.col2 < 0:
            raise ValueError("strict molecule coordinates must be non-negative")
        if not isinstance(self.raw_read_count, int) or self.raw_read_count < 1:
            raise ValueError("strict molecule raw read count must be a positive integer")

    @property
    def unit_2um_id(self) -> str:
        return bin_unit_id(self.row2, self.col2, 2)

    @property
    def unit_16um_id(self) -> str:
        return bin_unit_id(self.row2, self.col2, 16)


@dataclass(frozen=True)
class StrictBroadField:
    parent_counts: Mapping[str, int]
    parent_contributors: Mapping[str, tuple[str, ...]]
    sharp_counts: Mapping[str, int]
    sharp_contributors: Mapping[str, tuple[str, ...]]
    molecule_parent: Mapping[str, str]
    strict_molecule_count: int
    raw_read_count: int

    def log_prior(
        self, candidate_2um: str, *, alpha: float, excluded_molecule_id: str | None = None
    ) -> float:
        """Return broad log support, optionally subtracting one source molecule."""
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("broad-field alpha must be finite and positive")
        parent = candidate_parent(candidate_2um, size_um=16)
        count = self.parent_counts.get(parent, 0)
        if excluded_molecule_id is not None:
            source_parent = self.molecule_parent.get(excluded_molecule_id)
            if source_parent == parent:
                count -= 1
        if count < 0:
            raise ValueError("leave-one-molecule-out broad support became negative")
        return math.log(count + alpha)


def candidate_parent(candidate_2um: str, *, size_um: int = 16) -> str:
    prefix = "s_002um_"
    if not candidate_2um.startswith(prefix):
        raise ValueError(f"candidate is not a 2 um HD unit: {candidate_2um}")
    try:
        row, col = candidate_2um[len(prefix):].split("_", 1)
        return bin_unit_id(int(row), int(col), size_um)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"candidate is not a 2 um HD unit: {candidate_2um}") from exc


def build_strict_broad_field(molecules: Iterable[StrictMolecule]) -> StrictBroadField:
    """Collapse PCR reads to strict molecules and then to immutable 2/16 um fields."""
    parents: Counter[str] = Counter()
    sharp: Counter[str] = Counter()
    parent_contributors: dict[str, list[str]] = defaultdict(list)
    sharp_contributors: dict[str, list[str]] = defaultdict(list)
    molecule_parent: dict[str, str] = {}
    molecule_keys: set[tuple[str, str, int, int]] = set()
    raw_reads = 0
    for molecule in molecules:
        if molecule.strict_molecule_id in molecule_parent:
            raise ValueError(f"duplicate strict molecule ID: {molecule.strict_molecule_id}")
        molecule_key = (
            molecule.feature_id, molecule.umi, molecule.row2, molecule.col2
        )
        if molecule_key in molecule_keys:
            raise ValueError(f"duplicate strict molecule key: {molecule_key}")
        molecule_keys.add(molecule_key)
        parent = molecule.unit_16um_id
        unit_2um = molecule.unit_2um_id
        molecule_parent[molecule.strict_molecule_id] = parent
        parents[parent] += 1
        sharp[unit_2um] += 1
        parent_contributors[parent].append(molecule.strict_molecule_id)
        sharp_contributors[unit_2um].append(molecule.strict_molecule_id)
        raw_reads += molecule.raw_read_count
    return StrictBroadField(
        parent_counts=dict(sorted(parents.items())),
        parent_contributors={
            unit: tuple(sorted(values))
            for unit, values in sorted(parent_contributors.items())
        },
        sharp_counts=dict(sorted(sharp.items())),
        sharp_contributors={
            unit: tuple(sorted(values))
            for unit, values in sorted(sharp_contributors.items())
        },
        molecule_parent=dict(sorted(molecule_parent.items())),
        strict_molecule_count=len(molecule_parent),
        raw_read_count=raw_reads,
    )


def candidate_broad_log_priors(
    candidates: Iterable[str],
    field: StrictBroadField,
    *,
    alpha: float,
) -> dict[str, float]:
    return {
        candidate: field.log_prior(candidate, alpha=alpha)
        for candidate in sorted(set(candidates))
    }


def candidate_broad_log_priors_from_counts(
    candidates: Iterable[str],
    parent_counts: Mapping[str, int],
    *,
    alpha: float,
) -> dict[str, float]:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("broad-field alpha must be finite and positive")
    if any(not isinstance(count, int) or count < 0 for count in parent_counts.values()):
        raise ValueError("strict broad-field counts must be non-negative integers")
    return {
        candidate: math.log(parent_counts.get(candidate_parent(candidate), 0) + alpha)
        for candidate in sorted(set(candidates))
    }
