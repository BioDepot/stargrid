"""Assignment-neutral molecule membership matching for Visium HD audits."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MoleculeAssignment:
    molecule_id: str
    feature_id: str
    umi: str
    candidate: str
    read_ids: tuple[str, ...]
    tier: str = ""

    def __post_init__(self) -> None:
        if not self.molecule_id or not self.read_ids:
            raise ValueError("molecule_id and member reads are required")
        if tuple(sorted(set(self.read_ids))) != self.read_ids:
            raise ValueError("read_ids must be sorted and unique")


@dataclass(frozen=True)
class MembershipComponent:
    component_id: str
    classification: str
    star_ids: tuple[str, ...]
    sr_ids: tuple[str, ...]
    shared_read_ids: tuple[str, ...]


def match_membership_components(
    star: Iterable[MoleculeAssignment],
    space_ranger: Iterable[MoleculeAssignment],
) -> list[MembershipComponent]:
    """Build deterministic bipartite components from shared raw read names."""
    star_list = list(star)
    sr_list = list(space_ranger)
    star_rows = {row.molecule_id: row for row in star_list}
    sr_rows = {row.molecule_id: row for row in sr_list}
    if len(star_rows) != len(star_list) or len(sr_rows) != len(sr_list):
        raise ValueError("duplicate molecule identifiers")
    star_by_read = _one_molecule_per_read(star_rows.values(), "STAR")
    sr_by_read = _one_molecule_per_read(sr_rows.values(), "Space Ranger")

    star_to_sr: dict[str, set[str]] = defaultdict(set)
    sr_to_star: dict[str, set[str]] = defaultdict(set)
    pair_reads: dict[tuple[str, str], set[str]] = defaultdict(set)
    for read_id in sorted(set(star_by_read) & set(sr_by_read)):
        left, right = star_by_read[read_id], sr_by_read[read_id]
        star_to_sr[left].add(right)
        sr_to_star[right].add(left)
        pair_reads[(left, right)].add(read_id)

    output: list[MembershipComponent] = []
    seen_star: set[str] = set()
    seen_sr: set[str] = set()
    for seed_kind, seed in [
        *(('star', value) for value in sorted(star_rows)),
        *(('sr', value) for value in sorted(sr_rows)),
    ]:
        if (seed_kind == "star" and seed in seen_star) or (
            seed_kind == "sr" and seed in seen_sr
        ):
            continue
        queue = deque([(seed_kind, seed)])
        component_star: set[str] = set()
        component_sr: set[str] = set()
        while queue:
            kind, node = queue.popleft()
            if kind == "star":
                if node in component_star:
                    continue
                component_star.add(node)
                seen_star.add(node)
                queue.extend(("sr", other) for other in sorted(star_to_sr[node]))
            else:
                if node in component_sr:
                    continue
                component_sr.add(node)
                seen_sr.add(node)
                queue.extend(("star", other) for other in sorted(sr_to_star[node]))
        shared = sorted({
            read_id
            for left in component_star
            for right in component_sr
            for read_id in pair_reads.get((left, right), ())
        })
        classification = _classify_component(
            component_star, component_sr, shared, star_rows, sr_rows
        )
        star_ids = tuple(sorted(component_star))
        sr_ids = tuple(sorted(component_sr))
        component_id = "cmp_" + _digest_parts((*star_ids, "|", *sr_ids))
        output.append(MembershipComponent(
            component_id, classification, star_ids, sr_ids, tuple(shared)
        ))
    return sorted(output, key=lambda row: row.component_id)


def exact_member_matches(
    star: Iterable[MoleculeAssignment],
    space_ranger: Iterable[MoleculeAssignment],
) -> list[tuple[MoleculeAssignment, MoleculeAssignment]]:
    """Return unique one-to-one matches with identical raw member-read sets."""
    left: dict[tuple[str, ...], list[MoleculeAssignment]] = defaultdict(list)
    right: dict[tuple[str, ...], list[MoleculeAssignment]] = defaultdict(list)
    for row in star:
        left[row.read_ids].append(row)
    for row in space_ranger:
        right[row.read_ids].append(row)
    matches = []
    for members in sorted(set(left) & set(right)):
        if len(left[members]) == 1 and len(right[members]) == 1:
            matches.append((left[members][0], right[members][0]))
    return matches


def compact_residual_components(
    star: Iterable[MoleculeAssignment],
    space_ranger: Iterable[MoleculeAssignment],
    matches: Iterable[tuple[MoleculeAssignment, MoleculeAssignment]],
) -> list[MembershipComponent]:
    """Remove exact pairs before classifying the smaller residual graph."""
    matched_star: set[str] = set()
    matched_sr: set[str] = set()
    for left, right in matches:
        matched_star.add(left.molecule_id)
        matched_sr.add(right.molecule_id)
    return match_membership_components(
        (row for row in star if row.molecule_id not in matched_star),
        (row for row in space_ranger if row.molecule_id not in matched_sr),
    )


def _one_molecule_per_read(
    rows: Iterable[MoleculeAssignment], method: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        for read_id in row.read_ids:
            previous = result.setdefault(read_id, row.molecule_id)
            if previous != row.molecule_id:
                raise ValueError(f"{method} read belongs to multiple molecules: {read_id}")
    return result


def _classify_component(star_ids, sr_ids, shared, star_rows, sr_rows) -> str:
    if not star_ids:
        return "space_ranger_only"
    if not sr_ids:
        return "star_only"
    if len(star_ids) == 1 and len(sr_ids) == 1:
        left = star_rows[next(iter(star_ids))].read_ids
        right = sr_rows[next(iter(sr_ids))].read_ids
        return "exact_member_set" if left == right else "one_to_one_partial"
    if len(star_ids) == 1:
        return "space_ranger_split"
    if len(sr_ids) == 1:
        return "star_split"
    return "complex_split_merge"


def _digest_parts(parts: tuple[str, ...]) -> str:
    import hashlib

    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]
