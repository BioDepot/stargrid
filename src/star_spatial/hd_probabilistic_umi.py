"""Clear, policy-neutral reference model for candidate-preserving HD UMIs.

This module intentionally has no expression, cell-type, neighbourhood, graph,
GPU, histology, or Space Ranger dependency.  Inputs are finite sequence
candidate sets; read weights are combined before a read clique emits molecule mass.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .hd_assignment_ledger import bin_unit_id


SCHEMA = "star_spatial.hd.probabilistic_umi.v1"


@dataclass(frozen=True)
class CandidateRead:
    read_id: str
    feature_id: str
    umi: str
    log_likelihoods: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.read_id or not self.feature_id or not self.umi:
            raise ValueError("read_id, feature_id, and umi are required")
        if not self.log_likelihoods:
            raise ValueError("a read must retain at least one candidate")
        if any(not math.isfinite(value) for value in self.log_likelihoods.values()):
            raise ValueError("candidate log likelihoods must be finite")


@dataclass(frozen=True)
class ReadClique:
    clique_id: str
    feature_id: str
    umi: str
    read_ids: tuple[str, ...]
    candidates: tuple[str, ...]
    log_likelihood_sums: tuple[float, ...]
    log_read_priors: tuple[float, ...]
    log_broad_priors: tuple[float, ...]
    log_evidence: tuple[float, ...]
    posterior: tuple[float, ...]

    def probabilities(self) -> dict[str, float]:
        return dict(zip(self.candidates, self.posterior, strict=True))


@dataclass(frozen=True)
class Occupancy:
    feature_id: str
    corrected_umi: str
    candidate: str
    occupancy: float
    contributing_cliques: tuple[str, ...]


@dataclass(frozen=True)
class HardCall:
    clique_id: str
    status: str
    candidate: str | None
    posterior: float
    margin: float
    reason: str


def _stable_id(prefix: str, parts: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def normalize_log_weights(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("cannot normalize an empty sequence")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def build_read_cliques(
    reads: Iterable[CandidateRead],
    *,
    log_read_prior: Mapping[str, float] | None = None,
    log_broad_prior: Mapping[str, float] | None = None,
    temperature: float = 1.0,
    prior_beta: float = 1.0,
    spatial_lambda: float = 0.0,
) -> list[ReadClique]:
    """Build deterministic read cliques with a nonempty global candidate intersection.

    The stable sorted greedy partition prevents an unrestricted A-B, B-C chain
    from manufacturing an A-C clique. Each output clique represents one unit
    of molecule mass even when it contains many PCR reads.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(prior_beta) or prior_beta < 0:
        raise ValueError("prior_beta must be finite and non-negative")
    if not math.isfinite(spatial_lambda) or spatial_lambda < 0:
        raise ValueError("spatial_lambda must be finite and non-negative")
    log_read_prior = {} if log_read_prior is None else dict(log_read_prior)
    log_broad_prior = {} if log_broad_prior is None else dict(log_broad_prior)
    if any(not math.isfinite(value) for value in log_read_prior.values()):
        raise ValueError("read-prior log weights must be finite")
    if any(not math.isfinite(value) for value in log_broad_prior.values()):
        raise ValueError("broad-prior log weights must be finite")
    if spatial_lambda > 0 and not log_broad_prior:
        raise ValueError("a broad-prior table is required when spatial_lambda is positive")
    grouped: dict[tuple[str, str], list[CandidateRead]] = defaultdict(list)
    seen: set[str] = set()
    for read in reads:
        if read.read_id in seen:
            raise ValueError(f"duplicate read_id: {read.read_id}")
        seen.add(read.read_id)
        grouped[(read.feature_id, read.umi)].append(read)

    output: list[ReadClique] = []
    for (feature, umi), rows in sorted(grouped.items()):
        partitions: list[tuple[set[str], list[CandidateRead]]] = []
        for read in sorted(
            rows, key=lambda row: (row.read_id, tuple(sorted(row.log_likelihoods)))
        ):
            choices = [
                (tuple(sorted(intersection & set(read.log_likelihoods))), index)
                for index, (intersection, _) in enumerate(partitions)
                if intersection & set(read.log_likelihoods)
            ]
            if choices:
                _, index = min(choices)
                intersection, members = partitions[index]
                partitions[index] = (
                    intersection & set(read.log_likelihoods), members + [read]
                )
            else:
                partitions.append((set(read.log_likelihoods), [read]))

        for intersection, members in partitions:
            candidates = tuple(sorted(intersection))
            likelihood_sums = tuple(
                math.fsum(read.log_likelihoods[candidate] for read in members)
                for candidate in candidates
            )
            prior = tuple(log_read_prior.get(candidate, 0.0) for candidate in candidates)
            try:
                broad_prior = tuple(
                    log_broad_prior[candidate] if spatial_lambda > 0 else 0.0
                    for candidate in candidates
                )
            except KeyError as exc:
                raise ValueError(f"missing broad prior for candidate: {exc.args[0]}") from exc
            evidence = tuple(
                likelihood / temperature
                + prior_beta * prior_value
                + spatial_lambda * broad_value
                for likelihood, prior_value, broad_value in zip(
                    likelihood_sums, prior, broad_prior, strict=True
                )
            )
            member_ids = tuple(sorted(read.read_id for read in members))
            clique_id = _stable_id("clq", (feature, umi, *member_ids, *candidates))
            output.append(
                ReadClique(
                    clique_id,
                    feature,
                    umi,
                    member_ids,
                    candidates,
                    likelihood_sums,
                    prior,
                    broad_prior,
                    evidence,
                    normalize_log_weights(evidence),
                )
            )
    return sorted(output, key=lambda clique: clique.clique_id)


def factorized_h0_read_log_prior(
    bc1_index: int,
    bc2_index: int,
    counts: Mapping[tuple[str, int], int],
    *,
    alpha: float = 1.0,
) -> float:
    """Return a smoothed factorized prior from raw exact-H0 read counts.

    Counts intentionally include PCR-amplified reads and are not UMI
    deduplicated. The omitted normalization constant cancels within every
    finite candidate set.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if bc1_index < 0 or bc2_index < 0:
        raise ValueError("oligo indices must be non-negative")
    try:
        bc1_count = counts[("BC1", bc1_index)]
        bc2_count = counts[("BC2", bc2_index)]
    except KeyError as exc:
        raise ValueError(f"missing exact-H0 read count for {exc.args[0]}") from exc
    if not isinstance(bc1_count, int) or not isinstance(bc2_count, int):
        raise ValueError("exact-H0 read counts must be integers")
    if bc1_count < 0 or bc2_count < 0:
        raise ValueError("exact-H0 read counts must be non-negative")
    return math.log(bc1_count + alpha) + math.log(bc2_count + alpha)


def _hamming_one(left: str, right: str) -> bool:
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right)) == 1


def corrected_umi_maps(
    cliques: Iterable[ReadClique], *, mode: str = "exact"
) -> dict[tuple[str, str, str], str]:
    """Return candidate-specific corrected UMIs using expected support."""
    if mode not in {"exact", "1mm_cr"}:
        raise ValueError("mode must be 'exact' or '1mm_cr'")
    support: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for clique in cliques:
        for candidate, probability in clique.probabilities().items():
            support[(clique.feature_id, candidate)][clique.umi] += probability

    result: dict[tuple[str, str, str], str] = {}
    for (feature, candidate), umi_support in sorted(support.items()):
        ordered = sorted(umi_support, key=lambda umi: (-umi_support[umi], umi))
        roots: dict[str, str] = {}
        for index, umi in enumerate(ordered):
            root = umi
            if mode == "1mm_cr":
                eligible = [
                    other for other in ordered[:index]
                    if _hamming_one(umi, other)
                    and umi_support[other] >= 2.0 * umi_support[umi] - 1.0
                ]
                if eligible:
                    parent = min(eligible, key=lambda x: (-umi_support[x], x))
                    root = roots[parent]
            roots[umi] = root
            result[(feature, candidate, umi)] = root
    return result


def weighted_occupancies(
    cliques: Iterable[ReadClique], *, umi_mode: str = "exact"
) -> list[Occupancy]:
    cliques = list(cliques)
    corrections = corrected_umi_maps(cliques, mode=umi_mode)
    groups: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
    for clique in cliques:
        for candidate, probability in clique.probabilities().items():
            corrected = corrections[(clique.feature_id, candidate, clique.umi)]
            groups[(clique.feature_id, corrected, candidate)].append((clique.clique_id, probability))
    rows = []
    for (feature, corrected, candidate), values in sorted(groups.items()):
        occupancy = 1.0 - math.prod(1.0 - probability for _, probability in values)
        rows.append(Occupancy(feature, corrected, candidate, occupancy,
                              tuple(sorted(clique for clique, _ in values))))
    return rows


def collapse_clique_to_unit(clique: ReadClique, size_um: int) -> dict[str, float]:
    collapsed: dict[str, float] = defaultdict(float)
    for candidate, probability in clique.probabilities().items():
        try:
            _, row, col = candidate.rsplit("_", 2)
            unit = bin_unit_id(int(row), int(col), size_um)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"candidate is not an HD unit id: {candidate}") from exc
        collapsed[unit] += probability
    return dict(sorted(collapsed.items()))


def gated_hard_calls(
    cliques: Iterable[ReadClique], *, min_posterior: float, min_margin: float
) -> list[HardCall]:
    if not (0 <= min_posterior <= 1 and 0 <= min_margin <= 1):
        raise ValueError("hard-call gates must lie in [0, 1]")
    calls = []
    for clique in sorted(cliques, key=lambda row: row.clique_id):
        ranked = sorted(clique.probabilities().items(), key=lambda row: (-row[1], row[0]))
        candidate, posterior = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = posterior - second
        accepted = posterior >= min_posterior and margin >= min_margin
        calls.append(HardCall(clique.clique_id, "assigned" if accepted else "deferred",
                              candidate if accepted else None, posterior, margin,
                              "posterior_and_margin" if accepted else "gate_failed"))
    return calls


def strict_counts(cliques: Iterable[ReadClique], *, size_um: int = 2) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for clique in cliques:
        units = collapse_clique_to_unit(clique, size_um)
        if len(units) == 1:
            counts[(clique.feature_id, next(iter(units)))] += 1
    return dict(sorted(counts.items()))


def mass_summary(cliques: Iterable[ReadClique], occupancies: Iterable[Occupancy]) -> dict[str, float]:
    cliques = list(cliques)
    posterior_mass = math.fsum(math.fsum(clique.posterior) for clique in cliques)
    occupancy_mass = math.fsum(row.occupancy for row in occupancies)
    return {
        "read_clique_count": len(cliques),
        "posterior_mass": posterior_mass,
        "occupancy_mass": occupancy_mass,
        "deduplicated_mass": posterior_mass - occupancy_mass,
    }
