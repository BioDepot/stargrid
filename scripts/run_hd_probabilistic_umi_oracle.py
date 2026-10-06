#!/usr/bin/env python3
"""Materialize strict, soft, and gated candidate-preserving UMI products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from star_spatial.hd_probabilistic_umi import (
    SCHEMA,
    CandidateRead,
    build_read_cliques,
    factorized_h0_read_log_prior,
    gated_hard_calls,
    mass_summary,
    strict_counts,
    weighted_occupancies,
)
from star_spatial.hd_assignment_ledger import bin_unit_id
from star_spatial.hd_strict_broad_field import (
    BROAD_RESOLVER_SCHEMA,
    candidate_broad_log_priors_from_counts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-reads", type=Path, required=True)
    parser.add_argument("--h0-read-prior", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--strict-16um-field", type=Path)
    parser.add_argument("--spatial-alpha", type=float, default=1.0)
    parser.add_argument("--spatial-lambda", type=float, default=0.0)
    parser.add_argument("--umi-mode", choices=("exact", "1mm_cr"), default="exact")
    parser.add_argument("--hard-min-posterior", type=float, default=0.95)
    parser.add_argument("--hard-min-margin", type=float, default=0.90)
    args = parser.parse_args()
    if not math.isfinite(args.spatial_lambda) or args.spatial_lambda < 0:
        raise SystemExit("ERROR: --spatial-lambda must be finite and non-negative")
    if not math.isfinite(args.spatial_alpha) or args.spatial_alpha <= 0:
        raise SystemExit("ERROR: --spatial-alpha must be finite and positive")
    if args.spatial_lambda > 0 and args.strict_16um_field is None:
        raise SystemExit("ERROR: --strict-16um-field is required when --spatial-lambda > 0")

    prior_counts = _read_h0_read_prior(args.h0_read_prior)
    broad_counts = (
        _read_strict_broad_field(args.strict_16um_field)
        if args.strict_16um_field is not None else {}
    )
    grouped: dict[
        tuple[str, str, str], dict[str, tuple[float, int, int]]
    ] = defaultdict(dict)
    candidate_rows_seen = 0
    excluded_rows: Counter[str] = Counter()
    excluded_read_ids: set[str] = set()
    with args.candidate_reads.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "read_id", "umi", "candidate_row2", "candidate_col2",
            "log_sequence_likelihood",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(
                f"ERROR: candidate reads missing columns: {', '.join(missing)}"
            )
        feature_column = "feature_id" if "feature_id" in (reader.fieldnames or []) else "gene"
        if feature_column not in (reader.fieldnames or []):
            raise SystemExit("ERROR: candidate reads require feature_id or gene")
        for line_number, row in enumerate(reader, start=2):
            candidate_rows_seen += 1
            missing = []
            if not row[feature_column]:
                missing.append("missing_feature")
            if not row["umi"]:
                missing.append("missing_umi")
            if missing:
                reason = "+".join(missing)
                excluded_rows[reason] += 1
                excluded_read_ids.add(row["read_id"])
                continue
            row2 = int(row["candidate_row2"])
            col2 = int(row["candidate_col2"])
            candidate = bin_unit_id(row2, col2, 2)
            key = (row["read_id"], row[feature_column], row["umi"])
            if candidate in grouped[key]:
                raise SystemExit(
                    f"ERROR: duplicate candidate {candidate} at line {line_number}"
                )
            grouped[key][candidate] = (
                float(row["log_sequence_likelihood"]), row2, col2
            )

    excluded_complete_groups = 0
    for key in list(grouped):
        if key[0] in excluded_read_ids:
            excluded_rows["incomplete_group"] += len(grouped[key])
            del grouped[key]
            excluded_complete_groups += 1

    candidate_coordinates: dict[str, tuple[int, int]] = {}
    reads = []
    for (read_id, feature, umi), candidates in grouped.items():
        likelihoods = {}
        for candidate, (likelihood, row2, col2) in candidates.items():
            previous = candidate_coordinates.setdefault(candidate, (row2, col2))
            if previous != (row2, col2):
                raise SystemExit(f"ERROR: inconsistent coordinate for {candidate}")
            likelihoods[candidate] = likelihood
        reads.append(CandidateRead(read_id, feature, umi, likelihoods))

    log_read_prior = {
        candidate: factorized_h0_read_log_prior(
            col2, row2, prior_counts, alpha=args.prior_alpha
        )
        for candidate, (row2, col2) in candidate_coordinates.items()
    }
    log_broad_prior = candidate_broad_log_priors_from_counts(
        candidate_coordinates, broad_counts, alpha=args.spatial_alpha
    ) if args.strict_16um_field is not None else {}
    cliques = build_read_cliques(
        reads,
        log_read_prior=log_read_prior,
        log_broad_prior=log_broad_prior,
        temperature=args.temperature,
        prior_beta=args.prior_beta,
        spatial_lambda=args.spatial_lambda,
    )
    occupancies = weighted_occupancies(cliques, umi_mode=args.umi_mode)
    hard = gated_hard_calls(cliques, min_posterior=args.hard_min_posterior, min_margin=args.hard_min_margin)
    strict = strict_counts(cliques)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.spatial_lambda > 0:
        _write(
            args.out_dir / "candidate_molecule_ledger.tsv",
            ("read_clique_id", "feature_id", "umi", "read_ids", "candidate",
             "log_sequence_likelihood_sum", "log_exact_h0_read_prior",
             "log_strict_umi_16um_prior", "log_evidence", "posterior"),
            (
                (f.clique_id, f.feature_id, f.umi, ";".join(f.read_ids), candidate,
                 likelihood, read_prior, broad_prior, evidence, posterior)
                for f in cliques
                for candidate, likelihood, read_prior, broad_prior, evidence, posterior
                in zip(
                    f.candidates, f.log_likelihood_sums, f.log_read_priors,
                    f.log_broad_priors, f.log_evidence, f.posterior, strict=True
                )
            ),
        )
    else:
        _write(args.out_dir / "candidate_molecule_ledger.tsv",
               ("read_clique_id", "feature_id", "umi", "read_ids", "candidate",
                "log_sequence_likelihood_sum", "log_exact_h0_read_prior",
                "log_evidence", "posterior"),
               ((f.clique_id, f.feature_id, f.umi, ";".join(f.read_ids), c, ll, prior,
                 evidence, posterior)
                for f in cliques
                for c, ll, prior, evidence, posterior in zip(
                    f.candidates, f.log_likelihood_sums, f.log_read_priors,
                    f.log_evidence, f.posterior, strict=True
                )))
    _write(args.out_dir / "weighted_umi_occupancy.tsv",
           ("feature_id", "corrected_umi", "candidate", "expected_count", "read_clique_ids"),
           ((r.feature_id, r.corrected_umi, r.candidate, r.occupancy, ";".join(r.contributing_cliques)) for r in occupancies))
    _write(args.out_dir / "strict_2um.tsv", ("feature_id", "unit_2um", "integer_count"),
           ((feature, unit, count) for (feature, unit), count in strict.items()))
    _write(args.out_dir / "soft_expected_2um.tsv", ("feature_id", "unit_2um", "expected_count"),
           ((r.feature_id, r.candidate, r.occupancy) for r in occupancies))
    _write(args.out_dir / "gated_hard_2um.tsv",
           ("read_clique_id", "status", "candidate", "posterior", "margin", "reason"),
           ((r.clique_id, r.status, r.candidate or "", r.posterior, r.margin, r.reason) for r in hard))

    schema = BROAD_RESOLVER_SCHEMA if args.spatial_lambda > 0 else SCHEMA
    config = {
        "schema": schema, "candidate_reads": str(args.candidate_reads.resolve()),
        "candidate_reads_sha256": hashlib.sha256(args.candidate_reads.read_bytes()).hexdigest(),
        "h0_read_prior": str(args.h0_read_prior.resolve()),
        "h0_read_prior_sha256": hashlib.sha256(args.h0_read_prior.read_bytes()).hexdigest(),
        "h0_read_prior_units": "raw_exact_h0_reads_including_pcr_amplification",
        "h0_read_prior_application": "once_per_read_clique_shared_latent_coordinate",
        "temperature": args.temperature, "prior_alpha": args.prior_alpha,
        "prior_beta": args.prior_beta, "umi_mode": args.umi_mode,
        "spatial_alpha": args.spatial_alpha,
        "spatial_lambda": args.spatial_lambda,
        "strict_16um_field": (
            str(args.strict_16um_field.resolve())
            if args.strict_16um_field is not None else ""
        ),
        "strict_16um_field_sha256": (
            hashlib.sha256(args.strict_16um_field.read_bytes()).hexdigest()
            if args.strict_16um_field is not None else ""
        ),
        "strict_16um_field_units": "strict_umi_deduplicated_molecules",
        "strict_16um_field_application": "once_per_read_clique_at_16um_parent",
        "hard_min_posterior": args.hard_min_posterior, "hard_min_margin": args.hard_min_margin,
        "priors": {
            "raw_exact_h0_read_frequency": True,
            "strict_umi_broad_16um": args.spatial_lambda > 0,
            **{name: False for name in ("expression", "cell_type", "neighborhood", "graph", "gpu")},
        },
    }
    (args.out_dir / "resolved_runtime_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    summary = mass_summary(cliques, occupancies) | {
        "schema": schema, "reads": len(reads),
        "candidate_rows_seen": candidate_rows_seen,
        "candidate_rows_excluded": sum(excluded_rows.values()),
        "candidate_row_exclusion_reasons": dict(sorted(excluded_rows.items())),
        "read_ids_excluded_non_countable": len(excluded_read_ids),
        "partially_countable_groups_excluded": excluded_complete_groups,
        "raw_exact_h0_reads_per_half": sum(
            count for (half, _), count in prior_counts.items() if half == "BC1"
        ),
        "strict_16um_field_parents": len(broad_counts),
        "strict_16um_field_molecule_mass": sum(broad_counts.values()),
        "strict_counts": sum(strict.values()), "hard_assigned": sum(r.status == "assigned" for r in hard),
        "hard_deferred": sum(r.status == "deferred" for r in hard),
    }
    (args.out_dir / "probabilistic_resolver_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _read_h0_read_prior(path: Path) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"barcode_half", "oligo_index", "exact_h0_read_count"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"ERROR: H0 read prior missing columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            half = row["barcode_half"]
            if half not in {"BC1", "BC2"}:
                raise SystemExit(f"ERROR: invalid barcode_half at line {line_number}: {half}")
            key = (half, int(row["oligo_index"]))
            if key in counts:
                raise SystemExit(f"ERROR: duplicate H0 prior key at line {line_number}: {key}")
            count = int(row["exact_h0_read_count"])
            if count < 0:
                raise SystemExit(f"ERROR: negative H0 read count at line {line_number}")
            counts[key] = count
    if not counts:
        raise SystemExit("ERROR: H0 read prior is empty")
    totals = {
        half: sum(count for (candidate_half, _), count in counts.items() if candidate_half == half)
        for half in ("BC1", "BC2")
    }
    if totals["BC1"] != totals["BC2"]:
        raise SystemExit(
            "ERROR: H0 read-prior totals differ between BC1 and BC2 "
            f"({totals['BC1']} != {totals['BC2']})"
        )
    return counts


def _read_strict_broad_field(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"unit_16um_id", "strict_molecule_count"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(
                f"ERROR: strict 16 um field missing columns: {', '.join(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            unit = row["unit_16um_id"]
            if unit in counts:
                raise SystemExit(
                    f"ERROR: duplicate strict 16 um field unit at line {line_number}: {unit}"
                )
            count = int(row["strict_molecule_count"])
            if count < 0:
                raise SystemExit(
                    f"ERROR: negative strict 16 um count at line {line_number}"
                )
            counts[unit] = count
    if not counts:
        raise SystemExit("ERROR: strict 16 um field is empty")
    return counts


def _write(path: Path, header: tuple[str, ...], rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
