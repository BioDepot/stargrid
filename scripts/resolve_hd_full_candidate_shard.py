#!/usr/bin/env python3
"""Resolve one feature-sharded natural HD candidate/read ledger.

The input may be unsorted.  In that case GNU sort is used only as an external
ordering primitive; all Bayesian and UMI operations remain in the transparent
Python reference model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from star_spatial.hd_assignment_ledger import bin_unit_id
from star_spatial.hd_probabilistic_umi import (
    CandidateRead,
    build_read_cliques,
    corrected_umi_maps,
    gated_hard_calls,
)


CLIQUE_FIELDS = (
    "read_clique_id", "feature_id", "raw_umi", "member_read_count",
    "member_read_ids", "tier_profile", "candidate", "log_sequence_likelihood_sum",
    "log_exact_h0_read_prior", "log_evidence", "posterior",
)
MOLECULE_FIELDS = (
    "umi_mode", "product", "molecule_id", "feature_id", "corrected_umi",
    "unit_2um", "tier", "tier_profile", "member_read_count", "member_read_ids",
    "preassignment_candidate_count", "read_clique_ids",
)
SOFT_FIELDS = (
    "umi_mode", "feature_id", "corrected_umi", "unit_2um", "expected_count",
    "read_clique_ids",
)
GATE_FIELDS = (
    "read_clique_id", "status", "unit_2um", "posterior", "margin", "reason",
)
SR_FIELDS = (
    "molecule_id", "feature_id", "corrected_umi", "corrected_cb", "unit_2um",
    "tier", "member_read_count", "candidate_eligible_reads", "member_read_ids",
)
UNIT_2UM = re.compile(r"^s_002um_(\d+)_(\d+)(?:-1)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-shard", type=Path, required=True)
    parser.add_argument("--sr-read-shard", type=Path)
    parser.add_argument(
        "--no-compatibility-ledger", action="store_true",
        help="Resolve STAR/raw candidates without reading or constructing a Space Ranger ledger.",
    )
    parser.add_argument("--h0-read-prior", type=Path, required=True)
    parser.add_argument("--bc1-oligos", type=Path, required=True)
    parser.add_argument("--bc2-oligos", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sort-temp", type=Path)
    parser.add_argument("--inputs-sorted", action="store_true")
    parser.add_argument(
        "--sr-only", action="store_true",
        help="rebuild only the Space Ranger molecule ledger from the sorted read ledger",
    )
    parser.add_argument(
        "--sr-read-namespace", default="",
        help="Prefix added to Space Ranger member read IDs before molecule matching.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--hard-min-posterior", type=float, default=0.95)
    parser.add_argument("--hard-min-margin", type=float, default=0.90)
    parser.add_argument("--sort-memory", default="4G")
    return parser.parse_args()


class GzipTable:
    def __init__(self, path: Path, fields: tuple[str, ...]):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = path.open("wb")
        compressed = gzip.GzipFile(
            filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0,
        )
        self.handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        self.writer.writeheader()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)

    def close(self) -> None:
        self.handle.close()


def _sort_gzip_body(source: Path, destination: Path, keys: list[str], memory: str, temp: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    key_args = " ".join(shlex.quote(value) for value in keys)
    command = (
        f"gzip -cd -- {shlex.quote(str(source))} | "
        "{ IFS= read -r _header; "
        f"LC_ALL=C sort -T {shlex.quote(str(temp))} -S {shlex.quote(memory)} "
        f"-t $'\\t' {key_args}; }} | gzip -1 > {shlex.quote(str(destination))}"
    )
    subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True)


def _open_rows(path: Path, fields: tuple[str, ...]):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=fields, delimiter="\t")
        yield from reader


def _read_lines(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"invalid oligo table: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_h0_counts(path: Path, bc1: list[str], bc2: list[str]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            half = row["barcode_half"]
            index = int(row["oligo_index"])
            expected = bc1 if half == "BC1" else bc2 if half == "BC2" else None
            if expected is None or not 0 <= index < len(expected):
                raise ValueError(f"invalid prior row: {row}")
            if row["oligo_sequence"] != expected[index]:
                raise ValueError(f"prior/oligo mismatch at {half}:{index}")
            counts[(half, index)] = int(row["exact_h0_read_count"])
    if len(counts) != len(bc1) + len(bc2):
        raise ValueError("H0 prior is incomplete")
    return counts


def _digest(prefix: str, parts) -> str:
    return prefix + "_" + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def _tier_number(value: str) -> int:
    return int(value[1:]) if value.startswith("H") else int(value)


def _prior_for_candidates(reads: list[CandidateRead], counts) -> dict[str, float]:
    result = {}
    for candidate in sorted({value for read in reads for value in read.log_likelihoods}):
        _, row, col = candidate.rsplit("_", 2)
        result[candidate] = (
            math.log(counts[("BC1", int(col))] + 1.0)
            + math.log(counts[("BC2", int(row))] + 1.0)
        )
    return result


def _clique_tier_profile(clique, tiers: dict[str, str]) -> str:
    counts = Counter(tiers[read] for read in clique.read_ids)
    return ";".join(f"{tier}:{counts[tier]}" for tier in sorted(counts, key=_tier_number))


def _molecule_rows(cliques, corrections, assignments, mode, product, tiers):
    groups = defaultdict(list)
    for clique, candidate in assignments:
        corrected = corrections[(clique.feature_id, candidate, clique.umi)]
        groups[(clique.feature_id, corrected, candidate)].append(clique)
    for (feature, corrected, candidate), members in sorted(groups.items()):
        reads = tuple(sorted({read for clique in members for read in clique.read_ids}))
        clique_ids = tuple(sorted(clique.clique_id for clique in members))
        tier_counts = Counter(tiers[read] for read in reads)
        tier = max(tier_counts, key=_tier_number)
        yield {
            "umi_mode": mode,
            "product": product,
            "molecule_id": _digest("mol", (mode, product, feature, corrected, candidate, *reads)),
            "feature_id": feature,
            "corrected_umi": corrected,
            "unit_2um": candidate,
            "tier": tier,
            "tier_profile": ";".join(
                f"{name}:{tier_counts[name]}" for name in sorted(tier_counts, key=_tier_number)
            ),
            "member_read_count": len(reads),
            "member_read_ids": ";".join(reads),
            "preassignment_candidate_count": max(len(clique.candidates) for clique in members),
            "read_clique_ids": ";".join(clique_ids),
        }


def _write_occupancies(cliques, corrections, mode, output, summary):
    groups = defaultdict(list)
    for clique in cliques:
        for candidate, probability in zip(clique.candidates, clique.posterior, strict=True):
            corrected = corrections[(clique.feature_id, candidate, clique.umi)]
            groups[(clique.feature_id, corrected, candidate)].append(
                (clique.clique_id, probability)
            )
    for (feature, corrected, candidate), values in sorted(groups.items()):
        occupancy = 1.0 - math.prod(1.0 - probability for _, probability in values)
        output.write({
            "umi_mode": mode,
            "feature_id": feature,
            "corrected_umi": corrected,
            "unit_2um": candidate,
            "expected_count": repr(occupancy),
            "read_clique_ids": ";".join(sorted(clique for clique, _ in values)),
        })
        summary[f"soft_rows_{mode}"] += 1
        summary[f"soft_mass_{mode}"] += occupancy


def _process_feature(reads, tiers, counts, args, clique_out, molecule_out, soft_out, gate_out, summary):
    priors = _prior_for_candidates(reads, counts)
    cliques = build_read_cliques(
        reads, log_read_prior=priors, temperature=args.temperature,
        prior_beta=args.prior_beta, spatial_lambda=0.0,
    )
    summary["candidate_reads"] += len(reads)
    summary["read_cliques"] += len(cliques)
    for clique in cliques:
        profile = _clique_tier_profile(clique, tiers)
        members = ";".join(clique.read_ids)
        for candidate, likelihood, prior, evidence, posterior in zip(
            clique.candidates, clique.log_likelihood_sums, clique.log_read_priors,
            clique.log_evidence, clique.posterior, strict=True,
        ):
            clique_out.write({
                "read_clique_id": clique.clique_id,
                "feature_id": clique.feature_id,
                "raw_umi": clique.umi,
                "member_read_count": len(clique.read_ids),
                "member_read_ids": members,
                "tier_profile": profile,
                "candidate": candidate,
                "log_sequence_likelihood_sum": repr(likelihood),
                "log_exact_h0_read_prior": repr(prior),
                "log_evidence": repr(evidence),
                "posterior": repr(posterior),
            })
    calls = gated_hard_calls(
        cliques, min_posterior=args.hard_min_posterior,
        min_margin=args.hard_min_margin,
    )
    call_by_id = {call.clique_id: call for call in calls}
    for call in calls:
        gate_out.write({
            "read_clique_id": call.clique_id,
            "status": call.status,
            "unit_2um": call.candidate or "",
            "posterior": repr(call.posterior),
            "margin": repr(call.margin),
            "reason": call.reason,
        })
        summary[f"gated_{call.status}"] += 1

    for mode in ("1mm_cr", "exact"):
        corrections = corrected_umi_maps(cliques, mode=mode)
        _write_occupancies(cliques, corrections, mode, soft_out, summary)
        strict = [(clique, clique.candidates[0]) for clique in cliques if len(clique.candidates) == 1]
        hard = [
            (clique, min(zip(clique.candidates, clique.posterior), key=lambda row: (-row[1], row[0]))[0])
            for clique in cliques
        ]
        gated = [
            (clique, call_by_id[clique.clique_id].candidate)
            for clique in cliques if call_by_id[clique.clique_id].candidate is not None
        ]
        for product, assignments in (
            ("strict", strict), ("postcollapse_hard", hard), ("gated_hard", gated),
        ):
            count = 0
            for row in _molecule_rows(cliques, corrections, assignments, mode, product, tiers):
                molecule_out.write(row)
                count += 1
            summary[f"molecules_{mode}_{product}"] += count


def _process_candidate_file(path, counts, args, outputs, summary):
    fields = (
        "read_id", "feature_id", "raw_umi", "sr_corrected_umi", "sr_cb", "min_tier",
        "candidate_count", "row2", "col2", "bc1_edit", "bc2_edit", "bc1_obs_len",
        "bc2_obs_len", "tier_profile", "log_sequence_likelihood",
    )
    current_feature = None
    current_read = None
    current_umi = None
    current_likelihoods = {}
    current_tier = None
    feature_reads = []
    tiers = {}

    def flush_read():
        nonlocal current_read, current_umi, current_likelihoods, current_tier
        if current_read is None:
            return
        feature_reads.append(CandidateRead(
            current_read, current_feature, current_umi, current_likelihoods,
        ))
        tiers[current_read] = current_tier
        current_read = None
        current_umi = None
        current_likelihoods = {}
        current_tier = None

    def flush_feature():
        nonlocal feature_reads, tiers
        flush_read()
        if feature_reads:
            _process_feature(feature_reads, tiers, counts, args, *outputs, summary)
        feature_reads = []
        tiers = {}

    for row in _open_rows(path, fields):
        feature = row["feature_id"]
        if current_feature is not None and feature != current_feature:
            flush_feature()
        current_feature = feature
        read_id = row["read_id"]
        if current_read is not None and read_id != current_read:
            flush_read()
        if current_read is None:
            current_read = read_id
            current_umi = row["raw_umi"]
            current_tier = f"H{row['min_tier']}"
        candidate = bin_unit_id(int(row["row2"]), int(row["col2"]), 2)
        if candidate in current_likelihoods:
            raise ValueError(f"duplicate candidate for read {read_id}: {candidate}")
        current_likelihoods[candidate] = float(row["log_sequence_likelihood"])
    flush_feature()


def _cb_lookup(bc1: list[str], bc2: list[str]):
    # Do not materialize all 11.2M concatenated barcodes.  The returned
    # closure tries the two valid BC1 lengths against two small exact maps.
    bc1_index = {value: index for index, value in enumerate(bc1)}
    bc2_index = {value: index for index, value in enumerate(bc2)}

    def resolve(cb: str):
        # Space Ranger 4 BAMs may carry the corrected 2 um unit directly in
        # CB (for example, s_002um_00123_00456-1) and omit sb.  Older fixtures
        # may instead carry the corrected concatenated oligo.  Accept both
        # representations, with the explicit unit form taking precedence.
        unit_match = UNIT_2UM.fullmatch(cb)
        if unit_match is not None:
            return bin_unit_id(int(unit_match.group(1)), int(unit_match.group(2)), 2)
        for split in (15, 16):
            if split >= len(cb):
                continue
            try:
                col = bc1_index[cb[:split]]
                row = bc2_index[cb[split:]]
            except KeyError:
                continue
            return bin_unit_id(row, col, 2)
        return None

    return resolve


def _write_manifest_and_checksums(args, summary):
    products = [
        args.out_dir / "candidate_cliques.tsv.gz",
        args.out_dir / "star_molecules.tsv.gz",
        args.out_dir / "soft_expected.tsv.gz",
        args.out_dir / "gated_calls.tsv.gz",
        args.out_dir / "space_ranger_molecules.tsv.gz",
    ]
    manifest = {
        "schema": "star_spatial.hd.full_candidate_shard_resolution.v1",
        "candidate_shard": str(args.candidate_shard),
        "sr_read_shard": None if args.sr_read_shard is None else str(args.sr_read_shard),
        "parameters": {
            "temperature": args.temperature,
            "prior_beta": args.prior_beta,
            "spatial_lambda": 0.0,
            "hard_min_posterior": args.hard_min_posterior,
            "hard_min_margin": args.hard_min_margin,
            "umi_modes": ["1mm_cr", "exact"],
            "sr_cb_interpretation": "unit_or_sequence",
            "sr_read_namespace": args.sr_read_namespace,
            "compatibility_ledger_enabled": not args.no_compatibility_ledger,
        },
        "prohibited_prior_flags": {
            "spatial": False, "image": False, "expression": False,
            "cell_type": False, "neighborhood": False, "graph": False,
            "space_ranger_cb": False, "space_ranger_ub": False,
        },
        "counts": dict(sorted(summary.items())),
        "declared_outputs": {
            path.name: {"sha256": _sha256(path)} for path in products
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in products),
        encoding="utf-8",
    )


def _repair_sr_only(args, bc1, bc2, temporary_root):
    sr_path = args.out_dir / "sr_reads.sorted.tsv.gz"
    if not sr_path.is_file():
        _sort_gzip_body(
            args.sr_read_shard, sr_path,
            ["-k2,2", "-k4,4", "-k5,5", "-k1,1"],
            args.sort_memory, temporary_root,
        )
    summary_path = args.out_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"cannot repair SR ledger without existing summary: {summary_path}")
    manifest = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = defaultdict(float, {
        key: value for key, value in manifest.get("counts", {}).items()
        if not key.startswith("sr_") and key != "compatibility_ledger_disabled"
    })
    repaired_path = args.out_dir / "space_ranger_molecules.tsv.gz.repair"
    sr_out = GzipTable(repaired_path, SR_FIELDS)
    try:
        _process_sr_file(
            sr_path, bc1, bc2, sr_out, summary,
            read_namespace=args.sr_read_namespace,
        )
    finally:
        sr_out.close()
    os.replace(repaired_path, args.out_dir / "space_ranger_molecules.tsv.gz")
    _write_manifest_and_checksums(args, summary)


def _normalize_unit_2um(value: str):
    match = UNIT_2UM.fullmatch(value)
    if match is None:
        return None
    return bin_unit_id(int(match.group(1)), int(match.group(2)), 2)


def _process_sr_file(path, bc1, bc2, output, summary, *, read_namespace=""):
    fields = (
        "read_id", "feature_id", "raw_umi", "sr_corrected_umi", "sr_cb", "sr_unit_2um",
        "star_candidate_count", "star_min_tier",
    )
    resolve_cb = _cb_lookup(bc1, bc2)
    current_key = None
    reads = []
    candidate_eligible = 0
    tiers = []

    units = []

    def emit(key, member_reads, eligible, known_tiers, observed_units):
        if key is None:
            return
        feature, umi, cb = key
        normalized_units = {_normalize_unit_2um(value) for value in observed_units if value}
        if None in normalized_units:
            summary["sr_malformed_sb"] += 1
            normalized_units.discard(None)
        if len(normalized_units) > 1:
            raise ValueError(f"Space Ranger molecule has inconsistent sb tags: {key}")
        unit = next(iter(normalized_units), None)
        decoded_unit = resolve_cb(cb)
        if unit is None:
            unit = decoded_unit
            summary["sr_unit_fallback_from_cb"] += 1
        elif decoded_unit is not None and decoded_unit != unit:
            summary["sr_sb_cb_disagreement"] += 1
        if unit is None:
            summary["sr_rejected_cb"] += len(member_reads)
            return
        members = tuple(sorted(set(member_reads)))
        tier = f"H{max(known_tiers)}" if known_tiers else "unavailable"
        output.write({
            "molecule_id": _digest("sr", (feature, umi, cb, *members)),
            "feature_id": feature,
            "corrected_umi": umi,
            "corrected_cb": cb,
            "unit_2um": unit,
            "tier": tier,
            "member_read_count": len(members),
            "candidate_eligible_reads": eligible,
            "member_read_ids": ";".join(members),
        })
        summary["sr_molecules"] += 1
        summary["sr_member_reads"] += len(members)

    for row in _open_rows(path, fields):
        if not row["sr_cb"]:
            summary["sr_missing_cb_reads"] += 1
            continue
        key = (row["feature_id"], row["sr_corrected_umi"], row["sr_cb"])
        if current_key is not None and key != current_key:
            emit(current_key, reads, candidate_eligible, tiers, units)
            reads, tiers, units, candidate_eligible = [], [], [], 0
        current_key = key
        reads.append(read_namespace + row["read_id"])
        units.append(row["sr_unit_2um"])
        if int(row["star_candidate_count"]) > 0:
            candidate_eligible += 1
            tiers.append(int(row["star_min_tier"]))
    emit(current_key, reads, candidate_eligible, tiers, units)


def main() -> int:
    args = parse_args()
    if args.no_compatibility_ledger and args.sr_only:
        raise SystemExit("--no-compatibility-ledger and --sr-only are incompatible")
    if not args.no_compatibility_ledger and args.sr_read_shard is None:
        raise SystemExit("--sr-read-shard is required unless --no-compatibility-ledger is set")
    paths = [args.candidate_shard, args.h0_read_prior, args.bc1_oligos, args.bc2_oligos]
    if args.sr_read_shard is not None:
        paths.append(args.sr_read_shard)
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.temperature <= 0 or args.prior_beta < 0:
        raise SystemExit("invalid posterior parameter")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bc1, bc2 = _read_lines(args.bc1_oligos), _read_lines(args.bc2_oligos)
    temporary_root = args.sort_temp or (args.out_dir / "sort_tmp")
    if args.sr_only:
        _repair_sr_only(args, bc1, bc2, temporary_root)
        return 0

    counts = _load_h0_counts(args.h0_read_prior, bc1, bc2)
    if args.inputs_sorted:
        candidate_path, sr_path = args.candidate_shard, args.sr_read_shard
    else:
        candidate_path = args.out_dir / "candidate.sorted.tsv.gz"
        sr_path = None if args.no_compatibility_ledger else args.out_dir / "sr_reads.sorted.tsv.gz"
        _sort_gzip_body(
            args.candidate_shard, candidate_path,
            ["-k2,2", "-k3,3", "-k1,1", "-k8,8n", "-k9,9n"],
            args.sort_memory, temporary_root,
        )
        if not args.no_compatibility_ledger:
            _sort_gzip_body(
                args.sr_read_shard, sr_path,
                ["-k2,2", "-k4,4", "-k5,5", "-k1,1"],
                args.sort_memory, temporary_root,
            )

    clique_out = GzipTable(args.out_dir / "candidate_cliques.tsv.gz", CLIQUE_FIELDS)
    molecule_out = GzipTable(args.out_dir / "star_molecules.tsv.gz", MOLECULE_FIELDS)
    soft_out = GzipTable(args.out_dir / "soft_expected.tsv.gz", SOFT_FIELDS)
    gate_out = GzipTable(args.out_dir / "gated_calls.tsv.gz", GATE_FIELDS)
    sr_out = GzipTable(args.out_dir / "space_ranger_molecules.tsv.gz", SR_FIELDS)
    summary = defaultdict(float)
    try:
        _process_candidate_file(
            candidate_path, counts, args,
            (clique_out, molecule_out, soft_out, gate_out), summary,
        )
        if args.no_compatibility_ledger:
            summary["compatibility_ledger_disabled"] = 1
        else:
            _process_sr_file(
                sr_path, bc1, bc2, sr_out, summary,
                read_namespace=args.sr_read_namespace,
            )
    finally:
        for table in (clique_out, molecule_out, soft_out, gate_out, sr_out):
            table.close()

    _write_manifest_and_checksums(args, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
