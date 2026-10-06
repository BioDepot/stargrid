#!/usr/bin/env python3
"""Restartable full-slide natural candidate-reference workflow."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--bc1-oligos", type=Path, required=True)
    parser.add_argument("--bc2-oligos", type=Path, required=True)
    parser.add_argument("--h0-read-prior", type=Path, required=True)
    parser.add_argument("--annotation-2um", type=Path, required=True)
    parser.add_argument("--annotation-16um", type=Path, required=True)
    parser.add_argument("--barcode-mappings", type=Path, required=True)
    parser.add_argument(
        "--candidate-binary", type=Path,
        default=Path("build/hd_candidate_preserving_reference"),
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bam-threads", type=int, default=8)
    parser.add_argument("--compute-threads", type=int, default=16)
    parser.add_argument("--sort-memory", default="4G")
    parser.add_argument("--skip-image-evaluation", action="store_true")
    parser.add_argument("--slide", default="H1-VM2JXXK")
    parser.add_argument("--area", default="A1")
    parser.add_argument(
        "--cell-nucleus-target",
        default="published_10x_image_non_transcriptomic",
    )
    parser.add_argument(
        "--annotation-target",
        default="CC0_mixed_modality_marker_expression_contributed",
    )
    return parser.parse_args()


def _run(command, log_path, *, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_candidate_checksums(candidate_root):
    checksum_path = candidate_root / "checksums.sha256"
    if checksum_path.is_file():
        return
    products = [candidate_root / "summary.json"]
    products.extend(sorted((candidate_root / "candidate_shards").glob("part-*.tsv.gz")))
    products.extend(sorted((candidate_root / "sr_read_shards").glob("part-*.tsv.gz")))
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(candidate_root)}\n"
            for path in products
        ),
        encoding="utf-8",
    )


def _resolve_one(index, args, candidate_root, resolver_root, env):
    width = max(3, len(str(args.shards - 1)))
    name = f"part-{index:0{width}d}"
    out = resolver_root / name
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if (out / "checksums.sha256").is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("parameters", {}).get("sr_cb_interpretation") == "unit_or_sequence":
            return out
        repair = [
            sys.executable, "scripts/resolve_hd_full_candidate_shard.py",
            "--candidate-shard", candidate_root / "candidate_shards" / f"{name}.tsv.gz",
            "--sr-read-shard", candidate_root / "sr_read_shards" / f"{name}.tsv.gz",
            "--h0-read-prior", args.h0_read_prior,
            "--bc1-oligos", args.bc1_oligos,
            "--bc2-oligos", args.bc2_oligos,
            "--out-dir", out,
            "--sort-temp", out / "sort_tmp",
            "--sort-memory", args.sort_memory,
            "--sr-only",
        ]
        _run([str(value) for value in repair], out / "run.log", env=env)
        return out
    command = [
        sys.executable, "scripts/resolve_hd_full_candidate_shard.py",
        "--candidate-shard", candidate_root / "candidate_shards" / f"{name}.tsv.gz",
        "--sr-read-shard", candidate_root / "sr_read_shards" / f"{name}.tsv.gz",
        "--h0-read-prior", args.h0_read_prior,
        "--bc1-oligos", args.bc1_oligos,
        "--bc2-oligos", args.bc2_oligos,
        "--out-dir", out,
        "--sort-temp", out / "sort_tmp",
        "--sort-memory", args.sort_memory,
    ]
    _run([str(value) for value in command], out / "run.log", env=env)
    return out


def main():
    args = parse_args()
    if args.shards < 1 or args.workers < 1 or args.bam_threads < 1 or args.compute_threads < 1:
        raise SystemExit("shards, workers, and BAM threads must be positive")
    candidate_sources = (
        Path("native/hd_candidate_preserving_reference.cpp"),
        Path("native/hd_r1_anchored_decode.cpp"),
    )
    for source in candidate_sources:
        if not source.is_file():
            raise SystemExit(f"missing candidate source: {source}")
    if (
        not args.candidate_binary.is_file()
        or any(source.stat().st_mtime_ns > args.candidate_binary.stat().st_mtime_ns for source in candidate_sources)
    ):
        args.candidate_binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            args.cxx, "-std=c++17", "-O3", "-pthread",
            str(candidate_sources[0]), "-lhts", "-lz", "-o", str(args.candidate_binary),
        ], check=True)

    inputs = (
        args.bam, args.bc1_oligos, args.bc2_oligos, args.h0_read_prior,
        args.annotation_2um, args.annotation_16um, args.barcode_mappings,
        args.candidate_binary,
    )
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    candidate_root = args.out_dir / "candidate_preservation"
    if not (candidate_root / "summary.json").is_file():
        _run([
            str(args.candidate_binary), "--bam", str(args.bam),
            "--bc1-oligos", str(args.bc1_oligos),
            "--bc2-oligos", str(args.bc2_oligos),
            "--out-dir", str(candidate_root),
            "--shards", str(args.shards),
            "--bam-threads", str(args.bam_threads),
            "--threads", str(args.compute_threads),
        ], args.out_dir / "logs" / "candidate_preservation.log", env=env)
    _write_candidate_checksums(candidate_root)

    resolver_root = args.out_dir / "resolved_shards"
    resolver_root.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_resolve_one, index, args, candidate_root, resolver_root, env)
            for index in range(args.shards)
        ]
        resolved = [future.result() for future in concurrent.futures.as_completed(futures)]
    resolved.sort()

    aggregate_root = args.out_dir / "matched_audit"
    if not (aggregate_root / "checksums.sha256").is_file():
        if aggregate_root.exists():
            shutil.rmtree(aggregate_root)
        command = [
            sys.executable, "scripts/audit_hd_full_natural_matched_molecules.py",
            "--out-dir", aggregate_root,
            "--workers", str(min(args.workers, 4)),
        ]
        for path in resolved:
            command.extend(("--resolved-shard", path))
        _run([str(value) for value in command], args.out_dir / "logs" / "matched_audit.log", env=env)

    image_root = args.out_dir / "natural_image_oracle"
    if not args.skip_image_evaluation and not (image_root / "checksums.sha256").is_file():
        if image_root.exists():
            shutil.rmtree(image_root)
        field_dir = aggregate_root / "fields"
        method_specs = (
            ("exact_matched_star", "primary_matched", "exact_member_set_natural_shared", "exact_matched_star.tsv.gz"),
            ("exact_matched_space_ranger", "primary_matched", "exact_member_set_natural_shared", "exact_matched_space_ranger.tsv.gz"),
            ("postcollapse_hard", "whole_field_sensitivity", "star_candidate_eligible_1mm_cr", "postcollapse_hard_1mm_cr.tsv.gz"),
            ("postcollapse_soft", "uncertainty_ledger_sensitivity", "star_candidate_eligible_1mm_cr", "postcollapse_soft_1mm_cr.tsv.gz"),
            ("strict", "conservative_sensitivity", "star_unique_candidate_1mm_cr", "strict_1mm_cr.tsv.gz"),
            ("gated_hard", "gated_sensitivity", "star_gated_candidate_1mm_cr", "gated_hard_1mm_cr.tsv.gz"),
            ("space_ranger", "whole_field_sensitivity", "space_ranger_valid_cb_ub", "space_ranger.tsv.gz"),
            ("postcollapse_hard_exact_umi", "umi_sensitivity", "star_candidate_eligible_exact_umi", "postcollapse_hard_exact.tsv.gz"),
            ("postcollapse_soft_exact_umi", "umi_sensitivity", "star_candidate_eligible_exact_umi", "postcollapse_soft_exact.tsv.gz"),
        )
        command = [
            sys.executable, "scripts/evaluate_hd_natural_image_oracle.py",
            "--annotation-2um", args.annotation_2um,
            "--annotation-16um", args.annotation_16um,
            "--barcode-mappings", args.barcode_mappings,
            "--out-dir", image_root,
            "--primary-star", "exact_matched_star",
            "--primary-sr", "exact_matched_space_ranger",
            "--slide", args.slide,
            "--area", args.area,
            "--cell-nucleus-target", args.cell_nucleus_target,
            "--annotation-target", args.annotation_target,
        ]
        for name, role, eligibility, filename in method_specs:
            command.extend(("--method", name, role, eligibility, field_dir / filename, "expected_count"))
        _run([str(value) for value in command], args.out_dir / "logs" / "natural_image_oracle.log", env=env)

    manifest = {
        "schema": "star_spatial.hd.full_natural_candidate_reference_run.v1",
        "slide": args.slide, "area": args.area,
        "inputs": {str(path): _sha256(path) for path in inputs},
        "parameters": {
            "shards": args.shards, "workers": args.workers,
            "bam_threads": args.bam_threads, "compute_threads": args.compute_threads,
            "sort_memory": args.sort_memory,
            "temperature": 1.0, "prior_beta": 1.0,
            "hard_min_posterior": 0.95, "hard_min_margin": 0.90,
            "spatial_lambda": 0.0,
            "cxx": args.cxx,
        },
        "stages": {
            "candidate_preservation": str(candidate_root),
            "resolved_shards": str(resolver_root),
            "matched_audit": str(aggregate_root),
            "natural_image_oracle": None if args.skip_image_evaluation else str(image_root),
        },
        "prohibited_prior_flags": {
            "spatial": False, "image": False, "expression": False,
            "cell_type": False, "neighborhood": False, "graph": False,
            "space_ranger_assignment": False,
        },
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
