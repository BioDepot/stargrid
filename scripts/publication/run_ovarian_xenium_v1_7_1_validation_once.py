#!/usr/bin/env python3
"""Regenerate the hard/strict ovarian adjacent-Xenium validation once.

The accepted Space Ranger matrix is reused by checksum.  For each STAR
multimap-rescue mode this runner emits one 128-um broad comparison, one top-20
companion, and one cancer-panel comparison, with an argv record and GNU time
record for every generator invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "visium_hd_processing.ovarian_xenium_v1_7_1_generation.v1"
RELEASE_COMMIT = "b523c1f58c7f99eb7bc3d3f1b418ac4ab59112a4"
POLICIES = ("strict", "hard")
MODES = ("compatibility", "annotated")
REPO_ROOT = Path(__file__).resolve().parents[2]
NVME_CAMPAIGN_ROOT = Path("<storage>/")
EXPECTED_STORAGE_SOURCE = "/dev/nvme0n1p2"
GENERATOR_SOURCES = {
    "broad": REPO_ROOT / "scripts/publication/compare_visium_to_adjacent_xenium.py",
    "cancer_panel": (
        REPO_ROOT / "scripts/publication/compare_ovarian_panel_in_xenium_cancer_cells.py"
    ),
    "top20": REPO_ROOT / "scripts/publication/summarize_adjacent_xenium_top_genes.py",
}
GENERATORS = {
    name: NVME_CAMPAIGN_ROOT / "software/generators" / path.name
    for name, path in GENERATOR_SOURCES.items()
}

STAGED_INPUTS = NVME_CAMPAIGN_ROOT / "staged/xenium_validation"
SR_H5 = STAGED_INPUTS / "space_ranger/raw_feature_bc_matrix.h5"
POSITIONS = STAGED_INPUTS / "space_ranger/tissue_positions.parquet"
SCALEFACTORS = STAGED_INPUTS / "space_ranger/scalefactors_json.json"
XENIUM_H5 = STAGED_INPUTS / "xenium/cell_feature_matrix.h5"
XENIUM_CELLS = STAGED_INPUTS / "xenium/cells.parquet"
XENIUM_CLUSTERS = STAGED_INPUTS / "xenium/clusters.csv"
XENIUM_DIFFEXP = STAGED_INPUTS / "xenium/differential_expression.csv"
XENIUM_EXPERIMENT = STAGED_INPUTS / "xenium/experiment.xenium"
XENIUM_HE_ALIGNMENT = (
    STAGED_INPUTS / "xenium/Xenium_Prime_Human_Ovary_Cancer_FF_he_imagealignment.csv"
)
REGISTRATION = STAGED_INPUTS / "registration.json"
REFERENCE_PANEL = STAGED_INPUTS / "Xenium_V1_Human_Ovary_Cancer_FF_gene_panel.json"

SHARED_SHA256 = {
    SR_H5: "d8acee1dbdcf76902fa5d74edde2ebc3a818e7eec142d9e547d394899750db38",
    POSITIONS: "d561919911ba63aa814934e625e6501ebbda9158195ea2aebcd707ad020e57f8",
    SCALEFACTORS: "974a0b09505f22eb1f200d2d8ff92a5da65cab453d0603ee0f7fb430ca49e804",
    XENIUM_H5: "aba018799c1c8994f5fc729a3bc11b8cfabd9c19a71f63e0600fcaee839d72d7",
    XENIUM_CELLS: "eba57984c5c46a0cb0c62117850f365369a0a4ddd7817d754ce5d346f192de76",
    XENIUM_CLUSTERS: "eff7ba5de74a21f7e8166770eb7836546df530e1fc91f2edc6f36c714724b0fd",
    XENIUM_DIFFEXP: "ebc9f22c296c6d235bd9c51ecb4b27a65d9fb2f6ea8eda94bb8a7b92cbf7d942",
    XENIUM_EXPERIMENT: "2312b6980efcc2129606915ec4b76b88e28938062f3a99d3ca9077af0a188661",
    XENIUM_HE_ALIGNMENT: "1e9f7e95998eeb1ecb1f45a67187fd8949f30147f325ae92e6aac9b1a0b1ed79",
    REFERENCE_PANEL: "5ff770a348a17bfaa03cf4370612eabd3efa3ad23eedbcc1d437a90d5ac8641d",
    REGISTRATION: "9aa6723c9de75caae62c283d929bc6f94406ecaeff12491674b1293ad556a03a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def storage_source(path: Path) -> str:
    return subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "-T", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_nvme_paths(
    out_dir: Path, star_runs: dict[str, Path]
) -> dict[str, str]:
    paths = {
        "output_parent": out_dir.parent,
        **{f"generator/{name}": path for name, path in GENERATORS.items()},
        **{f"star_run/{mode}": path for mode, path in star_runs.items()},
        **{f"input/{path.name}": path for path in SHARED_SHA256},
        "input/registration.json": REGISTRATION,
    }
    observed = {name: storage_source(path) for name, path in paths.items()}
    drift = {name: source for name, source in observed.items() if source != EXPECTED_STORAGE_SOURCE}
    if drift:
        raise SystemExit(
            f"timed path is not on {EXPECTED_STORAGE_SOURCE}: {json.dumps(drift, sort_keys=True)}"
        )
    return observed


def command_record(name: str, argv: Iterable[object]) -> dict[str, Any]:
    values = [str(value) for value in argv]
    return {
        "schema": "argv.v1",
        "name": name,
        "argv": values,
        "shell_preview": shlex.join(values),
    }


def validate_star_run(path: Path, mode: str) -> dict[str, Any]:
    seal_path = path / "PRIMARY_SEAL.json"
    completion = path / "RECIPE_COMPLETE.json"
    if not seal_path.is_file() or not completion.is_file():
        raise SystemExit(f"STAR run is not accepted: {path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = {
        "status": "sealed",
        "rescue_evidence": mode,
        "star_commit": RELEASE_COMMIT,
        "reported_products": list(POLICIES),
        "space_ranger_executed": False,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise SystemExit(f"STAR seal mismatch for {mode}/{key}: {seal.get(key)!r}")
    mex: dict[str, str] = {}
    for policy in POLICIES:
        directory = path / f"starSpatialGex.out/{policy}/square_016um"
        for name in ("barcodes.tsv", "features.tsv", "matrix.mtx"):
            source = directory / name
            if not source.is_file():
                raise SystemExit(f"missing STAR input: {source}")
            mex[f"{policy}/{name}"] = sha256(source)
    return {"primary_seal_sha256": sha256(seal_path), "mex_sha256": mex}


def common_args() -> list[object]:
    return [
        "--sr-h5", SR_H5,
        "--positions", POSITIONS,
        "--scalefactors", SCALEFACTORS,
        "--xenium-h5", XENIUM_H5,
        "--xenium-cells", XENIUM_CELLS,
        "--xenium-clusters", XENIUM_CLUSTERS,
        "--xenium-diffexp", XENIUM_DIFFEXP,
        "--xenium-experiment", XENIUM_EXPERIMENT,
        "--xenium-he-alignment", XENIUM_HE_ALIGNMENT,
        "--registration", REGISTRATION,
    ]


def policy_args(star_run: Path) -> list[object]:
    values: list[object] = []
    for policy in POLICIES:
        values.extend(
            [
                "--policy-mex",
                f"{policy}={star_run}/starSpatialGex.out/{policy}/square_016um",
            ]
        )
    return values


def commands_for(mode: str, star_run: Path, out_dir: Path) -> list[dict[str, Any]]:
    mode_root = out_dir / mode
    broad = mode_root / "128um_broad"
    cancer = mode_root / "128um_cancer_panel"
    top20 = mode_root / "top20"
    return [
        command_record(
            f"{mode}_128um_broad",
            [
                sys.executable,
                GENERATORS["broad"],
                *common_args(),
                *policy_args(star_run),
                "--reference-panel-json", REFERENCE_PANEL,
                "--reference-panel-source-category", "current",
                "--reference-gene-set-name", "xenium_v1_ovarian_custom",
                "--patch-um", "128",
                "--out-dir", broad,
            ],
        ),
        command_record(
            f"{mode}_top20",
            [
                sys.executable,
                GENERATORS["top20"],
                "--gene-metrics", broad / "gene_spatial_metrics.tsv",
                "--xenium-diffexp", XENIUM_DIFFEXP,
                "--top-n", "20",
                "--selected-panel-json", REFERENCE_PANEL,
                "--selected-source-category", "current",
                "--out", top20 / "top20_genes.tsv",
                "--summary-out", top20 / "top20_summary.tsv",
            ],
        ),
        command_record(
            f"{mode}_128um_cancer_panel",
            [
                sys.executable,
                GENERATORS["cancer_panel"],
                *common_args(),
                *policy_args(star_run),
                "--reference-panel-json", REFERENCE_PANEL,
                "--patch-um", "128",
                "--out-dir", cancer,
            ],
        ),
    ]


def run_timed(record: dict[str, Any], root: Path) -> dict[str, Any]:
    name = str(record["name"])
    cache_started = datetime.now(timezone.utc).isoformat()
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True,
    )
    cache_finished = datetime.now(timezone.utc).isoformat()
    timed = [
        "/usr/bin/time", "-v", "-o", str(root / f"logs/{name}.time.txt"),
        *record["argv"],
    ]
    with (root / f"logs/{name}.stdout.log").open("wb") as stdout, (
        root / f"logs/{name}.stderr.log"
    ).open("wb") as stderr:
        completed = subprocess.run(
            timed, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"generator {name} failed with status {completed.returncode}")
    return {
        "name": name,
        "cache_drop_started_at": cache_started,
        "cache_drop_finished_at": cache_finished,
        "timing_log": f"logs/{name}.time.txt",
        "timing_log_sha256": sha256(root / f"logs/{name}.time.txt"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatibility-run", required=True, type=Path)
    parser.add_argument("--annotated-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and not args.dry_run:
        raise SystemExit(f"refusing to reuse existing output directory: {args.out_dir}")
    star_runs = {
        "compatibility": args.compatibility_run,
        "annotated": args.annotated_run,
    }
    records = {
        mode: commands_for(mode, star_runs[mode], args.out_dir) for mode in MODES
    }
    if args.dry_run:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    if not args.authorize:
        raise SystemExit("execution requires explicit --authorize")

    benchmark_tools = {
        name: shutil.which(name) for name in ("findmnt", "sudo", "sync")
    }
    missing_tools = [name for name, path in benchmark_tools.items() if path is None]
    if missing_tools:
        raise SystemExit(f"missing benchmark tools: {missing_tools}")
    subprocess.run(["sudo", "-n", "true"], check=True)

    shared: dict[str, dict[str, Any]] = {}
    for path, expected in SHARED_SHA256.items():
        if not path.is_file():
            raise SystemExit(f"missing shared input: {path}")
        observed = sha256(path)
        if observed != expected:
            raise SystemExit(f"shared input SHA256 drift: {path}: {observed} != {expected}")
        shared[path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    generator_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): sha256(path)
        for path in GENERATOR_SOURCES.values()
    }
    staged_generator_hashes = {name: sha256(path) for name, path in GENERATORS.items()}
    for name, source in GENERATOR_SOURCES.items():
        if staged_generator_hashes[name] != sha256(source):
            raise SystemExit(f"staged generator SHA256 drift: {name}")
    star_identities = {
        mode: validate_star_run(path, mode) for mode, path in star_runs.items()
    }
    storage = validate_nvme_paths(args.out_dir, star_runs)

    args.out_dir.mkdir()
    (args.out_dir / "commands").mkdir()
    (args.out_dir / "logs").mkdir()
    write_json(
        args.out_dir / "preflight.json",
        {
            "schema": f"{SCHEMA}.preflight",
            "status": "pass",
            "reported_policies": list(POLICIES),
            "space_ranger_executed": False,
            "shared_inputs": shared,
            "generator_sha256": generator_hashes,
            "staged_generator_sha256": staged_generator_hashes,
            "star_inputs": star_identities,
            "storage": {
                "expected_source": EXPECTED_STORAGE_SOURCE,
                "validated_paths": storage,
                "cold_page_cache_before_each_generator": True,
                "tool_paths": benchmark_tools,
                "noninteractive_cache_drop_preflight": "pass",
            },
        },
    )
    write_json(args.out_dir / "commands/generator.argv.json", records)
    (args.out_dir / "commands/rendered_commands.txt").write_text(
        "\n".join(
            record["shell_preview"]
            for mode in MODES
            for record in records[mode]
        )
        + "\n",
        encoding="utf-8",
    )

    started = datetime.now(timezone.utc).isoformat()
    timing_records: list[dict[str, Any]] = []
    try:
        for mode in MODES:
            for record in records[mode]:
                timing_records.append(run_timed(record, args.out_dir))
    except Exception as exc:
        write_json(
            args.out_dir / "FAILED.json",
            {
                "schema": f"{SCHEMA}.failure",
                "error": str(exc),
                "started_at": started,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise

    completion = {
        "schema": f"{SCHEMA}.completion",
        "status": "complete",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "modes": list(MODES),
        "reported_policies": list(POLICIES),
        "space_ranger_executed": False,
        "generator_sha256": generator_hashes,
        "timing_records": timing_records,
    }
    write_json(args.out_dir / "RUN_COMPLETE.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
