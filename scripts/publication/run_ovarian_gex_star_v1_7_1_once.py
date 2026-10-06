#!/usr/bin/env python3
"""Run one release-pinned ovarian Visium HD 3' STAR Suite v1.7.1 arm.

This runner is intentionally narrow: full H1-YQJBZ7X/D1, one evidence mode,
hard and strict assignment products, and no Space Ranger or publication analysis.
It refuses an existing output directory and records the timed run as an instance
separately from the portable primary output seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "visium_hd_processing.ovarian_gex_star_v1_7_1.v1"
SLIDE_AREA = "H1-YQJBZ7X/D1"
RELEASE_TAG = "v1.7.1"
RELEASE_COMMIT = "b523c1f58c7f99eb7bc3d3f1b418ac4ab59112a4"
RELEASE_TREE = "a0edb549b7171feec856b2446240edcfa83bcf00"
RELEASE_ARCHIVE = Path(
    "<local>/v1.7.1/download/"
    "STAR-suite-v1.7.1-linux-amd64-glibc234.tar.gz"
)
RELEASE_ARCHIVE_SHA256 = (
    "c0d21cf2f1934b0ac2de5bbddfaad9072a215a0fa7f490d8ad670d895fb107d0"
)
RELEASE_CHECKSUMS = Path(
    "<local>/v1.7.1/download/SHA256SUMS"
)
RELEASE_CHECKSUMS_SHA256 = (
    "3202da11bad408fc58a4b0453709191692c77619586878b512cabcaa854cf2ac"
)
NVME_CAMPAIGN_ROOT = Path("<storage>/")
EXPECTED_STORAGE_SOURCE = "/dev/nvme0n1p2"
STAR = NVME_CAMPAIGN_ROOT / "software/STAR"
STAR_SHA256 = "066e5dcda9bc00a7b5134aa11c91d47705ae8e029b4bf64c0e2ce77cdd4578a2"
SOURCE_TREE = Path("<local>/")

REFERENCE_ID = "refdata-gex-GRCh38-2024-A"
REFERENCE = Path(
    "<storage>/staged/references/gex_2024a_star"
)
BARCODE_CONTRACT = Path(
    "<storage>/runs/cleanroom_hd_mouse_brain/barcode_contract"
)
BC1_OLIGOS = Path(
    "<storage>/runs/cleanroom_hd_mouse_brain/slide_oligos/"
    "bc1_full_oligos.txt"
)
BC2_OLIGOS = Path(
    "<storage>/runs/cleanroom_hd_mouse_brain/slide_oligos/"
    "bc2_full_oligos.txt"
)
FASTQ_ROOT = Path(
    "<storage>/staged/inputs/ovarian/fastqs"
)
FASTQ_PREFIX = "Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth_S1"
EXPECTED_READS = 474_131_092
EXPECTED_CANDIDATES = 529_580_381
PRODUCTS = ("strict", "hard")
SCALES = ("square_002um", "square_008um", "square_016um")

REFERENCE_SHA256 = {
    "genomeParameters.txt": "c0a34434da7a29257036109a7593968b9b8c7faf3538f8c877c43de2c6f7663c",
    "chrName.txt": "12d83750559b9ae7d273b4b7ce6077e6f24c43eb65672d1fbcfa2e1593fd3ee8",
    "chrLength.txt": "b297c2398f93cd3d4c8aa1d4e10d6148c9550a1c2175139c29f9aab505422a17",
    "geneInfo.tab": "40f6ac3112cdb0c36ccef1543eb40c075904f02efc2aae33644f1d90424a6e08",
}
SPATIAL_SHA256 = {
    BARCODE_CONTRACT / "summary.json": (
        "56b9d6f2f6cfb2bed3ddefa63eb103da221a8e6a925ffa9baa3ba57da4cf5166"
    ),
    BC1_OLIGOS: "24dfc754a706239d990ef8037c78f39bfe9c745b4ad424aa64dfc2317c4857f8",
    BC2_OLIGOS: "2971bbbba0abdd7457b51a5646845df6ec2e3d620248dab17182fa69c660a1bb",
}
FASTQ_SHA256 = {
    f"{FASTQ_PREFIX}_L001_R1_001.fastq.gz": (
        "232f76857f80d4c076a7938247ef0845a3f6ec93f91f123509c8a0477ea675cb"
    ),
    f"{FASTQ_PREFIX}_L002_R1_001.fastq.gz": (
        "187eb6e9bb836f6e1dbf642cb8127936ec117c453f553e79817290ae0ed631e9"
    ),
    f"{FASTQ_PREFIX}_L003_R1_001.fastq.gz": (
        "bd0fc6f0cc1ca856d17fca279cb292d06f19bc4efb9eb8ad44c2900edd138398"
    ),
    f"{FASTQ_PREFIX}_L004_R1_001.fastq.gz": (
        "034b0ee96be6cddbe41a43d27dad74652280bbd00d9c8152b570ae35b71bd6b7"
    ),
    f"{FASTQ_PREFIX}_L001_R2_001.fastq.gz": (
        "60fd5670546363aa5fb630dbcbaef05444a4e8f0d182bbe2f193e424a1aa8181"
    ),
    f"{FASTQ_PREFIX}_L002_R2_001.fastq.gz": (
        "7aeddaa846b0bec91c1b5d7718de404d96e650ad2f2e75d1bc6b25af78494735"
    ),
    f"{FASTQ_PREFIX}_L003_R2_001.fastq.gz": (
        "34c1f3b4d91e45abeab876cc67218dc3c665cc310106dadf61e10267e7d8377e"
    ),
    f"{FASTQ_PREFIX}_L004_R2_001.fastq.gz": (
        "7045f3aa194b2defcf046416d76e152bb16ac102a52f15969c7f073b2b200cd0"
    ),
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


def command_record(argv: Iterable[object]) -> dict[str, Any]:
    values = [str(value) for value in argv]
    return {
        "schema": "argv.v1",
        "name": "star_native_spatial_gex",
        "argv": values,
        "shell_preview": shlex.join(values),
    }


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_TREE), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def fastqs(mate: int) -> tuple[Path, ...]:
    return tuple(
        FASTQ_ROOT / f"{FASTQ_PREFIX}_L00{lane}_R{mate}_001.fastq.gz"
        for lane in range(1, 5)
    )


def build_command(args: argparse.Namespace) -> dict[str, Any]:
    argv = [
        STAR,
        "--runThreadN", args.threads,
        "--genomeDir", REFERENCE,
        "--readFilesIn",
        ",".join(str(path) for path in fastqs(2)),
        ",".join(str(path) for path in fastqs(1)),
        "--readFilesCommand", "zcat",
        "--outFileNamePrefix", args.out_dir / "star",
        "--clipAdapterType", "CellRanger4",
        "--outFilterScoreMin", "30",
        "--soloType", "None",
        "--soloFeatures", "GeneFull",
        "--soloCrGexFeature", "GeneFull",
        "--soloCrMultimapRescue", "yes",
        "--soloCrMultimapRescueEvidence", args.evidence,
        "--soloCrMultimapRescueIntronic", "auto",
        "--soloUMIdedup", "1MM_CR",
        "--soloUMIfiltering", "MultiGeneUMI_CR",
        "--soloMultiMappers", "Unique",
        "--soloStrand", "Forward",
        "--soloCellFilter", "None",
        "--outSAMtype", "None",
        "--soloSpatialGexIntegrated", "yes",
        "--soloSpatialBarcodeContract", BARCODE_CONTRACT,
        "--soloSpatialBc1Oligos", BC1_OLIGOS,
        "--soloSpatialBc2Oligos", BC2_OLIGOS,
        "--soloSpatialAssignmentProducts", ",".join(PRODUCTS),
        "--soloSpatialBinSizes", "2,8,16",
        "--soloSpatialExpectedReads", EXPECTED_READS,
        "--soloSpatialExpectedCandidates", EXPECTED_CANDIDATES,
        "--soloSpatialMemoryFraction", args.memory_fraction,
        "--soloSpatialOverflowPolicy", "Spill",
    ]
    return command_record(argv)


def validate_digest(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    observed = sha256(path)
    if observed != expected:
        raise SystemExit(f"SHA256 drift for {path}: {observed} != {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def storage_source(path: Path) -> str:
    """Return the backing block device reported for an existing path."""
    return subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "-T", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_nvme_paths(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "output_parent": args.out_dir.parent,
        "star_binary": STAR,
        "reference": REFERENCE,
        "barcode_contract": BARCODE_CONTRACT,
        "bc1_oligos": BC1_OLIGOS,
        "bc2_oligos": BC2_OLIGOS,
    }
    paths.update({f"fastq/{path.name}": path for path in (*fastqs(1), *fastqs(2))})
    observed = {name: storage_source(path) for name, path in paths.items()}
    drift = {
        name: source
        for name, source in observed.items()
        if source != EXPECTED_STORAGE_SOURCE
    }
    if drift:
        raise SystemExit(
            f"timed path is not on {EXPECTED_STORAGE_SOURCE}: {json.dumps(drift, sort_keys=True)}"
        )
    device = subprocess.run(
        ["lsblk", "-ndo", "NAME,ROTA,TRAN,MODEL", "/dev/nvme0n1"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if " 0 " not in f" {device} " or "nvme" not in device.lower():
        raise SystemExit(f"storage device is not a non-rotational NVMe device: {device}")
    return {
        "expected_source": EXPECTED_STORAGE_SOURCE,
        "validated_paths": observed,
        "device": device,
    }


def release_identity() -> dict[str, Any]:
    archive = validate_digest(RELEASE_ARCHIVE, RELEASE_ARCHIVE_SHA256)
    checksums = validate_digest(RELEASE_CHECKSUMS, RELEASE_CHECKSUMS_SHA256)
    binary = validate_digest(STAR, STAR_SHA256)
    commit = git_value("rev-parse", "HEAD")
    tree = git_value("rev-parse", "HEAD^{tree}")
    tracked = git_value("status", "--porcelain", "--untracked-files=no")
    if commit != RELEASE_COMMIT or tree != RELEASE_TREE or tracked:
        raise SystemExit("v1.7.1 source worktree identity or cleanliness drift")
    version = subprocess.run(
        [str(STAR), "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    revision = subprocess.run(
        [str(STAR), "--source-revision"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if version != "1.7.1" or revision != RELEASE_COMMIT:
        raise SystemExit(f"unexpected STAR identity: version={version}, revision={revision}")
    return {
        "tag": RELEASE_TAG,
        "commit": commit,
        "tree": tree,
        "archive": archive,
        "published_checksums": checksums,
        "binary": binary,
        "version": version,
        "source_revision": revision,
        "release_url": "https://github.com/morphic-bio/STAR-suite/releases/tag/v1.7.1",
    }


def preflight(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    release = release_identity()
    storage = validate_nvme_paths(args)
    benchmark_tools = {
        name: shutil.which(name)
        for name in ("findmnt", "iostat", "vmstat", "sudo", "sync")
    }
    missing_tools = [name for name, path in benchmark_tools.items() if path is None]
    if missing_tools:
        raise SystemExit(f"missing benchmark tools: {missing_tools}")
    subprocess.run(["sudo", "-n", "true"], check=True)
    fixed_inputs: dict[str, Any] = {}
    for name, expected in REFERENCE_SHA256.items():
        fixed_inputs[f"reference/{name}"] = validate_digest(REFERENCE / name, expected)
    for path, expected in SPATIAL_SHA256.items():
        fixed_inputs[f"spatial/{path.name}"] = validate_digest(path, expected)
    input_fastqs = [
        validate_digest(path, FASTQ_SHA256[path.name])
        for path in (*fastqs(1), *fastqs(2))
    ]
    free_gib = shutil.disk_usage(args.out_dir.parent).free / (1024**3)
    if free_gib < args.minimum_free_gib:
        raise SystemExit(
            f"only {free_gib:.2f} GiB free; require {args.minimum_free_gib:.2f} GiB"
        )
    rendered = record["shell_preview"]
    forbidden = (
        "molecule_first_resolver",
        "molecule_first_materialize",
        "sort_hd_candidate_feature_ledger.py",
    )
    if "--soloSpatialGexIntegrated yes" not in rendered or any(
        token in rendered for token in forbidden
    ):
        raise SystemExit("native integrated route guard failed")
    return {
        "schema": f"{SCHEMA}.preflight",
        "status": "pass",
        "dataset": "ovarian_gex",
        "slide_area": SLIDE_AREA,
        "reference": REFERENCE_ID,
        "release": release,
        "scientific_contract": {
            "feature_scope": "GeneFull_including_intronic",
            "rescue_evidence": args.evidence,
            "rescue_intronic": "auto",
            "umi_mode": "1MM_CR",
            "products": list(PRODUCTS),
            "bin_sizes_um": [2, 8, 16],
            "bam": False,
            "space_ranger_executed": False,
        },
        "capacity": {
            "expected_reads": EXPECTED_READS,
            "expected_candidates": EXPECTED_CANDIDATES,
            "memory_fraction": args.memory_fraction,
            "overflow_policy": "Spill",
        },
        "fixed_inputs": fixed_inputs,
        "fastqs": input_fastqs,
        "free_gib_before_run": free_gib,
        "storage": storage,
        "benchmark_protocol": {
            "replicates": 1,
            "serialized": True,
            "cold_page_cache": True,
            "monitoring": ["iostat -dx 5", "vmstat 5"],
            "tool_paths": benchmark_tools,
            "noninteractive_cache_drop_preflight": "pass",
        },
    }


def drop_page_cache() -> dict[str, str]:
    started = datetime.now(timezone.utc).isoformat()
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True,
    )
    return {
        "method": "sync; echo 3 > /proc/sys/vm/drop_caches",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
    }


def start_monitors(log_dir: Path) -> list[tuple[subprocess.Popen[bytes], Any, list[str]]]:
    monitors: list[tuple[subprocess.Popen[bytes], Any, list[str]]] = []
    for argv, name in ((["iostat", "-dx", "5"], "iostat"), (["vmstat", "5"], "vmstat")):
        handle = (log_dir / f"star.{name}.txt").open("wb")
        try:
            process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)
        except Exception:
            handle.close()
            raise
        monitors.append((process, handle, argv))
    return monitors


def stop_monitors(
    monitors: list[tuple[subprocess.Popen[bytes], Any, list[str]]], log_dir: Path
) -> list[dict[str, Any]]:
    for process, _, _ in monitors:
        process.terminate()
    records: list[dict[str, Any]] = []
    for process, handle, argv in monitors:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        handle.close()
        path = log_dir / f"star.{Path(argv[0]).name}.txt"
        records.append(
            {
                "argv": argv,
                "exit_status": process.returncode,
                "log": path.name,
                "log_sha256": sha256(path),
            }
        )
    return records


def parse_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        result[key] = value
    return result


def mex_manifest(spatial: Path) -> tuple[str, list[str]]:
    lines: list[str] = []
    for product in PRODUCTS:
        for scale in SCALES:
            for name in ("barcodes.tsv", "features.tsv", "matrix.mtx"):
                path = spatial / product / scale / name
                if not path.is_file():
                    raise SystemExit(f"missing requested MEX component: {path}")
                relative = path.relative_to(spatial).as_posix()
                lines.append(f"{sha256(path)}  ./{relative}")
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest(), lines


def parse_time(path: Path) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    required = (
        "Elapsed (wall clock) time (h:mm:ss or m:ss)",
        "Maximum resident set size (kbytes)",
        "Exit status",
    )
    missing = [name for name in required if name not in fields]
    if missing or fields.get("Exit status") != "0":
        raise SystemExit(f"invalid timing record {path}: missing={missing}")
    return {
        "replicates": 1,
        "elapsed": fields[required[0]],
        "maximum_resident_set_kbytes": int(fields[required[1]]),
        "user_seconds": float(fields["User time (seconds)"]),
        "system_seconds": float(fields["System time (seconds)"]),
        "cpu_percent": fields["Percent of CPU this job got"],
        "exit_status": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", choices=("compatibility", "annotated"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-free-gib", type=float, default=180.0)
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1 or not 0 < args.memory_fraction <= 1:
        raise SystemExit("invalid thread count or memory fraction")
    if args.out_dir.exists() and not args.dry_run:
        raise SystemExit(f"refusing to reuse existing output directory: {args.out_dir}")
    if not args.out_dir.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {args.out_dir.parent}")
    record = build_command(args)
    if args.dry_run:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if not args.authorize_full:
        raise SystemExit("full execution requires explicit --authorize-full")

    checked = preflight(args, record)
    args.out_dir.mkdir()
    (args.out_dir / "commands").mkdir()
    (args.out_dir / "logs").mkdir()
    write_json(args.out_dir / "commands/star.argv.json", record)
    (args.out_dir / "commands/rendered_commands.txt").write_text(
        record["shell_preview"] + "\n", encoding="utf-8"
    )
    write_json(args.out_dir / "preflight.json", checked)
    write_json(
        args.out_dir / "composition.json",
        {
            "schema": SCHEMA,
            "dataset": "ovarian_gex",
            "slide_area": SLIDE_AREA,
            "route": "star_integrated_spatial",
            "products": list(PRODUCTS),
            "omitted_layers": ["soft_expected", "gated_hard", "Space Ranger", "Xenium"],
            "full_execution_authorized": True,
        },
    )

    cache_drop = drop_page_cache()
    monitors = start_monitors(args.out_dir / "logs")
    started = datetime.now(timezone.utc).isoformat()
    timed = [
        "/usr/bin/time", "-v", "-o", str(args.out_dir / "logs/star.time.txt"),
        *record["argv"],
    ]
    try:
        with (args.out_dir / "logs/star.stdout.log").open("wb") as stdout, (
            args.out_dir / "logs/star.stderr.log"
        ).open("wb") as stderr:
            completed = subprocess.run(timed, stdout=stdout, stderr=stderr, check=False)
    finally:
        monitor_records = stop_monitors(monitors, args.out_dir / "logs")
    finished = datetime.now(timezone.utc).isoformat()
    if completed.returncode:
        write_json(
            args.out_dir / "FAILED.json",
            {
                "schema": f"{SCHEMA}.failure",
                "stage": "star_native_spatial_gex",
                "exit_status": completed.returncode,
                "started_at": started,
                "finished_at": finished,
            },
        )
        raise SystemExit(f"STAR failed with status {completed.returncode}")

    spatial = args.out_dir / "starSpatialGex.out"
    run_summary_path = spatial / "run_summary.tsv"
    policy_summary_path = spatial / "summary.tsv"
    if not (spatial / "RUN_COMPLETE").is_file():
        raise SystemExit("STAR exited zero without SpatialGex RUN_COMPLETE")
    summary = parse_summary(run_summary_path)
    if summary.get("schema") != "star_suite.spatial_gex_integrated.v1":
        raise SystemExit("unexpected native SpatialGex summary schema")
    if summary.get("source_revision") != RELEASE_COMMIT:
        raise SystemExit("native output source revision drift")
    if int(summary.get("reads_decoded", "-1")) != EXPECTED_READS:
        raise SystemExit("native summary read count drift")
    if (spatial / "soft_expected").exists() or (spatial / "gated_hard").exists():
        raise SystemExit("unrequested soft or gated output was materialized")
    manifest_hash, manifest_lines = mex_manifest(spatial)
    (args.out_dir / "mex.relative.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    timing = parse_time(args.out_dir / "logs/star.time.txt")

    primary = {
        "schema": f"{SCHEMA}.portable_primary_seal",
        "status": "sealed",
        "dataset": "ovarian_gex",
        "slide_area": SLIDE_AREA,
        "reference": REFERENCE_ID,
        "feature_scope": "GeneFull_including_intronic",
        "rescue_evidence": args.evidence,
        "reported_products": list(PRODUCTS),
        "star_release_tag": RELEASE_TAG,
        "star_commit": RELEASE_COMMIT,
        "star_tree": RELEASE_TREE,
        "star_release_asset_sha256": RELEASE_ARCHIVE_SHA256,
        "star_binary_sha256": STAR_SHA256,
        "native_summary_sha256": sha256(run_summary_path),
        "policy_summary_sha256": sha256(policy_summary_path),
        "mex_components": len(manifest_lines),
        "mex_manifest_sha256": manifest_hash,
        "space_ranger_executed": False,
    }
    write_json(args.out_dir / "PRIMARY_SEAL.json", primary)
    write_json(
        args.out_dir / "RUN_INSTANCE.json",
        {
            "schema": f"{SCHEMA}.run_instance",
            "status": "complete",
            "output_directory": str(args.out_dir.resolve()),
            "binary_path": str(STAR),
            "source_worktree": str(SOURCE_TREE),
            "started_at": started,
            "finished_at": finished,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
            "threads": args.threads,
            "storage": checked["storage"],
            "cache_drop": cache_drop,
            "monitors": monitor_records,
            "timing": timing,
        },
    )
    write_json(args.out_dir / "RECIPE_COMPLETE.json", primary)
    print(json.dumps({"primary": primary, "timing": timing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
