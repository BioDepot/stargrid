#!/usr/bin/env python3
"""Run one cold-cache ovarian Space Ranger 4.1.0 timing on NVMe.

This runner is intentionally fixed to H1-YQJBZ7X/D1, GRCh38-2024-A,
intron-included counting, 32 cores, 108 GiB, no BAM, no secondary analysis,
no nucleus segmentation, and UMI registration.  It refuses an existing
campaign directory and accepts exactly one successful execution.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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


SCHEMA = "visium_hd_processing.ovarian_spaceranger_4_1_0_nvme_benchmark.v1"
RUNNER = Path(__file__).resolve()
RUNNER_REPOSITORY_PATH = "scripts/publication/run_ovarian_spaceranger_4_1_0_nvme_once.py"
EXPECTED_STORAGE_SOURCE = "/dev/nvme0n1p2"
SPACERANGER_ROOT = Path("<storage>/tools/spaceranger-4.1.0")
SPACERANGER = SPACERANGER_ROOT / "bin/spaceranger"
SPACERANGER_SHA256 = "02d8de06cce15484e3364cd0c87d02603a7ad96fb6c68eb008c58f2ebb30812e"
REFERENCE = Path(
    "<storage>/staged/references/"
    "refdata-gex-GRCh38-2024-A"
)
REFERENCE_IDENTITY_SHA256 = {
    "reference.json": "b3edd924e004c67a523809399a265e3da397b357073b703022c09427e1962819",
    "star/genomeParameters.txt": "d13886c55afff11ee157f33abe9937be40ef39ace7ffaf7ae00a35101b59df4d",
    "star/chrName.txt": "12d83750559b9ae7d273b4b7ce6077e6f24c43eb65672d1fbcfa2e1593fd3ee8",
    "star/chrLength.txt": "b297c2398f93cd3d4c8aa1d4e10d6148c9550a1c2175139c29f9aab505422a17",
    "star/geneInfo.tab": "40f6ac3112cdb0c36ccef1543eb40c075904f02efc2aae33644f1d90424a6e08",
}
INPUT_ROOT = Path(
    "<storage>/staged/inputs/ovarian"
)
FASTQ_ROOT = INPUT_ROOT / "fastqs"
FASTQ_PREFIX = "Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth_S1"
SAMPLE = "Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth"
TISSUE_IMAGE = INPUT_ROOT / "tissue_image.btf"
CYTASSIST_IMAGE = INPUT_ROOT / "cytassist.tif"
SLIDE_FILE = INPUT_ROOT / "slide.vlf"
IMAGE_SHA256 = {
    "tissue_image.btf": "735171f68da751093955ac832abb222ac9753ab23935cfb4459f339284115577",
    "cytassist.tif": "646c10abb09915c0fe58d59ca1954660ba1182ff9f4877392f76f727362296ae",
    "slide.vlf": "1b48b8365a42cae37072250528cd372e0e516bfe5294263dd8ad07dea7dd4f20",
}
FASTQ_SHA256 = {
    f"{FASTQ_PREFIX}_L001_R1_001.fastq.gz": "232f76857f80d4c076a7938247ef0845a3f6ec93f91f123509c8a0477ea675cb",
    f"{FASTQ_PREFIX}_L002_R1_001.fastq.gz": "187eb6e9bb836f6e1dbf642cb8127936ec117c453f553e79817290ae0ed631e9",
    f"{FASTQ_PREFIX}_L003_R1_001.fastq.gz": "bd0fc6f0cc1ca856d17fca279cb292d06f19bc4efb9eb8ad44c2900edd138398",
    f"{FASTQ_PREFIX}_L004_R1_001.fastq.gz": "034b0ee96be6cddbe41a43d27dad74652280bbd00d9c8152b570ae35b71bd6b7",
    f"{FASTQ_PREFIX}_L001_R2_001.fastq.gz": "60fd5670546363aa5fb630dbcbaef05444a4e8f0d182bbe2f193e424a1aa8181",
    f"{FASTQ_PREFIX}_L002_R2_001.fastq.gz": "7aeddaa846b0bec91c1b5d7718de404d96e650ad2f2e75d1bc6b25af78494735",
    f"{FASTQ_PREFIX}_L003_R2_001.fastq.gz": "34c1f3b4d91e45abeab876cc67218dc3c665cc310106dadf61e10267e7d8377e",
    f"{FASTQ_PREFIX}_L004_R2_001.fastq.gz": "7045f3aa194b2defcf046416d76e152bb16ac102a52f15969c7f073b2b200cd0",
}
PRIOR_PARITY = Path(
    "<local>/20260813_nvme_reclaim/"
    "star-spatial-ssd-benchmark-20260729/space_ranger/"
    "ovarian_sr410_nvme_match_v2/provenance/mex_decompressed_parity.tsv"
)
SCALES = ("square_002um", "square_008um", "square_016um")
MEX_KINDS = ("filtered_feature_bc_matrix", "raw_feature_bc_matrix")
MEX_FILES = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_record(argv: Iterable[object]) -> dict[str, Any]:
    values = [str(value) for value in argv]
    return {"schema": "argv.v1", "argv": values, "shell_preview": shlex.join(values)}


def build_command(campaign: Path) -> dict[str, Any]:
    argv = [
        SPACERANGER,
        "count",
        "--id=ovarian_hd3prime_sr410_introns_32c_no_bam_no_seg_nvme_20260813",
        f"--transcriptome={REFERENCE}",
        f"--fastqs={FASTQ_ROOT}",
        f"--sample={SAMPLE}",
        "--create-bam=false",
        "--nosecondary",
        "--nucleus-segmentation=false",
        "--umi-registration=true",
        "--include-introns=true",
        "--localcores=32",
        "--localmem=108",
        f"--output-dir={campaign / 'pipestance'}",
        f"--image={TISSUE_IMAGE}",
        f"--cytaimage={CYTASSIST_IMAGE}",
        "--slide=H1-YQJBZ7X",
        "--area=D1",
    ]
    return command_record(argv)


def storage_source(path: Path) -> str:
    return subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "-T", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_digest(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    observed = sha256(path)
    if observed != expected:
        raise SystemExit(f"SHA256 drift for {path}: {observed} != {expected}")
    return {"bytes": path.stat().st_size, "sha256": observed}


def relative_manifest(root: Path) -> tuple[str, list[str]]:
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest(), lines


def parse_time(path: Path) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    elapsed = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
    if fields.get("Exit status") != "0" or elapsed not in fields:
        raise SystemExit(f"invalid timing record: {path}")
    return {
        "replicates": 1,
        "elapsed": fields[elapsed],
        "maximum_resident_set_kbytes": int(fields["Maximum resident set size (kbytes)"]),
        "user_seconds": float(fields["User time (seconds)"]),
        "system_seconds": float(fields["System time (seconds)"]),
        "cpu_percent": fields["Percent of CPU this job got"],
        "exit_status": 0,
    }


def start_monitors(log_dir: Path) -> list[tuple[subprocess.Popen[bytes], Any, list[str]]]:
    monitors: list[tuple[subprocess.Popen[bytes], Any, list[str]]] = []
    for argv, name in ((["iostat", "-dx", "5"], "iostat"), (["vmstat", "5"], "vmstat")):
        handle = (log_dir / f"spaceranger.{name}.txt").open("wb")
        process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)
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
        path = log_dir / f"spaceranger.{Path(argv[0]).name}.txt"
        records.append(
            {
                "argv": argv,
                "exit_status": process.returncode,
                "log": path.name,
                "log_sha256": sha256(path),
            }
        )
    return records


def expected_mex_hashes() -> dict[str, str]:
    with PRIOR_PARITY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {
        row["relative_path"]: row["fresh_sha256_decompressed"] for row in rows
    }
    if len(result) != 18 or any(row["status"] != "identical" for row in rows):
        raise SystemExit("prior accepted Space Ranger MEX oracle is incomplete")
    return result


def validate_mex(pipestance: Path) -> dict[str, str]:
    expected = expected_mex_hashes()
    observed: dict[str, str] = {}
    root = pipestance / "outs/binned_outputs"
    for scale in SCALES:
        for kind in MEX_KINDS:
            for name in MEX_FILES:
                relative = f"{scale}/{kind}/{name}"
                path = root / relative
                if not path.is_file():
                    raise SystemExit(f"missing accepted MEX component: {path}")
                digest = decompressed_sha256(path)
                if digest != expected[relative]:
                    raise SystemExit(f"MEX parity failure for {relative}")
                observed[relative] = digest
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=250.0)
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = build_command(args.out_dir)
    if args.dry_run:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if not args.authorize_full:
        raise SystemExit("full execution requires explicit --authorize-full")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse campaign directory: {args.out_dir}")
    if not args.out_dir.parent.is_dir():
        raise SystemExit(f"campaign parent does not exist: {args.out_dir.parent}")

    tools = {name: shutil.which(name) for name in ("findmnt", "iostat", "vmstat", "sudo", "sync")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise SystemExit(f"missing benchmark tools: {missing}")
    subprocess.run(["sudo", "-n", "true"], check=True)
    version = subprocess.run(
        [str(SPACERANGER), "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if version != "spaceranger 4.1.0":
        raise SystemExit(f"unexpected Space Ranger version: {version}")
    binary = validate_digest(SPACERANGER, SPACERANGER_SHA256)
    reference_identity = {
        relative: validate_digest(REFERENCE / relative, digest)
        for relative, digest in REFERENCE_IDENTITY_SHA256.items()
    }
    fastqs = {
        name: validate_digest(FASTQ_ROOT / name, digest)
        for name, digest in FASTQ_SHA256.items()
    }
    if len(list(FASTQ_ROOT.glob("*.fastq.gz"))) != len(FASTQ_SHA256):
        raise SystemExit("staged FASTQ directory does not contain exactly eight FASTQs")
    images = {
        name: validate_digest(INPUT_ROOT / name, digest)
        for name, digest in IMAGE_SHA256.items()
    }
    expected_mex = expected_mex_hashes()

    timed_paths = {
        "campaign_parent": args.out_dir.parent,
        "spaceranger": SPACERANGER,
        "reference": REFERENCE,
        "fastqs": FASTQ_ROOT,
        "tissue_image": TISSUE_IMAGE,
        "cytassist_image": CYTASSIST_IMAGE,
    }
    storage = {name: storage_source(path) for name, path in timed_paths.items()}
    drift = {name: source for name, source in storage.items() if source != EXPECTED_STORAGE_SOURCE}
    if drift:
        raise SystemExit(f"timed paths are not all on NVMe: {json.dumps(drift, sort_keys=True)}")
    free_gib = shutil.disk_usage(args.out_dir.parent).free / (1024**3)
    if free_gib < args.minimum_free_gib:
        raise SystemExit(f"only {free_gib:.2f} GiB free; require {args.minimum_free_gib:.2f} GiB")

    args.out_dir.mkdir()
    log_dir = args.out_dir / "logs"
    provenance = args.out_dir / "provenance"
    tmp_dir = args.out_dir / "tmp"
    log_dir.mkdir()
    provenance.mkdir()
    tmp_dir.mkdir()
    reference_manifest_hash, reference_lines = relative_manifest(REFERENCE)
    software_manifest_hash, software_lines = relative_manifest(SPACERANGER_ROOT)
    (provenance / "reference.relative.sha256").write_text(
        "\n".join(reference_lines) + "\n", encoding="utf-8"
    )
    (provenance / "spaceranger.relative.sha256").write_text(
        "\n".join(software_lines) + "\n", encoding="utf-8"
    )
    write_json(provenance / "spaceranger.argv.json", record)
    (provenance / "spaceranger.command.txt").write_text(
        record["shell_preview"] + "\n", encoding="utf-8"
    )
    definition = {
        "schema": f"{SCHEMA}.portable_definition",
        "dataset": "H1-YQJBZ7X/D1",
        "assay": "Visium HD 3-prime",
        "spaceranger_version": "4.1.0",
        "spaceranger_launcher_sha256": SPACERANGER_SHA256,
        "spaceranger_distribution_manifest_sha256": software_manifest_hash,
        "reference": "refdata-gex-GRCh38-2024-A",
        "reference_manifest_sha256": reference_manifest_hash,
        "reference_identity": reference_identity,
        "fastqs": fastqs,
        "images": images,
        "parameters": {
            "threads": 32,
            "localmem_gib": 108,
            "include_introns": True,
            "create_bam": False,
            "secondary_analysis": False,
            "nucleus_segmentation": False,
            "umi_registration": True,
            "image_registration_in_timed_boundary": True,
        },
        "replicates": 1,
        "cold_page_cache": True,
        "runner": {
            "repository_path": RUNNER_REPOSITORY_PATH,
            "sha256": sha256(RUNNER),
        },
        "prior_parity_oracle_sha256": sha256(PRIOR_PARITY),
        "expected_output_mex_decompressed_sha256": expected_mex,
    }
    write_json(args.out_dir / "BENCHMARK_DEFINITION.json", definition)

    subprocess.run(["sync"], check=True)
    cache_started = utc_now()
    subprocess.run(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True
    )
    cache_finished = utc_now()
    monitors = start_monitors(log_dir)
    started = utc_now()
    timed = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(log_dir / "spaceranger.time.txt"),
        *record["argv"],
    ]
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_dir)
    try:
        with (log_dir / "spaceranger.log").open("wb") as log:
            completed = subprocess.run(
                timed,
                cwd=args.out_dir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        monitor_records = stop_monitors(monitors, log_dir)
    finished = utc_now()
    if completed.returncode:
        write_json(
            args.out_dir / "FAILED.json",
            {
                "schema": f"{SCHEMA}.failure",
                "exit_status": completed.returncode,
                "started_at": started,
                "finished_at": finished,
                "automatic_retry": False,
            },
        )
        raise SystemExit(f"Space Ranger failed with status {completed.returncode}")

    pipestance = args.out_dir / "pipestance"
    finalstate_path = pipestance / "_finalstate"
    invocation_path = pipestance / "_invocation"
    for path in (
        finalstate_path,
        invocation_path,
        pipestance / "outs/metrics_summary.csv",
        pipestance / "outs/web_summary.html",
    ):
        if not path.is_file():
            raise SystemExit(f"successful launcher omitted required output: {path}")
    states = json.loads(finalstate_path.read_text(encoding="utf-8"))
    if not states or any(row["state"] not in ("complete", "disabled") for row in states):
        raise SystemExit("Space Ranger final state is not complete")
    invocation = invocation_path.read_text(encoding="utf-8")
    required_contract = (
        "no_bam                    = true",
        "no_secondary_analysis     = true",
        "include_introns           = true",
        "skip_segmentation         = true",
        "disable: false",
    )
    if any(token not in invocation for token in required_contract):
        raise SystemExit("Space Ranger invocation contract drift")
    if next((path for path in (pipestance / "outs").rglob("*.bam")), None):
        raise SystemExit("BAM exists despite --create-bam=false")
    mex_hashes = validate_mex(pipestance)
    mex_manifest = "\n".join(
        f"{digest}  {relative}" for relative, digest in sorted(mex_hashes.items())
    ) + "\n"
    (provenance / "mex_decompressed.sha256").write_text(
        mex_manifest, encoding="utf-8"
    )
    mex_manifest_sha256 = hashlib.sha256(mex_manifest.encode()).hexdigest()
    timing = parse_time(log_dir / "spaceranger.time.txt")

    summary = {
        "schema": f"{SCHEMA}.summary",
        "status": "accepted_success",
        "dataset": "H1-YQJBZ7X/D1",
        "spaceranger_version": version,
        "spaceranger_launcher_sha256": binary["sha256"],
        "reference": "refdata-gex-GRCh38-2024-A",
        "timing": timing,
        "cold_page_cache": True,
        "storage_source": EXPECTED_STORAGE_SOURCE,
        "mex_decompressed_parity": "identical_to_prior_accepted_comparator",
        "mex_components": len(mex_hashes),
        "mex_decompressed_manifest_sha256": mex_manifest_sha256,
        "space_ranger_includes_registration": True,
        "space_ranger_includes_segmentation": False,
        "repeat_policy": "explicit_new_user_authorization_required",
    }
    write_json(args.out_dir / "BENCHMARK_SUMMARY.json", summary)
    write_json(
        args.out_dir / "RUN_INSTANCE.json",
        {
            "schema": f"{SCHEMA}.run_instance",
            "status": "complete",
            "absolute_paths": {name: str(path.resolve()) for name, path in timed_paths.items()},
            "output_directory": str(args.out_dir.resolve()),
            "started_at": started,
            "finished_at": finished,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
            "free_gib_before_run": free_gib,
            "storage": storage,
            "cache_drop": {
                "started_at": cache_started,
                "finished_at": cache_finished,
                "status": "pass",
            },
            "tools": tools,
            "monitors": monitor_records,
            "command": record,
            "timing": timing,
        },
    )
    write_json(args.out_dir / "ACCEPTED_SUCCESS.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
