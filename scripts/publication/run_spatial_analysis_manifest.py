#!/usr/bin/env python3
"""Run registered matrix-only analyses from sealed primaries and frozen inputs.

The manifest carries exact generator hashes and argv templates. Completed steps
are verified and reused. Failed steps retain their artifacts in numbered attempt
directories. No STAR, vendor, segmentation or registration runner is permitted.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from summarize_sealed_spatial_primary import sha256, summarize

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {
    "summarize_sealed_spatial_primary.py", "summarize_spatial_timing.py",
    "score_hd_spatch_codex.py", "score_spatch_codex_shared_excess.py",
    "compare_visium_to_adjacent_xenium.py", "summarize_adjacent_xenium_top_genes.py",
    "compare_ovarian_panel_in_xenium_cancer_cells.py",
    "prepare_hd_flex_star_sr_fields.py", "score_hd_flex_star_sr_morphology.py",
    "summarize_flex_star_sr_he_consistency.py", "plot_flex_star_sr_he_consistency.py",
    "prepare_hd_flex_he_bin_weights.py", "score_hd_flex_star_sr_gene_bin_residuals.py",
    "summarize_flex_star_sr_gene_bin_he_localization.py",
    "compare_hd_flex_policy_matrices.py", "compare_hd_flex_h5ad_aggregates.py",
    "compare_hd_flex_cross_reference_h5ad.py",
    "export_policy_mex_bin_totals.py", "audit_policy_bin_total_nesting.py",
    "score_gex_xenium_cancer_shared_excess.py", "summarize_three_slide_shared_excess_controls.py",
    "score_hd_native_bin_totals.py", "score_hd_flex_registration_roi.py",
    "score_hd_flex_full_compartments.py", "score_hd_flex_registration_sensitivity.py",
    "plot_spatial_paper_cellpose.py",
    "summarize_spatial_xenium_campaign.py",
}
TOKEN = re.compile(r"\$\{(PRIMARY|FIXED|PARENT|STEP):([a-zA-Z0-9_]+)\}|\$\{OUTPUT\}")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def render(arguments, bindings, output):
    def replace(match):
        if match.group() == "${OUTPUT}":
            return str(output)
        return str(bindings[match[1]][match[2]])
    result = [TOKEN.sub(replace, value) for value in arguments]
    if any("${" in value for value in result):
        raise ValueError("unresolved analysis argument")
    return result


def analysis_argv(step, source, arguments):
    image = step.get("container_image")
    if image is None:
        return [sys.executable, str(source), *arguments]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("analysis container must use an immutable image ID")
    return ["docker", "run", "--rm", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", "<local>:<local>:ro",
            "--volume", "/storage:/storage",
            "--workdir", str(ROOT),
            "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=32",
            image, "python", str(source), *arguments]


def run(manifest, output, selected):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bindings = {"PRIMARY": {}, "FIXED": {}, "PARENT": {}, "STEP": {}}
    identities = {}
    for name, path_text in manifest["primaries"].items():
        path = Path(path_text)
        summary = summarize(path, manifest["release"])
        seal = json.loads((path / "PRIMARY_SEAL.json").read_text())
        component_manifest = path / "mex.relative.sha256"
        if sha256(component_manifest) != seal["mex_manifest_sha256"]:
            raise ValueError(f"MEX manifest drift: {name}")
        lines = component_manifest.read_text().splitlines()
        if len(lines) != 36 or len(set(lines)) != 36:
            raise ValueError(f"invalid MEX component manifest: {name}")
        spatial = Path(summary["spatial_output"])
        for line in lines:
            digest, relative = line.split("  ", 1)
            source = spatial / relative
            if not source.resolve().is_relative_to(spatial) or sha256(source) != digest:
                raise ValueError(f"MEX component drift: {source}")
        bindings["PRIMARY"][name] = path
        identities["primary:" + name] = summary["primary_seal_sha256"]
    for name, item in manifest["fixed_inputs"].items():
        path = Path(item["path"])
        observed = sha256(path)
        if observed != item["sha256"]:
            raise ValueError(f"frozen input drift: {name}")
        bindings["FIXED"][name] = path
        bindings["PARENT"][name] = path.parent
        identities["fixed:" + name] = observed
    steps = manifest["steps"]
    if len({s["id"] for s in steps}) != len(steps):
        raise ValueError("duplicate step ID")
    for step in steps:
        name = step["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
            raise ValueError("invalid step ID")
        source = ROOT / step["generator"]
        if source.name not in ALLOWED or source.resolve().parent not in (ROOT / "scripts", ROOT / "scripts/publication"):
            raise ValueError(f"unapproved analysis entrypoint: {source}")
        if sha256(source) != step["generator_sha256"]:
            raise ValueError(f"generator drift: {source}")
        fingerprint = hashlib.sha256(json.dumps({"step": step, "inputs": identities}, sort_keys=True).encode()).hexdigest()
        complete_path = output / name / "COMPLETE.json"
        if complete_path.exists():
            old = json.loads(complete_path.read_text())
            if old["fingerprint"] != fingerprint or any(sha256(Path(p)) != digest for p, digest in old["outputs"].items()):
                raise ValueError(f"completed analysis changed: {name}")
            bindings["STEP"][name] = Path(old["output_directory"])
            continue
        if selected and name not in selected:
            continue
        missing = set(step.get("depends_on", [])) - bindings["STEP"].keys()
        if missing:
            raise ValueError(f"missing dependencies for {name}: {sorted(missing)}")
        step_root = output / name
        step_root.mkdir(exist_ok=True)
        attempt = 1
        while (step_root / f"attempt{attempt}").exists():
            attempt += 1
        attempt_root = step_root / f"attempt{attempt}"
        attempt_root.mkdir()
        destination = attempt_root / "result"
        argv = analysis_argv(step, source, render(step["arguments"], bindings, destination))
        record = {"status": "running", "fingerprint": fingerprint, "argv": argv,
                  "result_id": step.get("result_id"),
                  "started_utc": datetime.now(timezone.utc).isoformat(), "output_directory": str(destination)}
        write(attempt_root / "attempt.json", record)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="32")
        with (attempt_root / "console.log").open("w") as log:
            rc = subprocess.run(["/usr/bin/time", "-v", "-o", str(attempt_root / "time.txt"), *argv], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record.update(exit_code=rc, status="complete" if rc == 0 else "failed", finished_utc=datetime.now(timezone.utc).isoformat())
        write(attempt_root / "attempt.json", record)
        if rc:
            raise RuntimeError(f"analysis failed: {name}; see {attempt_root}")
        files = sorted(p for p in destination.rglob("*") if p.is_file()) if destination.is_dir() else [destination]
        if not files or any(not p.is_file() for p in files):
            raise ValueError(f"analysis produced no result: {name}")
        record["outputs"] = {str(p): sha256(p) for p in files}
        write(complete_path, record)
        bindings["STEP"][name] = destination
        print(name, "complete", flush=True)
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", action="append", default=[])
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text()), args.out, set(args.step))


if __name__ == "__main__":
    main()
