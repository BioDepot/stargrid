#!/usr/bin/env python3
"""Resolve prepared paper bindings against accepted, checksum-identified results.

This assembles an export manifest; it does not execute a scientific analysis or
modify an accepted result. All inputs are validated before writing anything.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from export_spatial_paper_numbers import display, evaluate
from summarize_sealed_spatial_primary import sha256


def resolve(base, remaining, analyses, timing, timing_sha256):
    result = {"release": base["release"], "sources": {}, "macros": {}}
    values = {}
    for section in ("sources", "macros"):
        if set(base[section]) & set(remaining[section]):
            raise ValueError(f"colliding {section} definitions")
    for name, original in (base["sources"] | remaining["sources"]).items():
        item = dict(original)
        if "analysis_step" in item:
            complete_path = analyses / item["analysis_step"] / "COMPLETE.json"
            accepted = json.loads(complete_path.read_text())
            if accepted.get("status") != "complete" or accepted.get("exit_code") != 0:
                raise ValueError(f"unaccepted analysis: {name}")
            root = Path(accepted["output_directory"])
            relative = item.get("relative_path", "")
            path = root / relative if relative else root
            digest = accepted["outputs"].get(str(path))
            if not digest:
                raise ValueError(f"source absent from accepted outputs: {name}")
            item.update(path=str(path), sha256=digest,
                        complete_record=str(complete_path),
                        complete_record_sha256=sha256(complete_path),
                        result_id=accepted["result_id"])
        elif "path" not in item:
            if name != "timing":
                raise ValueError(f"unresolved prepared source: {name}")
            item.update(path=str(timing), sha256=timing_sha256)
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source hash drift: {name}")
        if item.get("format", "json") == "json":
            data = json.loads(path.read_text())
        elif item["format"] == "tsv":
            with path.open() as handle:
                data = list(csv.DictReader(handle, delimiter="\t"))
        else:
            raise ValueError(f"unsupported source format: {name}")
        if "expected_cases" in item:
            observed = [row["primary_run_id"].removeprefix("20260915_v195_")
                        for row in data["cases"]]
            if observed != item["expected_cases"]:
                raise ValueError(f"timing case order differs: {name}")
        result["sources"][name], values[name] = item, data
    for name, rows in remaining.get("row_label_checks", {}).items():
        for index, labels in rows.items():
            row = values[name][int(index)]
            if any(str(row.get(key)) != str(value) for key, value in labels.items()):
                raise ValueError(f"row label drift: {name}[{index}]")
    result["macros"] = base["macros"] | remaining["macros"]
    for name, item in result["macros"].items():
        if "value" in item:
            display(evaluate(item["value"], values), item["display"])
        if "old_evidence" in item:
            evaluate(item["old_evidence"]["value"], values)
    result["binding_resolution"] = {
        "row_label_checks": remaining.get("row_label_checks", {}),
        "status": "accepted_sources_verified",
        "scientific_analysis_executed": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("base", "remaining", "analyses", "timing", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--timing-sha256", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing existing manifest: {args.out}")
    result = resolve(json.loads(args.base.read_text()), json.loads(args.remaining.read_text()),
                     args.analyses, args.timing, args.timing_sha256)
    result["binding_resolution"]["preparation_inputs"] = {
        str(path): sha256(path) for path in (args.base, args.remaining)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
