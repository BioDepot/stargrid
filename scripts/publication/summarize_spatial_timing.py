#!/usr/bin/env python3
"""Generate timings and speedups from checksum-pinned GNU time records."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from summarize_sealed_spatial_primary import sha256


def elapsed_seconds(text):
    matches = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", text)
    exits = re.findall(r"Exit status:\s*(\d+)", text)
    if len(matches) != 1 or exits != ["0"]:
        raise ValueError("expected one successful GNU time invocation")
    parts = matches[0].split(":")
    if len(parts) not in (2, 3):
        raise ValueError("invalid elapsed time")
    values = [float(x) for x in parts]
    if any(not math.isfinite(x) or x < 0 for x in values) or any(x >= 60 for x in values[1:]):
        raise ValueError("invalid elapsed time components")
    total = 0.0
    for value in values:
        total = total * 60 + value
    if total <= 0:
        raise ValueError("elapsed time must be positive")
    return total


def summarize(manifest):
    rows = []
    for case in manifest["cases"]:
        row = {"dataset": case["dataset"], "arm": case["arm"], "primary_run_id": case["primary_run_id"]}
        for tool in ("star", "vendor"):
            if tool == "vendor" and case.get(tool) is None:
                if not case.get("vendor_unavailable_reason"):
                    raise ValueError("missing comparator requires an explicit reason")
                row["vendor_seconds"] = None
                row["vendor_unavailable_reason"] = case["vendor_unavailable_reason"]
                continue
            item = case[tool]
            path = Path(item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"time input hash drift: {path}")
            row[f"{tool}_seconds"] = elapsed_seconds(path.read_text())
        if row["vendor_seconds"] is None:
            row["speedup"] = row["time_reduction_percent"] = None
        else:
            row["speedup"] = row["vendor_seconds"] / row["star_seconds"]
            row["time_reduction_percent"] = 100 * (1 - row["star_seconds"] / row["vendor_seconds"])
        rows.append(row)
    return {"schema": "visium_hd_processing.spatial_timings.v1", "release": manifest["release"], "cases": rows, "inputs": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = summarize(json.loads(args.manifest.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
