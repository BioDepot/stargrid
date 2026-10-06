#!/usr/bin/env python3
"""Export deterministic sparse 2-um bin-total fields for native image scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import io as scipy_io


POLICIES = ("strict", "soft_expected", "hard", "gated_hard")
BARCODE = re.compile(r"^s_002um_(\d+)_(\d+)-1$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_field(path: Path, rows) -> tuple[int, float]:
    occupied, mass = 0, 0.0
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
                output.write("unit_2um\tmolecule_mass\n")
                for unit, value in rows:
                    value = float(value)
                    if value == 0.0:
                        continue
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(f"invalid bin mass for {unit}")
                    output.write(f"{unit}\t{value:.17g}\n")
                    occupied += 1
                    mass += value
    temporary.replace(path)
    return occupied, mass


def mex_bin_totals(root: Path) -> tuple[list[str], np.ndarray, int]:
    barcodes = (root / "barcodes.tsv").read_text().splitlines()
    if len(barcodes) != len(set(barcodes)):
        raise ValueError(f"duplicate MEX barcode: {root}")
    for barcode in barcodes:
        if BARCODE.fullmatch(barcode) is None:
            raise ValueError(f"invalid 2 um MEX barcode: {barcode}")
    feature_count = sum(1 for _ in (root / "features.tsv").open())
    matrix = scipy_io.mmread(root / "matrix.mtx").tocoo()
    if matrix.shape != (feature_count, len(barcodes)):
        raise ValueError(f"MEX axes disagree with matrix dimensions: {root}")
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
        raise ValueError(f"invalid MEX values: {root}")
    totals = np.bincount(matrix.col, weights=matrix.data, minlength=len(barcodes)).astype(np.float64)
    return barcodes, totals, int(matrix.nnz)


def vendor_bin_totals(path: Path, *, column_chunk: int) -> tuple[list[str], np.ndarray]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(value) for value in matrix["shape"][:])
        barcodes_raw = matrix["barcodes"]
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        if len(indptr) != shape[1] + 1 or int(indptr[-1]) != len(matrix["data"]):
            raise ValueError("invalid Space Ranger sparse pointers")
        totals = np.zeros(shape[1], dtype=np.float64)
        for begin in range(0, shape[1], column_chunk):
            end = min(begin + column_chunk, shape[1])
            data_begin, data_end = int(indptr[begin]), int(indptr[end])
            values = np.asarray(matrix["data"][data_begin:data_end], dtype=np.float64)
            prefix = np.empty(len(values) + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(values, out=prefix[1:])
            offsets = indptr[begin:end + 1] - data_begin
            totals[begin:end] = prefix[offsets[1:]] - prefix[offsets[:-1]]
        barcodes = [value.decode() for value in barcodes_raw[:]]
    width = math.isqrt(len(barcodes))
    if width * width != len(barcodes):
        raise ValueError("Space Ranger barcode axis is not a square grid")
    for index in (0, min(width, len(barcodes) - 1), len(barcodes) - 1):
        expected = f"s_002um_{index // width:05d}_{index % width:05d}-1"
        if barcodes[index] != expected:
            raise ValueError(f"Space Ranger barcode axis is not row-major at {index}")
    return barcodes, totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-mex-root", type=Path, required=True)
    parser.add_argument("--space-ranger-h5", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--h5-column-chunk", type=int, default=100_000)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    if args.h5_column_chunk < 1:
        raise SystemExit("--h5-column-chunk must be positive")
    args.out_dir.mkdir(parents=True)

    products: dict[str, object] = {}
    for policy in POLICIES:
        root = args.policy_mex_root / policy / "square_002um"
        barcodes, totals, nnz = mex_bin_totals(root)
        output = args.out_dir / f"{policy}.bin_totals.tsv.gz"
        occupied, mass = write_field(output, zip(barcodes, totals, strict=True))
        products[policy] = {
            "source": str(root.resolve()),
            "matrix_sha256": sha256(root / "matrix.mtx"),
            "output": str(output.resolve()),
            "sha256": sha256(output),
            "matrix_nnz": nnz,
            "occupied_bins": occupied,
            "mass": mass,
        }
    barcodes, totals = vendor_bin_totals(
        args.space_ranger_h5, column_chunk=args.h5_column_chunk,
    )
    output = args.out_dir / "space_ranger.bin_totals.tsv.gz"
    occupied, mass = write_field(output, zip(barcodes, totals, strict=True))
    products["space_ranger"] = {
        "source": str(args.space_ranger_h5.resolve()),
        "source_sha256": sha256(args.space_ranger_h5),
        "output": str(output.resolve()),
        "sha256": sha256(output),
        "occupied_bins": occupied,
        "mass": mass,
    }
    summary = {
        "schema": "visium_hd_processing.policy_mex_bin_totals.v1",
        "scale": "square_002um",
        "products": products,
        "invariants": {
            "raw_mass_not_normalized": True,
            "coordinate_axis_format_and_uniqueness_checked": True,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
