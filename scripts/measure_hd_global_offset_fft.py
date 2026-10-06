#!/usr/bin/env python3
"""Whole-slide and regional nucleus-to-count cross-correlation over +/-60 um (exploratory).

Protocol: docs/the corresponding analysis record, Amendment 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

MAXS = 30
SQ_UM = 2.0


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def ncc_map(N: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """NCC(d) = corr(N(x), C(x + d)) over pairs of supported squares, d in [-MAXS, MAXS]^2."""
    s = S.astype(float)
    n = N * s
    c = C * s
    k = lambda a, b: fftconvolve(a, b[::-1, ::-1], mode="full")
    H, W = N.shape
    cy, cx = H - 1, W - 1
    sl = (slice(cy - MAXS, cy + MAXS + 1), slice(cx - MAXS, cx + MAXS + 1))
    # correlate(a, b)[d] = sum_x a(x) b(x - d); we need sum_x n(x) c(x + d) = correlate(c, n)
    cnt = k(s, s)[sl]
    sn = k(s, n)[sl]      # sum over valid pairs of n(x)
    sc = k(c, s)[sl]      # sum of c(x + d)
    snc = k(c, n)[sl]
    snn = k(s, n * n)[sl]
    scc = k(c * c, s)[sl]
    cov = snc / cnt - (sn / cnt) * (sc / cnt)
    vn = snn / cnt - (sn / cnt) ** 2
    vc = scc / cnt - (sc / cnt) ** 2
    return cov / np.sqrt(vn * vc)


def summarise(f: np.ndarray) -> dict:
    i, j = np.unravel_index(np.nanargmax(f), f.shape)
    return {"peak_dx_um": float((j - MAXS) * SQ_UM), "peak_dy_um": float((i - MAXS) * SQ_UM),
            "peak_ncc": float(f[i, j]), "ncc_at_zero": float(f[MAXS, MAXS])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--visium-inputs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing output: {args.out_dir}")
    lab = np.load(args.visium_inputs / "grid_labels.npz")
    tot = np.load(args.visium_inputs / "square_totals.npz")["total"]
    w = int(round(np.sqrt(len(tot))))
    C = np.log1p(tot.astype(float)).reshape(w, w)
    S = lab["supported"].reshape(w, w)
    regs = {"cellpose_our_registration": (lab["nucleus_id"] > 0).reshape(w, w).astype(float),
            "tenx_published": lab["tenx_nucleus"].reshape(w, w).astype(float)}
    out, surfaces = {}, {}
    edges = np.linspace(0, w, 4).astype(int)
    for name, N in regs.items():
        f = ncc_map(N, C, S)
        surfaces[name] = f
        res = {"whole_slide": summarise(f), "regions": {}}
        for a in range(3):
            for b in range(3):
                ys, xs = slice(edges[a], edges[a + 1]), slice(edges[b], edges[b + 1])
                if S[ys, xs].mean() < 0.2:
                    continue
                res["regions"][f"r{a}c{b}"] = summarise(ncc_map(N[ys, xs], C[ys, xs], S[ys, xs]))
        out[name] = res
        print(json.dumps({name: res["whole_slide"]}), flush=True)
        for k, v in res["regions"].items():
            print("  ", k, json.dumps(v), flush=True)
    args.out_dir.mkdir(parents=True)
    np.savez_compressed(args.out_dir / "ncc_surfaces.npz", **surfaces)
    summary = {"schema": "visium_hd_processing.global_offset_fft.v1",
               "protocol": "docs/the corresponding analysis record#amendment-1",
               "inputs": {"grid_labels.npz": sha256(args.visium_inputs / "grid_labels.npz"),
                          "square_totals.npz": sha256(args.visium_inputs / "square_totals.npz")},
               "max_shift_squares": MAXS, "results": out, "script_sha256": sha256(Path(__file__))}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_dir / "checksums.sha256").open("w") as h:
        for q in sorted(args.out_dir.glob("*")):
            if q.name != "checksums.sha256":
                h.write(f"{sha256(q)}  {q.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
