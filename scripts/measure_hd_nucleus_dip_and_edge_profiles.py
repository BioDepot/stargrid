#!/usr/bin/env python3
"""Where do counts sit relative to nuclei and to the tissue edge on a real Visium HD slide?

Post hoc correction analysis (2026-10-04), written after review showed that the peak of the
nucleus-count cross-correlation is not the nucleus-to-count offset: counts are lower on nucleus
squares than around them, so alignment is a dip. Not a protocol endpoint; exploratory.

For one slide, from output files only:
  1. shift surface: mean count on nucleus squares when each nucleus map is moved by (dy, dx) squares,
     for the published nucleus map and for any extra maps (--extra-nuclei NAME=grid.npz:key);
  2. displacement between each extra map and the published map (peak of their cross-correlation);
  3. counts against distance from the nearest published nucleus, inside tissue;
  4. counts beyond the published tissue mask against distance from the tissue edge.
Counts are per-square totals (npz array `total`, row-major on the 3350 x 3350 grid).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

W = 3350
SQ_UM = 2.0
S = 8                                                              # shift search, squares
NEAR = [(0, 2.1), (2.1, 4.1), (4.1, 6.1), (6.1, 8.1), (8.1, 10.1), (10.1, 15), (15, 20), (20, 30), (30, 50), (50, 1e9)]
OFF = [(0, 2.1), (2.1, 4.1), (4.1, 6.1), (6.1, 10.1), (10.1, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 500), (500, 1e9)]


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def grid_of(names: pd.Series) -> np.ndarray:
    return names.str.extract(r"s_002um_(\d+)_(\d+)-1").astype(np.int64).to_numpy()


def shift_surface(nuc: np.ndarray, counts: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.full((2 * S + 1, 2 * S + 1), np.nan)
    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            ys, xs = slice(max(dy, 0), W + min(dy, 0)), slice(max(dx, 0), W + min(dx, 0))
            yo, xo = slice(max(-dy, 0), W + min(-dy, 0)), slice(max(-dx, 0), W + min(-dx, 0))
            m = nuc[yo, xo] & support[ys, xs]                 # nucleus at (r, c) scored against counts at (r+dy, c+dx)
            out[dy + S, dx + S] = counts[ys, xs][m].mean()
    return out


def map_displacement(a: np.ndarray, b: np.ndarray) -> dict:
    """Shift (dy, dx) at which map b best matches map a: a(r, c) with b(r + dy, c + dx)."""
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    fa -= fa.mean(); fb -= fb.mean()
    cc = np.fft.irfft2(np.conj(np.fft.rfft2(fa)) * np.fft.rfft2(fb), s=fa.shape)
    norm = float(np.sqrt((fa ** 2).sum() * (fb ** 2).sum()))
    best = max(((cc[dy % W, dx % W], dy, dx) for dy in range(-20, 21) for dx in range(-20, 21)))
    return {"dy_squares": int(best[1]), "dx_squares": int(best[2]), "correlation_at_peak": float(best[0] / norm),
            "correlation_at_zero": float(cc[0, 0] / norm)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slide", required=True)
    p.add_argument("--barcode-mappings", type=Path, required=True)
    p.add_argument("--tissue-positions", type=Path, required=True)
    p.add_argument("--square-totals", type=Path, required=True)
    p.add_argument("--extra-nuclei", action="append", default=[], metavar="NAME=grid.npz:key")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing output: {args.out_dir}")
    bm = pd.read_parquet(args.barcode_mappings, columns=["square_002um", "in_nucleus"])
    rc = grid_of(bm["square_002um"])
    pub = np.zeros((W, W), bool)
    pub[rc[:, 0], rc[:, 1]] = bm["in_nucleus"].fillna(False).to_numpy().astype(bool)
    tp = pd.read_parquet(args.tissue_positions, columns=["array_row", "array_col", "in_tissue"])
    tis = np.zeros((W, W), bool)
    tis[tp["array_row"].to_numpy(), tp["array_col"].to_numpy()] = tp["in_tissue"].to_numpy() == 1
    counts = np.load(args.square_totals)["total"].reshape(W, W).astype(float)
    maps = {"published": pub}
    inputs = {str(args.barcode_mappings): sha256(args.barcode_mappings), str(args.tissue_positions): sha256(args.tissue_positions),
              str(args.square_totals): sha256(args.square_totals)}
    for spec in args.extra_nuclei:
        name, rest = spec.split("=", 1)
        path, key = rest.rsplit(":", 1)
        maps[name] = np.load(path)[key].reshape(W, W) > 0
        inputs[path] = sha256(Path(path))
    surf_rows, summary = [], {"slide": args.slide, "maps": {}}
    for name, m in maps.items():
        s = shift_surface(m, counts, tis)
        i, j = np.unravel_index(np.nanargmin(s), s.shape), np.unravel_index(np.nanargmax(s), s.shape)
        # the dip: minimum within 5 squares of any position, reported with its depth against the ring
        summary["maps"][name] = {"nucleus_squares_in_tissue": int((m & tis).sum()),
                                 "mean_count_at_zero_shift": float(s[S, S]),
                                 "minimum": {"dy": int(i[0] - S), "dx": int(i[1] - S), "mean_count": float(s[i])},
                                 "maximum": {"dy": int(j[0] - S), "dx": int(j[1] - S), "mean_count": float(s[j])}}
        if name != "published":
            summary["maps"][name]["displacement_of_published_map"] = map_displacement(m, pub)
        for dy in range(-S, S + 1):
            for dx in range(-S, S + 1):
                surf_rows.append({"slide": args.slide, "nucleus_map": name, "dy": dy, "dx": dx, "mean_count": s[dy + S, dx + S]})
    d_out = ndimage.distance_transform_edt(~pub) * SQ_UM
    d_in = ndimage.distance_transform_edt(pub) * SQ_UM
    near = [("nucleus, at least 4 um inside", pub & (d_in >= 4)), ("nucleus, 2-4 um inside", pub & (d_in >= 2.1) & (d_in < 4)),
            ("nucleus, edge square", pub & (d_in < 2.1))]
    near += [(f"outside nucleus, {lo:.0f}-{hi:.0f} um" if hi < 1e8 else "outside nucleus, beyond 50 um", (~pub) & (d_out > lo) & (d_out <= hi))
             for lo, hi in NEAR]
    near_rows = [{"slide": args.slide, "bin": lab, "mean_count": float(counts[m & tis].mean()), "squares": int((m & tis).sum())} for lab, m in near]
    d_off = ndimage.distance_transform_edt(~tis) * SQ_UM
    d_inside = ndimage.distance_transform_edt(tis) * SQ_UM
    ref = float(counts[tis & (d_inside >= 20) & (d_inside < 60)].mean())
    off_rows = []
    for lo, hi in OFF:
        m = (~tis) & (d_off > lo) & (d_off <= hi)
        if m.any():
            off_rows.append({"slide": args.slide, "bin": f"{lo:.0f}-{hi:.0f} um" if hi < 1e8 else "beyond 500 um",
                             "mean_count": float(counts[m].mean()), "relative_to_inside": float(counts[m].mean() / ref), "squares": int(m.sum())})
    summary.update({"tissue_share_of_squares": float(tis.mean()), "mean_count_in_tissue": float(counts[tis].mean()),
                    "mean_count_off_tissue": float(counts[~tis].mean()), "share_of_counts_off_tissue": float(counts[~tis].sum() / counts.sum()),
                    "reference_mean_count_20_60um_inside_edge": ref, "inputs": inputs, "script_sha256": sha256(Path(__file__)),
                    "status": "post hoc, exploratory; not a protocol endpoint"})
    args.out_dir.mkdir(parents=True)
    pd.DataFrame(surf_rows).to_csv(args.out_dir / "shift_surface.tsv", sep="\t", index=False)
    pd.DataFrame(near_rows).to_csv(args.out_dir / "nucleus_distance_profile.tsv", sep="\t", index=False)
    pd.DataFrame(off_rows).to_csv(args.out_dir / "off_tissue_profile.tsv", sep="\t", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("inputs",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
