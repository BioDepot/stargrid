#!/usr/bin/env python3
"""Measure same-run nucleus/count offsets using published output files only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from build_hd_cell_matrix import sha256
from measure_hd_global_offset_fft import ncc_map, summarise


def square_indices(barcodes, width):
    rc = pd.Series(barcodes).str.extract(r'^s_002um_(\d+)_(\d+)-\d+$')
    if rc.isna().any().any():
        raise ValueError('invalid 2 um barcode')
    rc = rc.astype(np.int64).to_numpy()
    if np.any(rc < 0) or np.any(rc >= width):
        raise ValueError('barcode outside grid')
    return rc[:, 0] * width + rc[:, 1]


def h5_totals(path, width):
    totals = np.zeros(width * width, np.float64)
    with h5py.File(path) as f:
        m = f['matrix']
        ptr = m['indptr'][:]
        for start in range(0, len(ptr) - 1, 200000):
            stop = min(start + 200000, len(ptr) - 1)
            values = m['data'][ptr[start]:ptr[stop]].astype(np.float64)
            sums = np.concatenate([[0.], np.cumsum(values)])
            column_totals = np.diff(sums[ptr[start:stop + 1] - ptr[start]])
            barcodes = m['barcodes'][start:stop].astype(str)
            idx = square_indices(barcodes, width)
            totals[idx] = column_totals
    return totals


def measure(nucleus, totals, support):
    counts = np.log1p(totals)
    surface = ncc_map(nucleus.astype(float), counts, support)
    result = {'whole_slide': summarise(surface), 'regions': {}}
    edges = np.linspace(0, nucleus.shape[0], 4).astype(int)
    for a in range(3):
        for b in range(3):
            ys, xs = slice(edges[a], edges[a + 1]), slice(edges[b], edges[b + 1])
            if support[ys, xs].mean() < .2:
                continue
            f = ncc_map(nucleus[ys, xs].astype(float), counts[ys, xs], support[ys, xs])
            result['regions'][f'r{a}c{b}'] = summarise(f)
    return result, surface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--barcode-mappings', type=Path, required=True)
    p.add_argument('--tissue-positions', type=Path, required=True)
    p.add_argument('--sr-matrix-h5', type=Path, required=True)
    p.add_argument('--star-totals', type=Path, required=True)
    p.add_argument('--reuse-star-offset', type=Path, help='Only if nucleus, totals and support equal the accepted inputs')
    p.add_argument('--reuse-star-inputs', type=Path)
    p.add_argument('--extra-nuclei', type=Path, help='Accepted native Cellpose grid_labels.npz for the third runbook observation')
    p.add_argument('--width', type=int, default=3350)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    bm = pd.read_parquet(a.barcode_mappings, columns=['square_002um', 'in_nucleus'])
    nucleus = np.zeros(a.width ** 2, bool)
    nucleus[square_indices(bm.square_002um, a.width)] = bm.in_nucleus.fillna(False).to_numpy(bool)
    del bm
    tp = pd.read_parquet(a.tissue_positions, columns=['array_row', 'array_col', 'in_tissue'])
    support = np.zeros(a.width ** 2, bool)
    support[tp.array_row.to_numpy() * a.width + tp.array_col.to_numpy()] = tp.in_tissue.to_numpy() == 1
    star = np.load(a.star_totals)['total']
    sr = h5_totals(a.sr_matrix_h5, a.width)
    np.savez_compressed(a.out_dir / 'inputs.npz', nucleus=nucleus, support=support, star_totals=star, sr_totals=sr)
    shape = (a.width, a.width)
    results, surfaces = {}, {}
    for name, totals in [('space_ranger_same_run', sr), ('star_v195', star)]:
        if name == 'star_v195' and a.reuse_star_offset:
            old = np.load(a.reuse_star_inputs / 'grid_labels.npz')
            np.testing.assert_array_equal(nucleus, old['tenx_nucleus'])
            np.testing.assert_array_equal(support, old['supported'])
            np.testing.assert_array_equal(star, np.load(a.reuse_star_inputs / 'square_totals.npz')['total'])
            results[name] = json.loads(a.reuse_star_offset.read_text())['results']['tenx_published']
            results[name]['reused_summary_sha256'] = sha256(a.reuse_star_offset)
        else:
            results[name], surfaces[name] = measure(nucleus.reshape(shape), totals.reshape(shape), support.reshape(shape))
        print(json.dumps({name: results[name]['whole_slide']}), flush=True)
    if a.extra_nuclei:
        native = np.load(a.extra_nuclei)
        own = (native['nucleus_id'] > 0).reshape(shape)
        for name, totals in [('native_cellpose_vs_space_ranger', sr), ('native_cellpose_vs_star', star)]:
            results[name], surfaces[name] = measure(own, totals.reshape(shape), support.reshape(shape))
            print(json.dumps({name: results[name]['whole_slide']}), flush=True)
    np.savez_compressed(a.out_dir / 'ncc_surfaces.npz', **surfaces)
    summary = {'schema': 'visium_hd_processing.same_run_offset.v1', 'protocol': 'Amendment 10',
        'max_shift_squares': 30, 'shift_sign': 'nucleus(row,col) versus counts(row+dy,col+dx)',
        'results': results, 'nucleus_squares': int(nucleus.sum()), 'supported_squares': int(support.sum()),
        'count_totals': {'space_ranger_same_run': float(sr.sum()), 'star_v195': float(star.sum())},
        'inputs': {str(q): sha256(q) for q in (a.barcode_mappings, a.tissue_positions, a.sr_matrix_h5, a.star_totals)},
        'script_sha256': sha256(Path(__file__)),
        'outputs': {q.name: sha256(q) for q in a.out_dir.glob('*.npz')}}
    if a.extra_nuclei:
        summary['inputs'][str(a.extra_nuclei)] = sha256(a.extra_nuclei)
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__': main()
