#!/usr/bin/env python3
"""Stage accepted SPATCH hard totals and the paper's unchanged Cellpose projection.

No registration, segmentation or sequencing is executed. Run through the CPU helper.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'publication'))
sys.path.insert(0, str(HERE.parent/'src'))
from score_hd_flex_registration_roi import load_cellpose_states_from_capture_grid, matrix_from_json, sha256
from prepare_hd_same_run_offsets import square_indices


def mex_totals(root, width=3350, block_bytes=16 << 20):
    """Stream Matrix Market triples; omitted zero-count squares remain zero."""
    root = Path(root)
    positions = []
    for batch in pd.read_csv(root/'barcodes.tsv', sep='\t', header=None, names=['barcode'], chunksize=200_000):
        positions.append(square_indices(batch.barcode, width))
    indices = np.concatenate(positions)
    del positions
    if len(np.unique(indices)) != len(indices):
        raise ValueError('duplicate square barcode')
    totals = np.zeros(width*width, np.float64)
    read = 0
    with (root/'matrix.mtx').open('rb') as h:
        header = h.readline()
        if header.strip() != b'%%MatrixMarket matrix coordinate integer general':
            raise ValueError('expected integer Matrix Market hard counts')
        line = h.readline()
        while line.startswith(b'%'): line = h.readline()
        genes, columns, nnz = map(int, line.split())
        if columns != len(indices): raise ValueError('barcode count differs from matrix header')
        while True:
            block = h.read(block_bytes)
            if not block: break
            block += h.readline()  # Finish the last triple before parsing.
            values = np.fromstring(block.decode('ascii'), sep=' ', dtype=np.int64)
            if len(values) % 3: raise ValueError('incomplete Matrix Market triple')
            values = values.reshape(-1, 3)
            if np.any(values[:,0]<1) or np.any(values[:,0]>genes) or np.any(values[:,1]<1) or np.any(values[:,1]>columns) or np.any(values[:,2]<0):
                raise ValueError('invalid Matrix Market entry')
            np.add.at(totals, indices[values[:,1]-1], values[:,2])
            read += len(values)
        if read != nnz: raise ValueError(f'triple count {read} differs from header {nnz}')
    return totals, {'genes':genes, 'matrix_columns':columns, 'entries':nnz, 'molecules':float(totals.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True, type=Path)
    a = p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'): raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    b = Path('<runs>/')
    root = b/'runs/20260727_spatch_flex_native_he_morphology_v1'
    mex = b/'runs/20260915_v195_spatch_flex_2020a/star/SpatialFlex.out/hard/square_002um'
    grid = root/'capture_grid_primary/capture_grid.json'
    registration = root/'registration/registration.json'
    print('Projecting existing masks with the original paper helper/settings', flush=True)
    states, supported, audit = load_cellpose_states_from_capture_grid(
        root/'cellpose', matrix_from_json(grid, ('spot_colrow_to_cytassist',)),
        np.arange(3350**2, dtype=np.int64), matrix_from_json(registration, ('moving_to_fixed',)),
        None, width=3350, height=3350, moving_downsample=16, moving_sampling='decimate')
    old = json.loads((root/'comparison/summary.json').read_text())['roi_geometry_audit']
    if int((states == 2).sum()) != old['nucleus_mask_bins']:
        raise ValueError('projected nucleus count differs from accepted paper geometry')
    if int(supported.sum()) != old['cellpose_supported_roi_2um_bins']:
        raise ValueError('projected support differs from accepted paper geometry')
    np.savez_compressed(a.out_dir/'grid_labels.npz', nucleus=states == 2, supported=supported)
    print('Summing accepted full-slide hard matrix', flush=True)
    total, matrix = mex_totals(mex)
    np.savez_compressed(a.out_dir/'square_totals.npz', total=total)
    inputs = [grid, registration, root/'cellpose/summary.json', root/'comparison/summary.json',
              mex/'matrix.mtx', mex/'barcodes.tsv', mex/'features.tsv',
              HERE/'publication/score_hd_flex_registration_roi.py']
    summary = {'status':'complete; no new segmentation or registration', 'matrix':matrix,
        'projection':audit, 'nucleus_squares':int((states==2).sum()),
        'inputs':{str(p):sha256(p) for p in inputs}, 'script_sha256':sha256(Path(__file__)),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'cpus':os.environ['OFF_BENCH_CPUS_USED'],
        'outputs':{p.name:sha256(p) for p in a.out_dir.iterdir() if p.is_file()}}
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'matrix':matrix,'nucleus_squares':summary['nucleus_squares']}),flush=True)


if __name__ == '__main__': main()
