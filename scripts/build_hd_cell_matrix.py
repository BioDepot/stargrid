#!/usr/bin/env python3
"""Sum 2 um square counts into cells, with deterministic fixed nucleus growth.

Matrix Market inputs and outputs are features by barcodes/cells. Label NPZs
contain a 2-D grid, or a flat grid with nx/ny metadata (or --grid-width).
Positive labels identify nuclei; nonpositive labels are background.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_limits


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def shift_labels(labels, dy=0, dx=0):
    """Positive shifts move nuclei towards increasing count row/column; no wrap."""
    out = np.zeros_like(labels)
    h, w = labels.shape
    if abs(dy) < h and abs(dx) < w:
        out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = labels[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


def expand_labels(labels, growth_um, square_um=2.0):
    """M1 nearest nucleus square, Euclidean distance, ties to smaller label."""
    if not np.isfinite(growth_um) or growth_um < 0 or square_um <= 0:
        raise ValueError('growth must be finite/nonnegative and square size positive')
    labels = np.where(labels > 0, labels, 0).astype(np.int64)
    if labels.ndim != 2:
        raise ValueError('labels must be 2-D')
    radius = int(np.floor(growth_um / square_um + 1e-9))
    out = labels.copy()
    best = np.full(labels.shape, np.inf)
    best[labels > 0] = 0
    h, w = labels.shape
    for dy, dx in sorted(((y, x) for y in range(-radius, radius + 1) for x in range(-radius, radius + 1)), key=lambda p: p[0] ** 2 + p[1] ** 2):
        d2 = dy * dy + dx * dx
        if d2 == 0 or d2 * square_um ** 2 > growth_um ** 2 + 1e-9 or abs(dy) >= h or abs(dx) >= w:
            continue
        target = np.s_[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
        source = labels[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
        current, distance = out[target], best[target]
        take = (source > 0) & ((d2 < distance) | ((d2 == distance) & (source < current)))
        current[take] = source[take]
        distance[take] = d2
    return out


def mex_file(root, name):
    for suffix in ('', '.gz'):
        p = Path(root) / (name + suffix)
        if p.is_file():
            return p
    raise FileNotFoundError(f'{root}/{name}[.gz]')


def open_text(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()


def barcode_rc(barcodes):
    matches = [re.fullmatch(r's_002um_(\d+)_(\d+)-\d+', b) for b in barcodes]
    if not all(matches):
        raise ValueError('expected s_002um_ROW_COL-SUFFIX square barcodes')
    rc = np.array([(int(m[1]), int(m[2])) for m in matches], dtype=np.int64).reshape(-1, 2)
    if len(np.unique(rc, axis=0)) != len(rc):
        raise ValueError('duplicate square coordinates')
    return rc


def load_labels(path, key, width=None):
    with np.load(path) as z:
        lab = np.asarray(z[key])
        if not np.issubdtype(lab.dtype, np.integer):
            raise ValueError('nucleus labels must be integers')
        if lab.ndim == 1:
            if 'nx' in z:
                width = int(z['nx'])
            if width is None:
                raise ValueError('flat labels require nx metadata or --grid-width')
            lab = lab.reshape(-1, width)
    return np.where(lab > 0, lab, 0).astype(np.int64)


def aggregate(matrix, column_labels, cell_ids):
    matrix = matrix.tocoo()
    if len(column_labels) != matrix.shape[1]:
        raise ValueError('matrix barcode dimension mismatch')
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError('counts must be finite and nonnegative')
    keep = column_labels[matrix.col] > 0
    selected = column_labels[matrix.col[keep]]
    cell_col = np.searchsorted(cell_ids, selected)
    if np.any(cell_col >= len(cell_ids)) or np.any(cell_ids[cell_col] != selected):
        raise ValueError('unknown cell label')
    result = sparse.coo_matrix((matrix.data[keep], (matrix.row[keep], cell_col)), shape=(matrix.shape[0], len(cell_ids))).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result, float(matrix.data.sum()), float(matrix.data[keep].sum())


def build(args):
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    paths = {name: mex_file(args.matrix_dir, name) for name in ('matrix.mtx', 'barcodes.tsv', 'features.tsv')}
    with open_text(paths['barcodes.tsv']) as f:
        barcodes = [line.strip() for line in f]
    rc = barcode_rc(barcodes)
    original = load_labels(args.nuclei, args.label_key, args.grid_width)
    cell_ids = np.unique(original[original > 0])
    if not len(cell_ids):
        raise ValueError('no nucleus cells')
    nucleus = shift_labels(original, args.shift_dy, args.shift_dx)
    grown = expand_labels(nucleus, args.growth_um)
    if np.any(rc < 0) or np.any(rc[:, 0] >= grown.shape[0]) or np.any(rc[:, 1] >= grown.shape[1]):
        raise ValueError('matrix square outside label grid')
    with threadpool_limits(limits=4):
        matrix = io.mmread(paths['matrix.mtx'])
    with open_text(paths['features.tsv']) as f:
        feature_count = sum(1 for _ in f)
    if matrix.shape[0] != feature_count:
        raise ValueError('matrix feature dimension mismatch')
    result, input_total, assigned_total = aggregate(matrix, grown[rc[:, 0], rc[:, 1]], cell_ids)
    if not np.isclose(float(result.sum()), assigned_total, rtol=1e-12, atol=1e-8):
        raise ValueError('aggregation failed count conservation')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    mex = args.out_dir / 'cell_by_gene'
    mex.mkdir()
    with threadpool_limits(limits=4):
        io.mmwrite(mex / 'matrix.mtx', result, precision=17, symmetry='general')
    with open_text(paths['features.tsv']) as src, (mex / 'features.tsv').open('w') as dst:
        shutil.copyfileobj(src, dst)
    identifiers = [f'cell_{c}' for c in cell_ids]
    (mex / 'barcodes.tsv').write_text(''.join(c + '\n' for c in identifiers))
    count = lambda grid: np.bincount(np.searchsorted(cell_ids, grid[grid > 0]), minlength=len(cell_ids))
    yy, xx = np.nonzero(original > 0)
    inv = np.searchsorted(cell_ids, original[yy, xx])
    n0 = np.bincount(inv, minlength=len(cell_ids))
    centres_y = (np.bincount(inv, weights=yy + .5, minlength=len(cell_ids)) / n0 + args.shift_dy) * 2
    centres_x = (np.bincount(inv, weights=xx + .5, minlength=len(cell_ids)) / n0 + args.shift_dx) * 2
    pd.DataFrame({'cell_id': identifiers, 'nucleus_id': cell_ids, 'centre_x_um': centres_x,
                  'centre_y_um': centres_y, 'nucleus_squares': count(nucleus), 'cell_squares': count(grown),
                  'molecules': np.asarray(result.sum(axis=0)).ravel()}).to_csv(args.out_dir / 'cells.tsv', sep='\t', index=False)
    yy, xx = np.indices(grown.shape, dtype=np.int32)
    pd.DataFrame({'square': np.arange(grown.size), 'array_row': yy.ravel(), 'array_col': xx.ravel(),
                  'nucleus_id': nucleus.ravel(), 'cell_id': grown.ravel()}).to_parquet(args.out_dir / 'square_to_cell.parquet', index=False)
    outputs = {str(p.relative_to(args.out_dir)): sha256(p) for p in sorted(args.out_dir.rglob('*')) if p.is_file()}
    summary = {'schema': 'visium_hd_processing.cell_matrix.v1', 'growth_um': args.growth_um,
               'shift_squares_dy_dx': [args.shift_dy, args.shift_dx], 'square_um': 2,
               'grid_shape': list(grown.shape), 'cells': len(cell_ids), 'input_molecules': input_total,
               'assigned_molecules': assigned_total, 'assigned_fraction': assigned_total / input_total if input_total else None,
               'matrix_dtype': str(result.dtype), 'matrix_orientation': 'features_by_cells',
               'nucleus_squares_before_shift': int((original > 0).sum()),
               'nucleus_squares_after_shift': int((nucleus > 0).sum()),
               'nucleus_squares_clipped_by_shift': int((original > 0).sum() - (nucleus > 0).sum()),
               'cells_with_all_nucleus_squares_clipped': int((count(nucleus) == 0).sum()),
               'inputs': {str(p.resolve()): sha256(p) for p in [*paths.values(), args.nuclei]},
               'outputs': outputs, 'script_sha256': sha256(Path(__file__))}
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix-dir', type=Path, required=True)
    p.add_argument('--nuclei', type=Path, required=True)
    p.add_argument('--label-key', default='label')
    p.add_argument('--grid-width', type=int)
    p.add_argument('--growth-um', type=float, required=True)
    p.add_argument('--shift-dy', type=int, default=0)
    p.add_argument('--shift-dx', type=int, default=0)
    p.add_argument('--out-dir', type=Path, required=True)
    print(json.dumps(build(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
