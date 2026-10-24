"""Reading counts, tissue flags and nucleus maps on the 2 um capture grid.

All arrays are indexed [row, column] on the grid of 2 um capture locations, 3,350 x 3,350 for a
Visium HD capture area. Barcodes of 2 um locations are named s_002um_<row>_<column>-1, with or
without zero padding, in both Space Ranger's and STARgrid's outputs.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import io as scipy_io

GRID_WIDTH = 3350


def square_indices(barcodes, width=GRID_WIDTH):
    """Flat grid indices of 2 um barcodes; rejects barcodes of other bin sizes."""
    rc = pd.Series(barcodes).str.extract(r'^s_002um_(\d+)_(\d+)-\d+$')
    if rc.isna().any().any():
        raise ValueError('invalid 2 um barcode')
    rc = rc.astype(np.int64).to_numpy()
    if np.any(rc < 0) or np.any(rc >= width):
        raise ValueError('barcode outside grid')
    return rc[:, 0] * width + rc[:, 1]


def h5_totals(path, width=GRID_WIDTH):
    """Total count of each 2 um location from a feature-barcode matrix in HDF5."""
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
            totals[square_indices(barcodes, width)] = column_totals
    return totals


def _open(path):
    path = Path(path)
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()


def _first(directory, *names):
    for name in names:
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f'none of {names} in {directory}')


def mtx_totals(directory, width=GRID_WIDTH):
    """Total count of each 2 um location from a Matrix Market directory (genes x barcodes)."""
    directory = Path(directory)
    with _open(_first(directory, 'barcodes.tsv.gz', 'barcodes.tsv')) as handle:
        barcodes = [line.split('\t')[0].strip() for line in handle if line.strip()]
    matrix = scipy_io.mmread(str(_first(directory, 'matrix.mtx.gz', 'matrix.mtx'))).tocsc()
    if matrix.shape[1] != len(barcodes):
        raise ValueError('matrix columns and barcodes differ in number')
    totals = np.zeros(width * width, np.float64)
    totals[square_indices(barcodes, width)] = np.asarray(matrix.sum(axis=0)).ravel()
    return totals


def load_counts(path, width=GRID_WIDTH, key='total'):
    """Counts per location as a width x width array, from .npz, .h5 or a Matrix Market directory."""
    path = Path(path)
    if path.is_dir():
        counts = mtx_totals(path, width)
    elif path.suffix == '.npz':
        with np.load(path) as data:
            counts = data[key].astype(np.float64)
    elif path.suffix in ('.h5', '.hdf5'):
        counts = h5_totals(path, width)
    else:
        raise ValueError('counts must be .npz, .h5 or a Matrix Market directory')
    counts = counts.reshape(width, width)
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('counts must be finite and nonnegative')
    return counts


def read_parquet(path, columns=None):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            return pd.read_parquet(handle, columns=columns)
    return pd.read_parquet(path, columns=columns)


def tissue_mask(path, width=GRID_WIDTH, key='tissue'):
    """Tissue flags from a 2 um tissue_positions parquet file, or a boolean array in .npz."""
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path) as data:
            return data[key].reshape(width, width).astype(bool)
    t = read_parquet(path)
    if 'barcode' in t:
        square_indices(t.barcode, width)          # rejects 8 um tables even if coordinates fit
    rows, cols = t.array_row.to_numpy(), t.array_col.to_numpy()
    idx = rows * width + cols
    if len(idx) != width * width or len(np.unique(idx)) != width * width:
        raise ValueError('tissue positions must cover every 2 um capture position')
    if not np.isin(t.in_tissue, [0, 1]).all():
        raise ValueError('invalid tissue flag')
    result = np.zeros(width * width, bool)
    result[idx] = t.in_tissue.to_numpy() == 1
    return result.reshape(width, width)


def published_nuclei(path, width=GRID_WIDTH):
    """Nucleus flags of 2 um locations from Space Ranger's barcode_mappings.parquet."""
    t = read_parquet(path, ['square_002um', 'in_nucleus'])
    idx = square_indices(t.square_002um, width)
    if len(np.unique(idx)) != len(idx):
        raise ValueError('duplicate nucleus-map square')
    out = np.zeros(width * width, bool)
    out[idx] = t.in_nucleus.fillna(False).to_numpy(bool)
    return out.reshape(width, width)


def load_grid(path, width=GRID_WIDTH, key=None):
    """A grid of nucleus labels (or flags) from .npz or .npy; label 0 is background."""
    path = Path(path)
    if path.suffix == '.npy':
        grid = np.load(path)
    else:
        with np.load(path) as data:
            name = key or ('labels' if 'labels' in data else data.files[0])
            grid = data[name]
    return grid.reshape(width, width)
