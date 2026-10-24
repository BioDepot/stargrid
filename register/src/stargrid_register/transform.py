"""Moving nucleus masks on the grid, and placing image masks on the grid.

These functions are written for this package; the analyses behind the STARgrid manuscript moved
masks with the scoring code of that work (see docs/PROVENANCE.md).
"""
from __future__ import annotations

import numpy as np

from .methods import plant


def shift_grid(labels, dy, dx):
    """Move a label grid by whole locations, (dy, dx), mask to counts."""
    return plant(labels, int(dy), int(dx))


def apply_affine(labels, coefficients):
    """Move each labelled location by the modelled displacement at its position, rounded.

    Where two locations land on the same place, the later one in row order is kept; locations
    moved off the grid are dropped.
    """
    rows, cols = np.nonzero(labels)
    if not len(rows):
        return np.zeros_like(labels)
    c = np.asarray(coefficients, float)
    d = np.column_stack([np.ones(len(rows)), rows, cols]) @ c
    new_r = np.rint(rows + d[:, 0]).astype(np.int64)
    new_c = np.rint(cols + d[:, 1]).astype(np.int64)
    keep = (new_r >= 0) & (new_r < labels.shape[0]) & (new_c >= 0) & (new_c < labels.shape[1])
    out = np.zeros_like(labels)
    out[new_r[keep], new_c[keep]] = labels[rows[keep], cols[keep]]
    return out


def fit_homography(source, target):
    """Projective transform taking source (x, y) points to target (x, y) points, by least squares.

    Uses normalised coordinates (Hartley). Returns the 3 x 3 matrix and the largest residual in
    target units.
    """
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    if source.shape != target.shape or source.shape[0] < 4:
        raise ValueError('at least four matching points are needed')

    def normaliser(p):
        centre = p.mean(axis=0)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(p - centre, axis=1))
        return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])

    ts, tt = normaliser(source), normaliser(target)
    s = (ts @ np.column_stack([source, np.ones(len(source))]).T).T
    t = (tt @ np.column_stack([target, np.ones(len(target))]).T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, t):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    # Avoid a quadratic-sized U for a tall fit, while retaining the null
    # vector for the minimally determined four-point (8 x 9) system.
    _, _, vt = np.linalg.svd(np.asarray(rows), full_matrices=len(rows) < 9)
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tt) @ h @ ts
    h /= h[2, 2]
    mapped = apply_homography(h, source)
    return h, float(np.max(np.linalg.norm(mapped - target, axis=1)))


def apply_homography(h, points):
    points = np.asarray(points, float)
    q = np.column_stack([points, np.ones(len(points))]) @ np.asarray(h).T
    return q[:, :2] / q[:, 2:3]


def spaceranger_grid_to_image(tissue_positions, sample=20000, seed=0, bin_size_um=None):
    """Transform from 2 um grid centres to image pixels, from 2 or 8 um positions.

    Space Ranger's output gives, for each bin (array_row, array_col), its position in
    the full-resolution microscope image (pxl_row_in_fullres, pxl_col_in_fullres). A projective
    transform is fitted to a sample of these and checked on all of them. Points are (x, y) =
    (column, row) in both spaces. An 8 um bin centre is at 4*index+1.5 on
    the 2 um grid. Barcode names identify the bin size when present; tables
    without barcodes default to 2 um unless bin_size_um is supplied.
    """
    t = tissue_positions
    if 'barcode' in t:
        sizes = t.barcode.astype(str).str.extract(r'^s_(\d+)um_\d+_\d+-\d+$')[0]
        if sizes.isna().any() or sizes.astype(int).nunique() != 1:
            raise ValueError('positions must have valid barcodes of one bin size')
        observed = int(sizes.iloc[0])
        if bin_size_um is not None and bin_size_um != observed:
            raise ValueError('explicit bin size disagrees with position barcodes')
        bin_size_um = observed
    if bin_size_um is None:
        bin_size_um = 2
    if bin_size_um not in (2, 8):
        raise ValueError('position bin size must be 2 or 8 um')
    grid = np.column_stack([t.array_col.to_numpy(float), t.array_row.to_numpy(float)])
    factor = bin_size_um / 2
    grid = factor * grid + (factor - 1) / 2
    image = np.column_stack([t.pxl_col_in_fullres.to_numpy(float), t.pxl_row_in_fullres.to_numpy(float)])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(grid), size=min(sample, len(grid)), replace=False)
    h, _ = fit_homography(grid[pick], image[pick])
    residual = float(np.max(np.linalg.norm(apply_homography(h, grid) - image, axis=1)))
    return h, residual


def sample_labels(label_image, grid_to_image, width, image_scale=1.0, *,
                  sampling='nearest', image_to_labels=None):
    """Sample labels at grid centres, using nearest pixels by default.

    `image_scale` converts full-resolution pixels to the pixels of `label_image` (for example 0.5
    for masks segmented on an image reduced twofold). Alternatively,
    `image_to_labels` is a 3 x 3 homography that preserves crop and pixel-centre
    offsets. `archived-floor` reproduces the archived analysis convention.
    """
    if sampling not in ('nearest', 'archived-floor'):
        raise ValueError('sampling must be nearest or archived-floor')
    if image_to_labels is not None:
        if image_scale != 1.0:
            raise ValueError('use image_scale or image_to_labels, not both')
        frame = np.asarray(image_to_labels, float)
        if frame.shape != (3, 3) or not np.isfinite(frame).all():
            raise ValueError('image_to_labels must be a finite 3 x 3 matrix')
        grid_to_image = frame @ np.asarray(grid_to_image)
    rows, cols = np.mgrid[0:width, 0:width]
    centres = np.column_stack([cols.ravel(), rows.ravel()]).astype(float)
    xy = apply_homography(grid_to_image, centres) * image_scale
    round_pixel = np.rint if sampling == 'nearest' else np.floor
    x = round_pixel(xy[:, 0]).astype(np.int64)
    y = round_pixel(xy[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < label_image.shape[1]) & (y >= 0) & (y < label_image.shape[0])
    out = np.zeros(width * width, label_image.dtype)
    out[inside] = label_image[y[inside], x[inside]]
    return out.reshape(width, width)
