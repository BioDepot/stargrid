"""The registration shift: the profile of counts across the nucleus edge, matched by the correlation ratio.

On most Visium HD slides examined so far, the counts around nuclei follow a pattern that changes with
the distance from the nucleus edge: usually fewer molecules under the nucleus than in a ring just
around it (the dip, which versions 0.1 to 0.5 of this package used) and, in some tissues, more under
the nucleus and its ring than farther away. The check labels the capture locations of the nucleus map
by their distance from the nucleus edge, in bands one location (2 um) wide: three inside the nuclei,
six outside them, and the rest. It divides each count by the mean count of the supported locations in
the square of 2 * `baseline` + 1 locations around it, so that only the local contrast remains. At each
shift of the nucleus map it measures how much of the variance of these corrected counts the bands
explain: the correlation ratio eta^2 (Roche et al., 1998), a similarity measure of multimodal image
registration. The shift is the highest local maximum of eta^2 within the search, accepted if it is
interior to the search and stable when the slide is resampled in spatial blocks. Where no shift is
accepted, the first registration stands. A displacement (dy, dx) is in capture locations of 2 um, rows
downward, from mask to counts: a nucleus at (row, col) lies over the counts at (row + dy, col + dx).
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage

from .dip import BLOCK, DRAWS, SEARCH, SEED, STABLE, block_weights, blocks_for

BANDS = ('I3', 'I2', 'I1', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6')  # inside, deepest first; then outside
BASELINE = 2          # half-width of the baseline window, in locations (5 x 5 locations, +-4 um)


def band_labels(nucleus):
    """The band of every location, by Euclidean distance in locations: 0, 1, 2 under a nucleus more
    than 2, 1 to 2 and at most 1 from its edge; 3 to 8 outside, (0, 1] to (5, 6] from a nucleus; -1
    beyond."""
    nucleus = np.asarray(nucleus, bool)
    inside = ndimage.distance_transform_edt(nucleus)
    outside = ndimage.distance_transform_edt(~nucleus)
    labels = np.full(nucleus.shape, -1, np.int8)
    labels[nucleus & (inside > 2)] = 0
    labels[nucleus & (inside > 1) & (inside <= 2)] = 1
    labels[nucleus & (inside <= 1)] = 2
    for k in range(1, 7):
        labels[~nucleus & (outside > k - 1) & (outside <= k)] = 2 + k
    return labels


def corrected(counts, support, baseline=BASELINE):
    """Each supported count over the mean count of the supported locations within `baseline`
    locations of it on each axis; zero off the support and where that mean is zero."""
    support = np.asarray(support, bool)
    c = np.where(support, np.asarray(counts, float), 0.0)
    size = 2 * baseline + 1
    total = ndimage.uniform_filter(c, size, mode='constant')
    flags = ndimage.uniform_filter(support.astype(float), size, mode='constant')
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = np.where(support & (flags > 1e-12), total / flags, 0.0)
        return np.where(mean > 0, c / mean, 0.0)


def band_sums(labels, values, support, blocks, radius=SEARCH + 1):
    """Per shift, block and band, the number of supported locations under the shifted band map and the
    sum of `values` on them, as ((2r+1)^2, blocks, bands, 2), shifts in row-major order; and per block
    the number of supported locations and the sum and sum of squares of `values`, as (blocks, 3).

    A location counts in the block it lies in, under whichever band the shift brings onto it; the
    bands come from anywhere within `radius` of the block. Sums by FFT of each block; the location
    counts are exact integers.
    """
    size = 2 * radius + 1
    lag = radius - np.arange(-radius, radius + 1)       # the FFT lag of each shift
    height, width = labels.shape
    out = np.zeros((len(blocks), size, size, len(BANDS), 2))
    tissue = np.zeros((len(blocks), 3))
    for k, (y0, y1, x0, x1) in enumerate(blocks):
        flags = support[y0:y1, x0:x1].astype(float)
        if not flags.any():
            continue
        weights = np.where(support[y0:y1, x0:x1], values[y0:y1, x0:x1], 0.0)
        tissue[k] = flags.sum(), weights.sum(), (weights ** 2).sum()
        h, w = flags.shape
        window = np.full((h + 2 * radius, w + 2 * radius), -1, np.int8)
        a, b = max(0, y0 - radius), min(height, y1 + radius)
        c, d = max(0, x0 - radius), min(width, x1 + radius)
        window[a - y0 + radius:b - y0 + radius, c - x0 + radius:d - x0 + radius] = labels[a:b, c:d]
        shape = (fft.next_fast_len(h + 2 * radius, real=True), fft.next_fast_len(w + 2 * radius, real=True))
        bands = fft.rfft2(np.stack([window == j for j in range(len(BANDS))]).astype(float), s=shape)
        small = np.conj(fft.rfft2(np.stack([flags, weights]), s=shape))
        corr = fft.irfft2(small[:, None] * bands[None], s=shape)[:, :, lag[:, None], lag[None, :]]
        n = np.rint(corr[0])
        if np.max(np.abs(corr[0] - n)) > 0.01:
            raise ValueError('FFT integer-sum residual exceeds 0.01')
        out[k, ..., 0] = n.transpose(1, 2, 0)
        out[k, ..., 1] = corr[1].transpose(1, 2, 0)
    return out.reshape(len(blocks), size * size, len(BANDS), 2).transpose(1, 0, 2, 3).copy(), tissue


def eta2(n_band, s_band, n, s, q):
    """The correlation ratio at every shift: the share of the variance of the values explained by the
    bands and the rest. `n_band`, `s_band`: (..., shifts, bands); `n`, `s`, `q`: (...) totals of the
    locations, the values and their squares."""
    n, s, q = (np.asarray(v, float) for v in (n, s, q))
    n_far = n[..., None] - n_band.sum(-1)
    s_far = s[..., None] - s_band.sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        between = (np.where(n_band > 0, s_band ** 2 / n_band, 0).sum(-1)
                   + np.where(n_far > 0, s_far ** 2 / n_far, 0) - (s ** 2 / n)[..., None])
        return between / (q - s ** 2 / n)[..., None]


def highest(surfaces, search=SEARCH):
    """The highest local maximum (above all eight neighbours) within `search` of each surface in a
    stack of shape (n, 2r+1, 2r+1), r > search. Returns (found, dy, dx); ties go to the first shift
    in row-major order."""
    s = np.asarray(surfaces, float)
    radius = (s.shape[1] - 1) // 2
    if radius < search + 1:
        raise ValueError('the surface radius must be at least the search plus one')
    lo, hi = radius - search, radius + search + 1
    centre = s[:, lo:hi, lo:hi]
    around = np.stack([s[:, lo + j:hi + j, lo + i:hi + i] for j in (-1, 0, 1) for i in (-1, 0, 1)
                       if (j, i) != (0, 0)], axis=-1)
    with np.errstate(invalid='ignore'):
        hit = np.isfinite(centre) & np.isfinite(around).all(-1) & (centre[..., None] > around).all(-1)
    flat = np.where(hit, centre, -np.inf).reshape(len(s), -1)
    k = flat.argmax(axis=1)
    side = 2 * search + 1
    return np.isfinite(flat[np.arange(len(s)), k]), k // side - search, k % side - search


def assess(nucleus, counts, support, bounds=None, *, search=SEARCH, draws=DRAWS, seed=SEED, block=BLOCK,
           stable=STABLE, baseline=BASELINE, labels=None, values=None):
    """The registration shift of the nuclei in `bounds` (default the whole grid), and whether to trust it.

    The shift is the highest local maximum of eta^2 within `search` locations. It is usable if it
    lies inside the search (|dy| and |dx| below `search`) and is stable: in at least `stable` of the
    `draws` spatial-block resamples of the slide, the highest local maximum lies within one location
    of it. `labels` (`band_labels` of the nuclei) and `values` (`corrected` counts) may be passed to
    reuse them; both are of the whole grid. `status` is 'usable' if a shift is given and 'not
    registered' otherwise. `profile` gives the mean corrected count in each band, and beyond them,
    at the shift. Returns the result and the eta^2 surface of the full data.
    """
    if bounds is None:
        bounds = (0, nucleus.shape[0], 0, nucleus.shape[1])
    radius = search + 1
    size = 2 * radius + 1
    blocks = blocks_for(support, bounds, block=block, margin=radius)
    row = dict(status='not registered', usable=False, method='correlation ratio', dy=None, dx=None,
               dy_um=None, dx_um=None, eta2=None, eta2_at_zero=None, interior=None, share_any_peak=None,
               share_within_one=None, share_exact=None, candidate_dy=None, candidate_dx=None, profile=None,
               baseline=baseline, blocks=len(blocks), draws=draws, seed=seed, search=search)
    if not len(blocks):
        row['reason'] = 'no tissue'
        return row, np.full((size, size), np.nan)
    if labels is None:
        labels = band_labels(nucleus)
    if values is None:
        values = corrected(counts, support, baseline)
    stats, tissue = band_sums(labels, values, support, blocks, radius)
    n_band, s_band = stats[..., 0].sum(1), stats[..., 1].sum(1)
    n, s, q = tissue.sum(0)
    surface = eta2(n_band, s_band, n, s, q).reshape(size, size)
    row['eta2_at_zero'] = float(surface[radius, radius])
    found, dy, dx = highest(surface[None], search)
    if not found[0]:
        row['reason'] = 'no local maximum'
        return row, surface
    dy, dx = int(dy[0]), int(dx[0])
    w = block_weights(len(blocks), draws, seed)
    boot = eta2(np.einsum('rb,pbj->rpj', w, stats[..., 0], optimize=True),
                np.einsum('rb,pbj->rpj', w, stats[..., 1], optimize=True),
                w @ tissue[:, 0], w @ tissue[:, 1], w @ tissue[:, 2]).reshape(draws, size, size)
    hit, by, bx = highest(boot, search)
    far = np.where(hit, np.maximum(np.abs(by - dy), np.abs(bx - dx)), 10 ** 6)
    k = (radius + dy) * size + radius + dx
    with np.errstate(divide='ignore', invalid='ignore'):
        means = list(s_band[k] / n_band[k]) + [(s - s_band[k].sum()) / (n - n_band[k].sum())]
    row.update(eta2=float(surface[radius + dy, radius + dx]), interior=bool(max(abs(dy), abs(dx)) < search),
               share_any_peak=float(hit.mean()), share_within_one=float((far <= 1).mean()),
               share_exact=float((far == 0).mean()), candidate_dy=dy, candidate_dx=dx,
               profile={b: (float(v) if np.isfinite(v) else None) for b, v in zip(BANDS + ('far',), means)})
    if row['interior'] and row['share_within_one'] >= stable:
        row.update(status='usable', usable=True, dy=dy, dx=dx, dy_um=2 * dy, dx_um=2 * dx, reason='usable')
    else:
        row['reason'] = (f'highest maximum within one location in fewer than {stable:.0%} of resamples'
                         if row['interior'] else 'highest maximum on the edge of the search')
    return row, surface


def regions(nucleus, counts, support, *, divisions=5, minimum=30000, search=SEARCH, draws=DRAWS, seed=SEED,
            block=BLOCK, stable=STABLE, baseline=BASELINE):
    """The check in each region of a divisions x divisions split of the grid.

    A region is usable if it holds at least `minimum` nucleus locations in the support and `assess`
    gives it a shift. The centroid of its nucleus locations is recorded for fitting a displacement
    that varies across the slide.
    """
    labels = band_labels(nucleus)
    values = corrected(counts, support, baseline)
    ey = np.linspace(0, nucleus.shape[0], divisions + 1).astype(int)
    ex = np.linspace(0, nucleus.shape[1], divisions + 1).astype(int)
    rows, surfaces = [], {}
    for iy in range(divisions):
        for ix in range(divisions):
            y0, y1 = ey[iy:iy + 2]
            x0, x1 = ex[ix:ix + 2]
            yy, xx = np.nonzero(nucleus[y0:y1, x0:x1] & support[y0:y1, x0:x1])
            row = dict(region=f'r{iy}c{ix}', row=iy, col=ix, nucleus_squares_in_tissue=len(yy), usable=False,
                       centroid_y=float(yy.mean() + y0) if len(yy) else None,
                       centroid_x=float(xx.mean() + x0) if len(xx) else None)
            if len(yy) >= minimum:
                result, surfaces[row['region']] = assess(
                    nucleus, counts, support, (int(y0), int(y1), int(x0), int(x1)), search=search, draws=draws,
                    seed=seed, block=block, stable=stable, baseline=baseline, labels=labels, values=values)
                row.update(result)
            else:
                row.update(status='not tested', reason=f'fewer than {minimum} nucleus squares in tissue')
            rows.append(row)
    return rows, surfaces
