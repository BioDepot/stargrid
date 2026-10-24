"""The count-based check: where a nucleus mask lies relative to the counts.

On the Visium HD tumour slides examined so far, the total count on capture locations under nuclei
is, on average, lower than on the same locations moved a few micrometres. This is a local
comparison: it does not require nuclei to hold fewer counts than the rest of the tissue. The mean
count under a nucleus mask, as the mask is moved over the grid, is therefore lower where the mask
lies on the nuclei than in a ring of shifts around that position (a dip); the lowest value over a
wide window can lie elsewhere, so the dip is found against its ring. A displacement (dy, dx) is in capture locations of 2 um, rows downward,
from mask to counts: a nucleus at (row, col) samples the counts at (row + dy, col + dx).
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage

RADIUS = 13           # half-width of the shift surface, in locations
SEARCH = 8            # largest shift searched on each axis
RING = (3, 5)         # the ring the centre is compared with, as Chebyshev distances in locations


def shift_surface(nucleus, counts, support, radius=RADIUS):
    """Mean count under the mask at every shift up to `radius`, over supported locations.

    Exact integer sums through a padded FFT, with no wrap at the edge of the capture area.
    Counts must be integers (as counts of molecules are).
    """
    shape = tuple(fft.next_fast_len(n + radius) for n in nucleus.shape)
    nf = fft.rfft2(nucleus.astype(float), s=shape)
    at = np.arange(-radius, radius + 1)
    ix = np.ix_(at % shape[0], at % shape[1])
    values = []
    for target in (counts * support, support.astype(float)):
        corr = fft.irfft2(np.conj(nf) * fft.rfft2(target, s=shape), s=shape)[ix]
        rounded = np.rint(corr)
        if np.max(np.abs(corr - rounded)) > 0.01:
            raise ValueError('FFT integer-sum residual exceeds 0.01; are the counts integers?')
        values.append(rounded)
    out = np.full(values[0].shape, np.nan)
    np.divide(values[0], values[1], out=out, where=values[1] > 0)
    return out


def contrast(surface, dy, dx, radius=RADIUS):
    """One minus the mean count at (dy, dx) over the median on the ring 3 to 5 locations away."""
    yy, xx = np.mgrid[-5:6, -5:6]
    ring = np.maximum(np.abs(yy), np.abs(xx))
    values = surface[radius + dy + yy, radius + dx + xx]
    ref = np.median(values[(ring >= RING[0]) & (ring <= RING[1])])
    centre = surface[radius + dy, radius + dx]
    return float(1 - centre / ref) if np.isfinite(ref) and ref > 0 else None


def feature(surface, radius=RADIUS, search=SEARCH):
    """The shift with the strongest contrast against its ring, with its profile.

    Returns dy, dx (locations), the contrast (positive for a dip), whether it lies on the search
    boundary, and the median count at each Chebyshev radius 0 to 5 around it.
    """
    if radius < search + RING[1]:
        raise ValueError('the surface radius must be at least the search plus the ring radius (5)')
    candidates = []
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            c = contrast(surface, dy, dx, radius)
            if c is not None and np.isfinite(c):
                candidates.append((c, dy, dx))
    if not candidates:
        return {'dy': None, 'dx': None, 'contrast': None, 'sign': 'unresolved',
                'border': False, 'width_um': None, 'ring_profile': []}
    c, dy, dx = min(candidates, key=lambda p: (-abs(p[0]), max(abs(p[1]), abs(p[2])), p[1], p[2]))
    centre = float(surface[radius + dy, radius + dx])
    yy, xx = np.mgrid[-5:6, -5:6]
    rr = np.maximum(np.abs(yy), np.abs(xx))
    patch = surface[radius + dy + yy, radius + dx + xx]
    reference = float(np.median(patch[(rr >= RING[0]) & (rr <= RING[1])]))
    half = (centre + reference) / 2
    profile = [{'radius_squares': r, 'radius_um': 2 * r, 'median_count': float(np.median(patch[rr == r]))}
               for r in range(6)]
    crossing = [p['radius_um'] for p in profile[1:]
                if (reference > centre and p['median_count'] >= half)
                or (reference < centre and p['median_count'] <= half)]
    return {'dy': dy, 'dx': dx, 'dy_um': 2 * dy, 'dx_um': 2 * dx,
            'contrast': c, 'sign': 'dip' if c > 0 else 'peak' if c < 0 else 'flat',
            'border': max(abs(dy), abs(dx)) == search, 'mean_count': centre,
            'ring_median': reference, 'width_um': min(crossing) if crossing else None,
            'ring_profile': profile}


def local_dip(surface, radius=RADIUS, search=SEARCH):
    """The deepest shift whose mean count is below that at all eight neighbouring shifts.

    A weaker criterion than `feature`, for slides where the count under nuclei is lower than at
    neighbouring shifts but not lower than further away. Depth is one minus the value over the
    median of its eight neighbours.
    """
    best = None
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            y, x = radius + dy, radius + dx
            centre = surface[y, x]
            around = surface[y - 1:y + 2, x - 1:x + 2].ravel()
            neighbours = np.delete(around, 4)
            if not np.isfinite(centre) or not np.isfinite(neighbours).all() or not (centre < neighbours).all():
                continue
            depth = float(1 - centre / np.median(neighbours))
            if best is None or depth > best['depth']:
                best = {'dy': dy, 'dx': dx, 'depth': depth, 'mean_count': float(centre),
                        'neighbour_median': float(np.median(neighbours)),
                        'boundary': max(abs(dy), abs(dx)) == search}
    return best or {'status': 'no local dip'}


def regions(nucleus, counts, support, *, divisions=5, minimum=30000, radius=17, search=12,
            minimum_contrast=0.03):
    """The check in each region of a divisions x divisions split of the grid.

    A region is usable if it holds at least `minimum` nucleus locations in the support and its
    feature is an interior dip of contrast at least `minimum_contrast`. The centroid of its nucleus
    locations is recorded for fitting a displacement that varies across the slide.
    """
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
                a, b = max(0, y0 - radius), min(nucleus.shape[0], y1 + radius)
                c, d = max(0, x0 - radius), min(nucleus.shape[1], x1 + radius)
                local = np.zeros((b - a, d - c), bool)
                local[y0 - a:y1 - a, x0 - c:x1 - c] = nucleus[y0:y1, x0:x1]
                surface = shift_surface(local, counts[a:b, c:d], support[a:b, c:d], radius)
                f = feature(surface, radius, search)
                row.update({k: v for k, v in f.items() if k != 'ring_profile'})
                row['contrast_at_zero'] = contrast(surface, 0, 0, radius)
                row['usable'] = bool(f['sign'] == 'dip' and f['contrast'] >= minimum_contrast and not f['border'])
                row['reason'] = 'usable' if row['usable'] else f'not an interior dip of contrast at least {minimum_contrast}'
                surfaces[row['region']] = surface
            else:
                row['reason'] = f'fewer than {minimum} nucleus squares in tissue'
            rows.append(row)
    return rows, surfaces


def edge_distance_profile(counts, support, nucleus, bins_um=(0, 2, 4, 6, 10, 20, 50)):
    """Mean count against distance from the nucleus edge, inside and outside nuclei (descriptive)."""
    d_out = ndimage.distance_transform_edt(~nucleus) * 2
    d_in = ndimage.distance_transform_edt(nucleus) * 2
    rows = [('nucleus, at least 4 um inside', nucleus & (d_in >= 4)),
            ('nucleus, 2-4 um inside', nucleus & (d_in >= 2.1) & (d_in < 4)),
            ('nucleus, edge location', nucleus & (d_in < 2.1))]
    for lo, hi in zip(bins_um[:-1], bins_um[1:]):
        rows.append((f'outside nucleus, {lo}-{hi} um', (~nucleus) & (d_out > lo) & (d_out <= hi)))
    return [{'bin': name, 'mean_count': float(counts[m & support].mean()) if (m & support).any() else None,
             'locations': int((m & support).sum())} for name, m in rows]
