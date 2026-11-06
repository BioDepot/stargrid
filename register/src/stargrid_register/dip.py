"""The count-based check: where a nucleus mask lies relative to the counts.

On the Visium HD tumour slides examined so far, the total count on capture locations under nuclei
is, on average, lower than on the same locations moved a few micrometres. This is a local
comparison: it does not require nuclei to hold fewer counts than the rest of the tissue. The mean
count under a nucleus mask, as the mask is moved over the grid, is therefore lower where the mask
lies on the nuclei than at the shifts next to it (a dip). The shift of the deepest dip within the
search is the registration shift, accepted only if it is interior to the search and stable when
the slide is resampled in spatial blocks. Where counts are higher under nuclei there is no dip and
no shift, unless an experimental fallback is asked for (see `assess`). A displacement (dy, dx) is in capture locations of 2 um, rows downward, from mask to
counts: a nucleus at (row, col) samples the counts at (row + dy, col + dx).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import fft, ndimage

RADIUS = 13           # half-width of the shift surface, in locations (the search plus one)
SEARCH = 12           # largest shift searched on each axis
RING = (3, 5)         # the ring of the descriptive contrast, as Chebyshev distances in locations
DRAWS = 200           # spatial-block bootstrap draws
BLOCK = 100           # block side, in capture locations
SEED = 20261007
STABLE = 0.95         # share of draws whose deepest dip must lie within one location of the full-data dip
FRINGE = 3            # width of the band around the nuclei for the experimental fringe fallback, in locations
FALLBACKS = ('fringe', 'peak')  # experimental, off by default (see `assess`)


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
    """One minus the mean count at (dy, dx) over the median on the ring 3 to 5 locations away.

    Descriptive only: no decision rests on it.
    """
    yy, xx = np.mgrid[-5:6, -5:6]
    ring = np.maximum(np.abs(yy), np.abs(xx))
    values = surface[radius + dy + yy, radius + dx + xx]
    ref = np.median(values[(ring >= RING[0]) & (ring <= RING[1])])
    centre = surface[radius + dy, radius + dx]
    return float(1 - centre / ref) if np.isfinite(ref) and ref > 0 else None


def feature(surface, radius=RADIUS, search=RADIUS - RING[1]):
    """The shift with the strongest contrast against its ring, with its profile (descriptive).

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


def dips(surfaces, search=SEARCH):
    """The deepest dip of each surface in a stack, and whether one exists.

    A dip is an offset whose value is lower than at all eight neighbouring offsets; its depth is one
    minus the value over the median of the neighbours. `surfaces` has shape (n, 2r+1, 2r+1) with
    r > search. Returns (found, dy, dx, depth), each of length n; ties go to the first offset in
    row-major order.
    """
    s = np.asarray(surfaces, float)
    radius = (s.shape[1] - 1) // 2
    if radius < search + 1:
        raise ValueError('the surface radius must be at least the search plus one')
    lo, hi = radius - search, radius + search + 1
    centre = s[:, lo:hi, lo:hi]
    around = np.stack([s[:, lo + j:hi + j, lo + i:hi + i] for j in (-1, 0, 1) for i in (-1, 0, 1)
                       if (j, i) != (0, 0)], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        is_dip = np.isfinite(centre) & np.isfinite(around).all(-1) & (centre[..., None] < around).all(-1)
        depth = np.where(is_dip, 1 - centre / np.median(around, axis=-1), -np.inf)
    flat = depth.reshape(len(s), -1)
    k = flat.argmax(axis=1)
    best = flat[np.arange(len(s)), k]
    width = 2 * search + 1
    return np.isfinite(best), k // width - search, k % width - search, best


def local_dip(surface, radius=RADIUS, search=SEARCH):
    """The deepest dip of one surface: the shift whose mean count is below all eight neighbours.

    Returns dy, dx, depth, the mean count, the median of the neighbours and whether the dip lies on
    the edge of the search; `{'status': 'no dip'}` if there is none.
    """
    s = np.asarray(surface, float)
    found, dy, dx, depth = dips(s[None, radius - search - 1:radius + search + 2,
                                  radius - search - 1:radius + search + 2], search)
    if not found[0]:
        return {'status': 'no dip'}
    dy, dx = int(dy[0]), int(dx[0])
    y, x = radius + dy, radius + dx
    neighbours = np.delete(s[y - 1:y + 2, x - 1:x + 2].ravel(), 4)
    return {'dy': dy, 'dx': dx, 'depth': float(depth[0]), 'mean_count': float(s[y, x]),
            'neighbour_median': float(np.median(neighbours)), 'boundary': max(abs(dy), abs(dx)) == search}


def depth_at(surfaces, dy, dx):
    """Depth at a fixed offset on each surface of a stack (nan where it or a neighbour is missing)."""
    s = np.asarray(surfaces, float)
    radius = (s.shape[1] - 1) // 2
    y, x = radius + dy, radius + dx
    patch = s[:, y - 1:y + 2, x - 1:x + 2].reshape(len(s), 9)
    centre, around = patch[:, 4], np.delete(patch, 4, axis=1)
    ref = np.median(around, axis=1)
    ok = np.isfinite(centre) & np.isfinite(around).all(1) & (ref > 0)
    return np.where(ok, 1 - centre / np.where(ok, ref, 1), np.nan)


def blocks_for(support, bounds, *, block=BLOCK, margin=RADIUS):
    """Blocks of `block` x `block` locations over `bounds` (y0, y1, x0, x1) that lie within `margin`
    of tissue, as rows (y0, y1, x0, x1) in row-major order, clipped to the bounds."""
    y0, y1, x0, x1 = bounds
    ya, yb = max(0, y0 - margin), min(support.shape[0], y1 + margin)
    xa, xb = max(0, x0 - margin), min(support.shape[1], x1 + margin)
    near = ndimage.maximum_filter(support[ya:yb, xa:xb], size=2 * margin + 1, mode='constant')
    near = near[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    ny, nx = math.ceil((y1 - y0) / block), math.ceil((x1 - x0) / block)
    ids = (np.indices(near.shape)[0] // block) * nx + np.indices(near.shape)[1] // block
    eligible = np.unique(ids[near])
    return np.array([[y0 + i // nx * block, min(y0 + i // nx * block + block, y1),
                      x0 + i % nx * block, min(x0 + i % nx * block + block, x1)] for i in eligible],
                    np.int64).reshape(-1, 4)


def block_sums(nucleus, counts, support, blocks, radius=RADIUS):
    """Per block and shift, the number of nucleus locations and the summed count under the shifted
    mask over supported locations, as an array of shape ((2r+1)^2, blocks, 2).

    Exact integer sums through a padded FFT of each block. Shifts are in row-major order.
    """
    size = 2 * radius + 1
    at = np.arange(-radius, radius + 1)
    totals = np.pad(counts * support, radius)
    flags = np.pad(support.astype(float), radius)
    out = np.zeros((len(blocks), size, size, 2))
    for k, (y0, y1, x0, x1) in enumerate(blocks):
        a = nucleus[y0:y1, x0:x1]
        if not a.any():
            continue
        h, w = a.shape
        shape = (fft.next_fast_len(h + 2 * radius), fft.next_fast_len(w + 2 * radius))
        nf = np.conj(fft.rfft2(a.astype(float), s=shape))
        ix = np.ix_((at + radius) % shape[0], (at + radius) % shape[1])   # the block starts `radius` into the window
        for j, target in enumerate((flags, totals)):
            corr = fft.irfft2(nf * fft.rfft2(target[y0:y1 + 2 * radius, x0:x1 + 2 * radius], s=shape), s=shape)[ix]
            rounded = np.rint(corr)
            if np.max(np.abs(corr - rounded)) > 0.01:
                raise ValueError('FFT integer-sum residual exceeds 0.01; are the counts integers?')
            out[k, :, :, j] = rounded
    return out.reshape(len(blocks), size * size, 2).transpose(1, 0, 2).copy()


def _ratio(stats):
    with np.errstate(divide='ignore', invalid='ignore'):
        return stats[..., 1] / stats[..., 0]


def resample(stats, draws=DRAWS, seed=SEED):
    """Surfaces of `draws` spatial-block resamples, from the output of `block_sums`.

    The surface at each shift is the weighted sum of the block counts over the weighted sum of the
    block location counts, with multinomial block weights.
    """
    n = stats.shape[1]
    rows = np.random.default_rng(seed).integers(n, size=(draws, n))
    weights = np.array([np.bincount(r, minlength=n) for r in rows], dtype=float)
    return _ratio(np.einsum('rb,pbk->rpk', weights, stats, optimize=True))


def fringe_mask(nucleus, width=FRINGE):
    """The band around the nuclei: locations within `width` locations of a nucleus that are not nucleus."""
    return (ndimage.distance_transform_edt(~nucleus) <= width) & ~nucleus


def peak(surface, boot, search=SEARCH, stable=STABLE):
    """The highest peak of a shift surface, accepted by the rule of the dip (for the experimental fallbacks).

    A peak is a shift whose mean count is higher than at all eight neighbours; the highest (largest
    value over the median of its neighbours) is taken. It is usable if it is interior to the search
    and, in at least `stable` of the resampled surfaces `boot`, the highest peak lies within one
    location of it. Returns peak_dy, peak_dx, peak_height, peak_share_within_one and peak_usable.
    """
    radius = (surface.shape[0] - 1) // 2
    out = dict(peak_dy=None, peak_dx=None, peak_height=None, peak_share_within_one=None, peak_usable=False)
    with np.errstate(divide='ignore'):
        found = local_dip(1 / surface, radius, search)     # a peak of the surface is a dip of its reciprocal
        if 'depth' not in found:
            return out
        hit, by, bx, _ = dips(1 / boot, search)
    dy, dx = found['dy'], found['dx']
    y, x = radius + dy, radius + dx
    height = surface[y, x] / np.median(np.delete(surface[y - 1:y + 2, x - 1:x + 2].ravel(), 4)) - 1
    far = np.where(hit, np.maximum(np.abs(by - dy), np.abs(bx - dx)), 10 ** 6)
    share = float((far <= 1).mean())
    out.update(peak_dy=dy, peak_dx=dx, peak_height=float(height), peak_share_within_one=share,
               peak_usable=bool(max(abs(dy), abs(dx)) < search and share >= stable))
    return out


def assess(nucleus, counts, support, bounds=None, *, search=SEARCH, draws=DRAWS, seed=SEED,
           block=BLOCK, stable=STABLE, fallback=None, fringe=None, fringe_width=FRINGE):
    """The registration shift of the nuclei in `bounds` (default the whole grid), and whether to trust it.

    The shift is the deepest dip within `search` locations of the shift surface. It is usable if it
    lies inside the search (|dy| and |dx| below `search`) and is stable: in at least `stable` of the
    `draws` spatial-block resamples of the slide, the deepest dip lies within one location of it.
    Otherwise the status is 'no dip' or 'unstable' and no shift is given (the deepest dip, if any,
    is kept as `candidate_dy`, `candidate_dx`). The contrast against the ring of shifts 3 to 5
    locations away is descriptive only. Returns the result and the full-data surface.

    `fallback` (experimental, for research; off by default) gives a shift where the dip does not:
    - 'fringe': where there is no usable dip, the highest stable peak of the surface of the band
      around the nuclei (`fringe_mask` of `fringe_width` locations; pass `fringe` to reuse one).
      Where counts are higher just around nuclei, the band has its highest mean count when it lies
      on that band, even where the dip under the nuclei is too weak to pass.
    - 'peak': where the surface of the nuclei has no dip at all, its highest stable peak. Where a
      weak dip is present the highest peak lies on its rim, which is why an unstable dip never
      falls back to it.
    Neither has been tested against an independent reference in a tissue without a dip.
    """
    if bounds is None:
        bounds = (0, nucleus.shape[0], 0, nucleus.shape[1])
    radius = search + 1
    blocks = blocks_for(support, bounds, block=block, margin=radius)
    size = 2 * radius + 1
    row = dict(status='no dip', usable=False, dy=None, dx=None, dy_um=None, dx_um=None, depth=None,
               mean_count=None, neighbour_median=None, interior=None, share_any_dip=None,
               share_within_one=None, share_exact=None, depth_lo=None, depth_hi=None,
               ring_contrast_at_dip=None, candidate_dy=None, candidate_dx=None, method=None, blocks=len(blocks),
               draws=draws, seed=seed, search=search)
    if not len(blocks):
        return row, np.full((size, size), np.nan)
    stats = block_sums(nucleus, counts, support, blocks, radius)
    surface = _ratio(stats.sum(1)).reshape(size, size)
    boot = resample(stats, draws, seed).reshape(draws, size, size)
    found, by, bx, _ = dips(boot, search)
    row['share_any_dip'] = float(found.mean())
    def fall_back(no_dip):
        if fallback == 'peak' and no_dip:
            row.update(peak(surface, boot, search, stable))
        elif fallback == 'fringe':
            band = fringe if fringe is not None else fringe_mask(nucleus, fringe_width)
            fstats = block_sums(band, counts, support, blocks, radius)
            row.update(peak(_ratio(fstats.sum(1)).reshape(size, size),
                            resample(fstats, draws, seed).reshape(draws, size, size), search, stable),
                       fringe_width=fringe_width)
        else:
            return
        row['fallback'] = fallback
        if row.pop('peak_usable'):
            label = 'fringe peak (experimental)' if fallback == 'fringe' else 'peak (experimental)'
            row.update(status=label, method=label, usable=True, dy=row['peak_dy'], dx=row['peak_dx'],
                       dy_um=2 * row['peak_dy'], dx_um=2 * row['peak_dx'], reason='no usable dip; ' + label)

    dip = local_dip(surface, radius, search)
    if 'depth' not in dip:
        fall_back(True)
        return row, surface
    dy, dx = dip['dy'], dip['dx']
    far = np.where(found, np.maximum(np.abs(by - dy), np.abs(bx - dx)), 10 ** 6)
    row['share_within_one'] = float((far <= 1).mean())
    row['share_exact'] = float((far == 0).mean())
    fixed = depth_at(boot, dy, dx)
    if np.isfinite(fixed).any():
        row['depth_lo'], row['depth_hi'] = (float(v) for v in np.percentile(fixed[np.isfinite(fixed)], [2.5, 97.5]))
    row.update(depth=dip['depth'], mean_count=dip['mean_count'], neighbour_median=dip['neighbour_median'],
               candidate_dy=dy, candidate_dx=dx, interior=bool(max(abs(dy), abs(dx)) < search))
    row['usable'] = bool(row['interior'] and row['share_within_one'] >= stable)
    if row['usable']:
        row.update(status='usable', method='dip', dy=dy, dx=dx, dy_um=2 * dy, dx_um=2 * dx)
        a, b = max(0, bounds[0] - RING[1] - radius), min(nucleus.shape[0], bounds[1] + RING[1] + radius)
        c, d = max(0, bounds[2] - RING[1] - radius), min(nucleus.shape[1], bounds[3] + RING[1] + radius)
        local = np.zeros((b - a, d - c), bool)
        local[bounds[0] - a:bounds[1] - a, bounds[2] - c:bounds[3] - c] = nucleus[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        wide = shift_surface(local, counts[a:b, c:d], support[a:b, c:d], radius + RING[1])
        row['ring_contrast_at_dip'] = contrast(wide, dy, dx, radius + RING[1])
    else:
        row['status'] = 'unstable'
        row['reason'] = ('deepest dip on the edge of the search' if not row['interior']
                         else f'deepest dip within one location in fewer than {stable:.0%} of resamples')
        fall_back(False)
    return row, surface


def regions(nucleus, counts, support, *, divisions=5, minimum=30000, search=SEARCH, draws=DRAWS,
            seed=SEED, block=BLOCK, fallback=None, fringe_width=FRINGE):
    """The check in each region of a divisions x divisions split of the grid.

    A region is usable if it holds at least `minimum` nucleus locations in the support and its
    deepest dip is usable (see `assess`). The centroid of its nucleus locations is recorded for
    fitting a displacement that varies across the slide.
    """
    ey = np.linspace(0, nucleus.shape[0], divisions + 1).astype(int)
    ex = np.linspace(0, nucleus.shape[1], divisions + 1).astype(int)
    band = fringe_mask(nucleus, fringe_width) if fallback == 'fringe' else None
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
                result, surfaces[row['region']] = assess(nucleus, counts, support, (int(y0), int(y1), int(x0), int(x1)),
                                                         search=search, draws=draws, seed=seed, block=block,
                                                         fallback=fallback, fringe=band, fringe_width=fringe_width)
                row.update(result)
                row['reason'] = row.get('reason', 'usable' if row['usable'] else row['status'])
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
