"""Two standard image-registration measures, used to confirm the dip and refine it.

Normalised cross-correlation (minimised: the counts dip under the mask) and mutual
information (maximised) between the mask and log counts, both after subtracting from each image a
copy blurred at the scale of the tissue (band-pass, sigma 8 locations). In the tests behind the
STARgrid manuscript these two found the planted and the real displacements; phase correlation,
ECC and optimised mutual information did not. Those are kept in `other_methods` for later work.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage


def prepare(nucleus, counts, tissue, bandpass=True, sigma=8):
    """Mask and log(1 + counts) on the tissue, each minus a Gaussian blur of itself."""
    if nucleus.shape != counts.shape or counts.shape != tissue.shape:
        raise ValueError('mask, counts, and tissue must have the same shape')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('counts must be finite and nonnegative')
    a = np.asarray(nucleus, np.float32) * tissue
    b = np.log1p(counts).astype(np.float32) * tissue
    if bandpass:
        a -= ndimage.gaussian_filter(a, sigma, mode='constant', cval=0)
        b -= ndimage.gaussian_filter(b, sigma, mode='constant', cval=0)
    return np.ascontiguousarray(a * tissue), np.ascontiguousarray(b * tissue)


def slices(shape, dy, dx):
    """Non-wrapping source and destination slices for mask -> counts."""
    h, w = shape
    return ((slice(max(-dy, 0), h - max(dy, 0)), slice(max(-dx, 0), w - max(dx, 0))),
            (slice(max(dy, 0), h - max(-dy, 0)), slice(max(dx, 0), w - max(-dx, 0))))


def plant(mask, dy, dx):
    """The mask moved by (dy, dx), without wrapping."""
    out = np.zeros_like(mask)
    src, dst = slices(mask.shape, dy, dx)
    out[dst] = mask[src]
    return out


def ncc_surface(a, b, tissue, radius=12):
    """Pearson correlation on joint tissue support, at every bounded integer lag."""
    if radius >= min(a.shape):
        raise ValueError('radius must be smaller than both image dimensions')
    shape = tuple(fft.next_fast_len(n + radius) for n in a.shape)
    at = np.arange(-radius, radius + 1)
    ix = np.ix_(at % shape[0], at % shape[1])
    t = tissue.astype(np.float64)
    a = np.asarray(a, np.float64) * t
    b = np.asarray(b, np.float64) * t
    tf, af, bf = fft.rfft2(t, s=shape), fft.rfft2(a, s=shape), fft.rfft2(b, s=shape)

    def corr(x, y):
        return fft.irfft2(np.conj(x) * y, s=shape)[ix]

    n = np.rint(corr(tf, tf))
    sa, sb = corr(af, tf), corr(tf, bf)
    sab = corr(af, bf)
    saa = corr(fft.rfft2(a * a, s=shape), tf)
    sbb = corr(tf, fft.rfft2(b * b, s=shape))
    with np.errstate(invalid='ignore', divide='ignore'):
        denominator = np.sqrt(np.maximum(0, saa - sa * sa / n) * np.maximum(0, sbb - sb * sb / n))
        surface = (sab - sa * sb / n) / denominator
    surface[(n < 3) | (denominator <= 0)] = np.nan
    return surface


def histogram_codes(image, tissue, bins):
    if not tissue.any() or bins < 2:
        raise ValueError('nonempty tissue and at least two bins required')
    lo, hi = float(image[tissue].min()), float(image[tissue].max())
    if hi == lo:
        return np.zeros(image.shape, np.int32), [lo, hi]
    codes = np.floor((image.astype(np.float64) - lo) * bins / (hi - lo))
    return np.clip(codes, 0, bins - 1).astype(np.int32), [lo, hi]


def mutual_information(histogram):
    total = histogram.sum()
    if total <= 0:
        return float('nan')
    p = histogram / total
    expected = p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :]
    present = p > 0
    return float(np.sum(p[present] * np.log(p[present] / expected[present])))


def mi_surface(a, b, tissue, radius=12, bins=32):
    """Empirical joint-histogram mutual information in nats, with fixed bin edges across lags."""
    ac, arange = histogram_codes(a, tissue, bins)
    bc, brange = histogram_codes(b, tissue, bins)
    out = np.full((2 * radius + 1, 2 * radius + 1), np.nan)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            src, dst = slices(a.shape, dy, dx)
            keep = tissue[src] & tissue[dst]
            hist = np.bincount((ac[src][keep] * bins + bc[dst][keep]),
                               minlength=bins * bins).reshape(bins, bins)
            out[dy + radius, dx + radius] = mutual_information(hist)
    return out, dict(bins=bins, mask_range=arange, count_range=brange)


def optimum(surface, *, minimise, subpixel=False):
    """The best lag of a metric surface, with an optional parabolic sub-location estimate."""
    radius = surface.shape[0] // 2
    values = surface if minimise else -surface
    if not np.isfinite(values).any():
        return dict(status='unresolved', reason='no finite metric values')
    y, x = np.unravel_index(np.nanargmin(values), values.shape)
    offset = np.zeros(2)
    if subpixel:
        for axis, index in enumerate((y, x)):
            if 0 < index < values.shape[axis] - 1:
                line = values[y - 1:y + 2, x] if axis == 0 else values[y, x - 1:x + 2]
                denom = line[0] - 2 * line[1] + line[2]
                if np.isfinite(line).all() and denom > 0:
                    offset[axis] = np.clip((line[0] - line[2]) / (2 * denom), -.5, .5)
    filled = np.where(np.isfinite(values), values, np.inf)
    local = np.argwhere(filled == ndimage.minimum_filter(filled, size=3, mode='constant', cval=np.inf))
    local = sorted((int(i), int(j)) for i, j in local if np.isfinite(values[i, j]))
    local.sort(key=lambda ij: (values[ij], max(abs(ij[0] - radius), abs(ij[1] - radius)), ij))
    alternatives = [dict(dy=i - radius, dx=j - radius, metric=float(surface[i, j]))
                    for i, j in local if (i, j) != (y, x)]
    return dict(status='ok', dy=float(y - radius + offset[0]), dx=float(x - radius + offset[1]),
                integer_dy=int(y - radius), integer_dx=int(x - radius),
                metric=float(surface[y, x]), boundary=bool(y in (0, 2 * radius) or x in (0, 2 * radius)),
                next_local_optimum=alternatives[0] if alternatives else None)


def register(nucleus, counts, tissue, *, method='ncc', radius=12, sigma=8, subpixel=True):
    """The displacement of the mask by one measure: 'ncc' (minimised) or 'mi' (maximised)."""
    a, b = prepare(nucleus, counts, tissue, bandpass=True, sigma=sigma)
    if method == 'ncc':
        return optimum(ncc_surface(a, b, tissue, radius), minimise=True, subpixel=subpixel)
    if method == 'mi':
        surface, _ = mi_surface(a, b, tissue, radius)
        return optimum(surface, minimise=False, subpixel=False)
    raise ValueError("method must be 'ncc' or 'mi'")
