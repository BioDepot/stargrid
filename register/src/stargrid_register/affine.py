"""A displacement that varies linearly across the slide, fitted to regional dips.

The model is [dy, dx] = [1, row, col] @ coefficients, fitted to the regions where the dip is
resolved. It is fitted to alternate regions (a checkerboard of the division) and accepted only if
it predicts at least 75% of the regions left out to within one location; the accepted model is then
refitted to all usable regions. One round of outlier rejection (residual above two locations on
either axis) is allowed in each fit.
"""
from __future__ import annotations

import numpy as np


def design(rows):
    return np.array([[1., r['centroid_y'], r['centroid_x']] for r in rows])


def fit_once_with_rejection(rows):
    if len(rows) < 3 or np.linalg.matrix_rank(design(rows)) < 3:
        raise ValueError('affine design has fewer than three independent positions')
    x = design(rows)
    y = np.array([[r['dy'], r['dx']] for r in rows], float)
    first = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ first
    keep = np.max(np.abs(residual), axis=1) <= 2
    if keep.sum() < 3 or np.linalg.matrix_rank(x[keep]) < 3:
        raise ValueError('outlier rejection leaves fewer than three independent positions')
    fitted = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
    return dict(coefficients=fitted.tolist(), first_coefficients=first.tolist(),
                first_residuals=residual.tolist(),
                retained=[r['region'] for r, k in zip(rows, keep) if k],
                rejected=[r['region'] for r, k in zip(rows, keep) if not k])


def fit_regions(rows, *, minimum_final=12, held_out_tolerance=1, minimum_held_out_fraction=.75):
    """Fit, test on held-out regions, and refit; `rows` as returned by dip.regions."""
    usable = [r for r in rows if r['usable']]
    train = [r for r in usable if (r['row'] + r['col']) % 2 == 0]
    held = [r for r in usable if (r['row'] + r['col']) % 2 == 1]
    out = dict(usable_regions=len(usable), training_regions=len(train), held_out_total=len(held),
               held_out_pass=0, accepted=False,
               model='[dy,dx] = [1,source_row,source_col] @ coefficients',
               held_out=[], final_fit=None)
    try:
        fit = fit_once_with_rejection(train)
    except ValueError as exc:
        out['stop_reason'] = str(exc)
        return out
    out['training_fit'] = fit
    if held:
        pred = design(held) @ np.array(fit['coefficients'])
        for r, p in zip(held, pred):
            residual = np.array([r['dy'], r['dx']]) - p
            within = bool(np.max(np.abs(residual)) <= held_out_tolerance)
            out['held_out'].append(dict(region=r['region'], prediction=p.tolist(),
                                        residual=residual.tolist(), within_one_square=within))
        out['held_out_pass'] = sum(r['within_one_square'] for r in out['held_out'])
    out['held_out_fraction'] = out['held_out_pass'] / len(held) if held else None
    if not held or out['held_out_fraction'] < minimum_held_out_fraction:
        out['stop_reason'] = f'held-out prediction fraction below {minimum_held_out_fraction}'
        return out
    try:
        final = fit_once_with_rejection(usable)
    except ValueError as exc:
        out['stop_reason'] = str(exc)
        return out
    out['final_fit'] = final
    if len(final['retained']) < minimum_final:
        out['stop_reason'] = f'fewer than {minimum_final} regions remain'
        return out
    out['accepted'] = True
    return out


def displacement(coefficients, row, col):
    """The modelled (dy, dx) at grid position (row, col)."""
    return np.array([1.0, row, col]) @ np.asarray(coefficients)
