"""Command-line interface: stargrid-register {check, refine, fit, apply, place}."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__, affine, dip, io, methods, transform


def _common(p):
    p.add_argument('--counts', type=Path, required=True,
                   help='2 um counts: feature-barcode .h5, Matrix Market directory, or .npz (key "total")')
    p.add_argument('--tissue', type=Path, required=True,
                   help='2 um tissue_positions parquet (Space Ranger output) or .npz (key "tissue")')
    p.add_argument('--nuclei', type=Path, required=True,
                   help='nucleus labels on the grid (.npz or .npy), or Space Ranger barcode_mappings.parquet')
    p.add_argument('--nuclei-key', default=None, help='array name inside the .npz file')
    p.add_argument('--width', type=int, default=io.GRID_WIDTH)
    p.add_argument('--out', type=Path, required=True, help='output JSON')


def _load(a):
    counts = io.load_counts(a.counts, a.width)
    tissue = io.tissue_mask(a.tissue, a.width)
    if a.nuclei.name.endswith('.parquet') or a.nuclei.name.endswith('.parquet.gz'):
        nuclei = io.published_nuclei(a.nuclei, a.width)
    else:
        nuclei = io.load_grid(a.nuclei, a.width, a.nuclei_key) > 0
    return counts, tissue, nuclei


def _distance_um(dy, dx):
    return None if dy is None else 2 * math.hypot(dy, dx)


def check(a):
    counts, tissue, nuclei = _load(a)
    surface = dip.shift_surface(nuclei, counts, tissue, a.radius)
    whole = dip.feature(surface, a.radius, a.search)
    whole['offset_um'] = _distance_um(whole['dy'], whole['dx'])
    result = dict(tool=f'stargrid-register {__version__}', step='check',
                  nucleus_locations_in_tissue=int((nuclei & tissue).sum()), whole=whole,
                  local_dip=dip.local_dip(surface, a.radius, a.search),
                  contrast_at_zero=dip.contrast(surface, 0, 0, a.radius))
    if a.divisions:
        rows, _ = dip.regions(nuclei, counts, tissue, divisions=a.divisions)
        result['regions'] = rows
        usable = [r for r in rows if r['usable']]
        result['usable_regions'] = len(usable)
        result['regions_at_zero'] = sum(r['dy'] == 0 and r['dx'] == 0 for r in usable)
    if a.surface:
        np.save(a.surface, surface)
    return result


def refine(a):
    counts, tissue, nuclei = _load(a)
    return dict(tool=f'stargrid-register {__version__}', step='refine', method=a.method,
                **methods.register(nuclei, counts, tissue, method=a.method, radius=a.radius))


def fit(a):
    counts, tissue, nuclei = _load(a)
    rows, _ = dip.regions(nuclei, counts, tissue, divisions=a.divisions)
    model = affine.fit_regions(rows)
    final = model.get('final_fit') or model.get('training_fit')
    corners = {}
    if final:
        w = a.width - 1
        for name, (r, c) in {'centre': (w / 2, w / 2), 'top_left': (0, 0), 'top_right': (0, w),
                             'bottom_left': (w, 0), 'bottom_right': (w, w)}.items():
            d = affine.displacement(final['coefficients'], r, c)
            corners[name] = dict(displacement_squares=d.tolist(), magnitude_um=float(2 * np.linalg.norm(d)))
    return dict(tool=f'stargrid-register {__version__}', step='fit', regions=rows, fit=model,
                displacement_at_locations=corners)


def apply(a):
    labels = io.load_grid(a.nuclei, a.width, a.nuclei_key)
    if a.shift:
        dy, dx = (int(v) for v in a.shift.split(','))
        moved = transform.shift_grid(labels, dy, dx)
        how = dict(shift=[dy, dx])
    else:
        model = json.loads(a.affine.read_text())['fit']
        coefficients = (model.get('final_fit') or {}).get('coefficients')
        if not model.get('accepted') or coefficients is None:
            raise SystemExit('the affine fit was not accepted; nothing applied')
        moved = transform.apply_affine(labels, coefficients)
        how = dict(affine=coefficients)
    np.savez_compressed(a.out_grid, labels=moved)
    return dict(tool=f'stargrid-register {__version__}', step='apply', output=str(a.out_grid), **how)


def place(a):
    positions = io.read_parquet(a.tissue_positions)
    h, residual = transform.spaceranger_grid_to_image(positions, bin_size_um=a.bin_size_um)
    if a.labels.suffix == '.npy':
        label_image = np.load(a.labels)
    else:
        with np.load(a.labels) as data:
            label_image = data[a.labels_key or data.files[0]]
    frame = json.loads(a.image_to_labels.read_text()) if a.image_to_labels else None
    grid = transform.sample_labels(label_image, h, a.width, a.image_scale,
                                   sampling=a.sampling, image_to_labels=frame)
    np.savez_compressed(a.out_grid, labels=grid)
    return dict(tool=f'stargrid-register {__version__}', step='place', grid_to_image=h.tolist(),
                largest_residual_pixels=residual, output=str(a.out_grid),
                labelled_locations=int((grid > 0).sum()), sampling=a.sampling,
                image_scale=a.image_scale, image_to_labels=frame)


def main(argv=None):
    p = argparse.ArgumentParser(prog='stargrid-register', description=__doc__)
    p.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    sub = p.add_subparsers(dest='step', required=True)

    c = sub.add_parser('check', help='where the nuclei lie relative to the counts (the dip)')
    _common(c)
    c.add_argument('--radius', type=int, default=dip.RADIUS)
    c.add_argument('--search', type=int, default=dip.SEARCH)
    c.add_argument('--divisions', type=int, default=5, help='regions per axis; 0 for the whole slide only')
    c.add_argument('--surface', type=Path, help='also save the shift surface (.npy)')

    r = sub.add_parser('refine', help='confirm with band-pass cross-correlation or mutual information')
    _common(r)
    r.add_argument('--method', choices=['ncc', 'mi'], default='ncc')
    r.add_argument('--radius', type=int, default=12)

    f = sub.add_parser('fit', help='fit a displacement that varies linearly across the slide')
    _common(f)
    f.add_argument('--divisions', type=int, default=5)

    ap = sub.add_parser('apply', help='move a nucleus label grid by a shift or a fitted affine displacement')
    ap.add_argument('--nuclei', type=Path, required=True)
    ap.add_argument('--nuclei-key', default=None)
    ap.add_argument('--width', type=int, default=io.GRID_WIDTH)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--shift', help='dy,dx in locations')
    group.add_argument('--affine', type=Path, help='JSON written by "fit"')
    ap.add_argument('--out-grid', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)

    pl = sub.add_parser('place', help="place image nucleus labels on the grid with Space Ranger's registration")
    pl.add_argument('--tissue-positions', type=Path, required=True, help='Space Ranger 2 or 8 um tissue_positions parquet')
    pl.add_argument('--bin-size-um', type=int, choices=(2, 8), default=None,
                    help='input bin size; inferred from barcodes, otherwise defaults to 2 um')
    pl.add_argument('--labels', type=Path, required=True, help='nucleus label image (.npy or .npz)')
    pl.add_argument('--labels-key', default=None)
    pl.add_argument('--image-scale', type=float, default=1.0,
                    help='label-image pixels per full-resolution pixel (0.5 for a twofold reduction)')
    pl.add_argument('--image-to-labels', type=Path,
                    help='JSON 3 x 3 full-resolution-image to label-image homography')
    pl.add_argument('--sampling', choices=('nearest', 'archived-floor'), default='nearest',
                    help='pixel selection convention; archived-floor matches archived analyses')
    pl.add_argument('--width', type=int, default=io.GRID_WIDTH)
    pl.add_argument('--out-grid', type=Path, required=True)
    pl.add_argument('--out', type=Path, required=True)

    a = p.parse_args(argv)
    result = {'check': check, 'refine': refine, 'fit': fit, 'apply': apply, 'place': place}[a.step](a)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    summary = result.get('whole') or result.get('fit') or {k: v for k, v in result.items() if k != 'regions'}
    print(json.dumps({k: v for k, v in summary.items() if k not in ('ring_profile', 'regions', 'held_out')},
                     default=str)[:2000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
