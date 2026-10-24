"""Synthetic tests: a known displacement between nuclei and the dip in the counts is recovered."""
import json

import numpy as np
import pandas as pd
import pytest

from stargrid_register import affine, cli, dip, io, methods, transform


def slide(width=360, n=900, radius=3, shift=(0, 0), field=None, seed=1, base=12.0, dip_fraction=0.3):
    """Nucleus labels, and integer counts that are lower under the nuclei moved by `shift`.

    `field(row, col)` may give a displacement that varies across the slide instead.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((width, width), np.int32)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = (yy ** 2 + xx ** 2) <= radius ** 2
    centres = rng.integers(radius + 20, width - radius - 20, size=(n, 2))
    for k, (r, c) in enumerate(centres, start=1):
        sub = labels[r - radius:r + radius + 1, c - radius:c + radius + 1]
        sub[disc & (sub == 0)] = k
    rate = np.full((width, width), base)
    rows, cols = np.nonzero(labels)
    if field is None:
        d = np.tile(np.asarray(shift, float), (len(rows), 1))
    else:
        d = np.array([field(r, c) for r, c in zip(rows, cols)])
    tr = np.clip(np.rint(rows + d[:, 0]).astype(int), 0, width - 1)
    tc = np.clip(np.rint(cols + d[:, 1]).astype(int), 0, width - 1)
    rate[tr, tc] = base * (1 - dip_fraction)
    counts = rng.poisson(rate).astype(float)
    return labels, counts, np.ones((width, width), bool)


def test_square_indices_accepts_padded_and_unpadded_names():
    idx = io.square_indices(['s_002um_00002_00003-1', 's_002um_2_3-1'], width=10)
    assert list(idx) == [23, 23]
    with pytest.raises(ValueError):
        io.square_indices(['s_008um_00002_00003-1'], width=10)


def test_shift_surface_matches_direct_sums():
    labels, counts, tissue = slide(width=80, n=40, shift=(1, -2))
    nuc = labels > 0
    surface = dip.shift_surface(nuc, counts, tissue, radius=4)
    for dy, dx in [(0, 0), (1, -2), (-3, 4)]:
        moved = methods.plant(nuc, dy, dx)
        assert surface[4 + dy, 4 + dx] == pytest.approx(counts[moved].mean())


@pytest.mark.parametrize('shift', [(0, 0), (-3, -3), (2, -1)])
def test_feature_finds_the_planted_shift(shift):
    labels, counts, tissue = slide(shift=shift)
    f = dip.feature(dip.shift_surface(labels > 0, counts, tissue))
    assert (f['dy'], f['dx']) == shift
    assert f['sign'] == 'dip' and f['contrast'] > 0.1


def test_local_dip_finds_the_planted_shift():
    labels, counts, tissue = slide(shift=(-2, 1))
    d = dip.local_dip(dip.shift_surface(labels > 0, counts, tissue))
    assert (d['dy'], d['dx']) == (-2, 1)


@pytest.mark.parametrize('method', ['ncc', 'mi'])
def test_band_pass_measures_find_the_planted_shift(method):
    labels, counts, tissue = slide(shift=(-3, 2))
    out = methods.register(labels > 0, counts, tissue, method=method, radius=8)
    assert (out['integer_dy'], out['integer_dx']) == (-3, 2)


def test_regions_and_affine_fit_recover_a_varying_displacement():
    width = 500

    def field(r, c):
        return (-4 + 6 * r / width, 2 - 5 * c / width)

    labels, counts, tissue = slide(width=width, n=2600, field=field, seed=3)
    rows, _ = dip.regions(labels > 0, counts, tissue, divisions=5, minimum=1500, radius=13, search=8)
    assert sum(r['usable'] for r in rows) >= 20
    model = affine.fit_regions(rows, minimum_final=12)
    assert model['accepted'], model.get('stop_reason')
    assert model['held_out_pass'] == model['held_out_total']
    for r, c in [(50, 50), (250, 250), (450, 450)]:
        assert np.max(np.abs(affine.displacement(model['final_fit']['coefficients'], r, c)
                             - np.array(field(r, c)))) < 1.0


def test_apply_affine_moves_labels_onto_the_dip():
    labels, counts, tissue = slide(shift=(-3, 2))
    moved = transform.apply_affine(labels, [[-3, 2], [0, 0], [0, 0]])
    assert (moved > 0).sum() == (transform.shift_grid(labels, -3, 2) > 0).sum()
    f = dip.feature(dip.shift_surface(moved > 0, counts, tissue))
    assert (f['dy'], f['dx']) == (0, 0)


def test_homography_from_space_ranger_positions():
    width = 50
    rows, cols = np.mgrid[0:width, 0:width]
    h_true = np.array([[3.65, 0.01, 1200.0], [-0.02, 3.66, 900.0], [1e-7, -2e-7, 1.0]])
    grid = np.column_stack([cols.ravel(), rows.ravel()]).astype(float)
    image = transform.apply_homography(h_true, grid)
    table = pd.DataFrame(dict(array_row=rows.ravel(), array_col=cols.ravel(),
                              pxl_col_in_fullres=image[:, 0], pxl_row_in_fullres=image[:, 1]))
    h, residual = transform.spaceranger_grid_to_image(table, sample=500)
    assert residual < 1e-6
    label_image = np.zeros((1200, 1500), np.int32)
    label_image[1000:1100, 1300:1400] = 7
    grid_labels = transform.sample_labels(label_image, h, width)
    assert (grid_labels == 7).any()


def test_command_line_check_and_fit(tmp_path):
    labels, counts, tissue = slide(width=360, shift=(-2, -1))
    np.savez(tmp_path / 'counts.npz', total=counts)
    np.savez(tmp_path / 'tissue.npz', tissue=tissue)
    np.savez(tmp_path / 'nuclei.npz', labels=labels)
    args = ['--counts', str(tmp_path / 'counts.npz'), '--tissue', str(tmp_path / 'tissue.npz'),
            '--nuclei', str(tmp_path / 'nuclei.npz'), '--width', '360']
    assert cli.main(['check', *args, '--divisions', '0', '--out', str(tmp_path / 'check.json')]) == 0
    result = json.loads((tmp_path / 'check.json').read_text())
    assert (result['whole']['dy'], result['whole']['dx']) == (-2, -1)
    assert cli.main(['refine', *args, '--radius', '6', '--out', str(tmp_path / 'refine.json')]) == 0
    assert json.loads((tmp_path / 'refine.json').read_text())['integer_dy'] == -2
    assert cli.main(['apply', '--nuclei', str(tmp_path / 'nuclei.npz'), '--width', '360', '--shift=-2,-1',
                     '--out-grid', str(tmp_path / 'moved.npz'), '--out', str(tmp_path / 'apply.json')]) == 0
    moved = io.load_grid(tmp_path / 'moved.npz', 360)
    f = dip.feature(dip.shift_surface(moved > 0, counts, tissue))
    assert (f['dy'], f['dx']) == (0, 0)


def test_other_methods_run_on_a_planted_shift():
    """The kept measures run and return a finite displacement; accuracy is not claimed."""
    cv2 = pytest.importorskip('cv2')
    from stargrid_register import other_methods
    labels, counts, tissue = slide(shift=(-3, 2))
    a, b = methods.prepare(labels > 0, counts, tissue)
    out = other_methods.phase(a, b, tissue, window=False)
    assert out['status'] == 'ok' and np.isfinite([out['dy'], out['dx']]).all()
    out = other_methods.ecc(a, b, tissue, other_methods.FROZEN_2026_10_07, start=(-3, 2))
    assert out['status'] in ('ok', 'nonconverged')


def test_simpleitk_mutual_information_runs():
    pytest.importorskip('SimpleITK')
    from stargrid_register import other_methods
    labels, counts, tissue = slide(shift=(-3, 2))
    a, b = methods.prepare(labels > 0, counts, tissue)
    config = dict(other_methods.FROZEN_2026_10_07, threads=1)
    out = other_methods.sitk_mi(a, b, tissue, config, start=(-3, 2))
    assert out['status'] in ('ok', 'nonconverged')
