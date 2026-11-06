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


def surface_with(dip_at=(0, 0), depth=0.3, ring=0.0, far=0.0, radius=13, seed=0):
    """A synthetic shift surface: flat at 1, a one-location dip at `dip_at`, a ring of values raised
    by `ring` at 3 to 5 locations around it, and values raised by `far` beyond (lung-like)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    surface = 1 + 0.002 * rng.standard_normal(yy.shape)
    d = np.maximum(np.abs(yy - dip_at[0]), np.abs(xx - dip_at[1]))
    surface[(d >= 3) & (d <= 5)] += ring
    surface[d > 5] += far
    surface[d == 0] = 1 - depth
    return surface


def test_dip_is_chosen_with_a_ring_of_higher_values():
    d = dip.local_dip(surface_with((-2, 3), ring=0.2))
    assert (d['dy'], d['dx']) == (-2, 3) and d['depth'] == pytest.approx(0.3, abs=0.01)


def test_dip_is_chosen_where_higher_values_lie_farther_out():
    """The ring rule would pick the rim (a peak at 6 to 8 locations); the local dip is the dip."""
    surface = surface_with((1, -1), depth=0.1, far=0.3)
    d = dip.local_dip(surface)
    assert (d['dy'], d['dx']) == (1, -1)
    f = dip.feature(surface)
    assert (f['dy'], f['dx']) != (1, -1)


def test_pure_peak_has_no_dip():
    yy, xx = np.mgrid[-13:14, -13:14]
    peak = 1 + 0.3 * np.exp(-(((yy - 2) ** 2 + (xx - 2) ** 2) / 8.0))
    assert dip.local_dip(peak) == {'status': 'no dip'}
    assert not dip.dips(peak[None])[0].any()


def test_pure_peak_is_reported_as_no_dip_on_a_slide():
    """Counts higher under nuclei: no usable dip, no shift."""
    labels, counts, tissue = slide(shift=(-2, 1), dip_fraction=-0.5)
    result, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=100)
    assert not result['usable'] and result['dy'] is None and result['dx'] is None
    assert result['status'] in ('no dip', 'unstable')


def lattice_slide(width=360, spacing=36, radius=3, shift=(-2, 1), under=18, band=0):
    """Nuclei on a lattice, and noise-free counts that are `under` on the nuclei moved by `shift`
    and on a band of `band` locations around them, 12 elsewhere.

    The shift surface is flat away from the planted shift, so it has a peak and no dip at all.
    """
    labels = np.zeros((width, width), np.int32)
    reach = radius + band
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    disc = (yy ** 2 + xx ** 2) <= radius ** 2
    high = (yy ** 2 + xx ** 2) <= reach ** 2
    counts = np.full((width, width), 12.0)
    for k, (r, c) in enumerate(((r, c) for r in range(spacing, width - spacing, spacing)
                                for c in range(spacing, width - spacing, spacing)), start=1):
        labels[r - reach:r + reach + 1, c - reach:c + reach + 1][disc] = k
        moved = counts[r + shift[0] - reach:r + shift[0] + reach + 1, c + shift[1] - reach:c + shift[1] + reach + 1]
        moved[high] = under
    return labels, counts, np.ones((width, width), bool)


def test_fallbacks_are_experimental_and_off_by_default():
    labels, counts, tissue = lattice_slide()
    off, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60)
    assert off['status'] == 'no dip' and off['dy'] is None and 'fallback' not in off
    on, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60, fallback='peak')
    assert on['status'] == 'peak (experimental)' and on['method'] == 'peak (experimental)' and on['usable']
    assert (on['dy'], on['dx']) == (-2, 1) and on['peak_height'] > 0 and on['peak_share_within_one'] == 1


def test_fringe_fallback_finds_the_band_around_the_nuclei():
    """Counts high on the nuclei and on a band around them: the band's surface peaks at the shift."""
    labels, counts, tissue = lattice_slide(band=2)
    on, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60, fallback='fringe', fringe_width=2)
    assert on['status'] == 'fringe peak (experimental)' and on['fallback'] == 'fringe' and on['usable']
    assert (on['dy'], on['dx']) == (-2, 1) and on['fringe_width'] == 2


@pytest.mark.parametrize('fallback', ['peak', 'fringe'])
def test_fallbacks_are_not_used_where_there_is_a_usable_dip(fallback):
    labels, counts, tissue = slide(shift=(-2, 1), seed=7)
    off, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60)
    on, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60, fallback=fallback)
    assert on == off and on['method'] == 'dip'


def test_noise_dip_is_unstable_and_rejected():
    rng = np.random.default_rng(5)
    width = 360
    labels = np.zeros((width, width), np.int32)
    centres = rng.integers(20, width - 20, size=(900, 2))
    for k, (r, c) in enumerate(centres, start=1):
        labels[r - 2:r + 3, c - 2:c + 3] = k
    counts = rng.poisson(12.0, size=(width, width)).astype(float)
    result, _ = dip.assess(labels > 0, counts, np.ones((width, width), bool), search=8, draws=100)
    assert not result['usable'] and result['dy'] is None
    assert result['status'] in ('no dip', 'unstable')


def test_assessment_is_deterministic_in_the_seed():
    labels, counts, tissue = slide(shift=(-2, 1), seed=7)
    a, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60)
    b, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60)
    c, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60, seed=1)
    assert a == b
    assert (a['dy'], a['dx']) == (-2, 1) and a['status'] == 'usable'
    assert a['share_exact'] <= a['share_within_one'] <= a['share_any_dip'] <= 1
    assert a['depth_lo'] <= a['depth_hi']
    assert (c['dy'], c['dx']) == (-2, 1)


def test_block_sums_match_the_whole_surface():
    labels, counts, tissue = slide(width=260, n=300, shift=(1, 2), seed=4)
    nuc = labels > 0
    blocks = dip.blocks_for(tissue, (0, 260, 0, 260))
    stats = dip.block_sums(nuc, counts, tissue, blocks, 6)
    whole = (stats.sum(1)[:, 1] / stats.sum(1)[:, 0]).reshape(13, 13)
    assert np.allclose(whole, dip.shift_surface(nuc, counts, tissue, 6), rtol=0, atol=1e-12)


def test_dip_on_the_search_edge_is_not_usable():
    labels, counts, tissue = slide(shift=(-8, 0))
    result, _ = dip.assess(labels > 0, counts, tissue, search=8, draws=60)
    assert not result['usable'] and result['dy'] is None and result['status'] == 'unstable'


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
    rows, _ = dip.regions(labels > 0, counts, tissue, divisions=5, minimum=1500, search=8, draws=100)
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
    assert cli.main(['check', *args, '--divisions', '0', '--search', '8', '--draws', '100',
                     '--out', str(tmp_path / 'check.json')]) == 0
    result = json.loads((tmp_path / 'check.json').read_text())
    assert result['whole']['status'] == 'usable'
    assert (result['whole']['dy'], result['whole']['dx']) == (-2, -1)
    assert result['whole']['share_within_one'] >= 0.95 and result['whole']['ring_contrast_at_dip'] > 0
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
