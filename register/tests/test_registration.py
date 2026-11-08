"""Synthetic tests: a known displacement between nuclei and the pattern of counts around them is recovered."""
import json

import numpy as np
import pandas as pd
import pytest

from stargrid_register import affine, cli, dip, io, methods, ratio, transform


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


def test_peaks_finds_the_highest_peak():
    yy, xx = np.mgrid[-13:14, -13:14]
    surface = 1 + 0.3 * np.exp(-((yy - 2) ** 2 + (xx + 1) ** 2) / 2.0) + 0.1 * np.exp(-((yy + 5) ** 2 + (xx - 5) ** 2) / 2.0)
    found, dy, dx, height = dip.peaks(surface[None])
    assert found[0] and (dy[0], dx[0]) == (2, -1)
    assert height[0] == pytest.approx(surface[15, 12] / np.median(np.delete(surface[14:17, 11:14].ravel(), 4)) - 1)


def lattice_slide(width=360, spacing=36, radius=3, shift=(-2, 1), under=18, band=0, around=None):
    """Nuclei on a lattice, and noise-free counts that are `under` on the nuclei moved by `shift`
    and `around` (default `under`) on a band of `band` locations around them, 12 elsewhere.

    With `around` equal to `under` the mean count under the nuclei has a peak and no dip at all.
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
        moved[high] = under if around is None else around
        moved[disc] = under
    return labels, counts, np.ones((width, width), bool)


def test_block_sums_match_the_whole_surface():
    labels, counts, tissue = slide(width=260, n=300, shift=(1, 2), seed=4)
    nuc = labels > 0
    blocks = dip.blocks_for(tissue, (0, 260, 0, 260))
    stats = dip.block_sums(nuc, counts, tissue, blocks, 6)
    whole = (stats.sum(1)[:, 1] / stats.sum(1)[:, 0]).reshape(13, 13)
    assert np.allclose(whole, dip.shift_surface(nuc, counts, tissue, 6), rtol=0, atol=1e-12)


def test_band_labels_follow_the_distance_from_the_nucleus_edge():
    nuc = np.zeros((21, 21), bool)
    nuc[5:16, 5:16] = True
    labels = ratio.band_labels(nuc)
    assert labels[10, 10] == 0 and labels[6, 10] == 1 and labels[5, 10] == 2      # 6, 2 and 1 from the edge
    assert [labels[10, 15 + k] for k in range(1, 6)] == [3, 4, 5, 6, 7]            # 1 to 5 outside
    assert labels[0, 0] == -1 and labels[4, 4] == 4                                 # far; sqrt(2) outside


def test_band_sums_match_direct_sums():
    """Each supported location in a block counts under the band that the shift brings onto it."""
    labels, counts, tissue = slide(width=140, n=60, shift=(1, -2), seed=2)
    tissue[:, :9] = False
    bands, values = ratio.band_labels(labels > 0), ratio.corrected(counts, tissue)
    bounds = (20, 120, 10, 130)
    blocks = dip.blocks_for(tissue, bounds, block=50, margin=5)
    stats, totals = ratio.band_sums(bands, values, tissue, blocks, radius=5)
    ys, xs = np.mgrid[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    inside = tissue[ys, xs]
    v = values[ys, xs][inside]
    assert np.allclose(totals.sum(0), [inside.sum(), v.sum(), (v ** 2).sum()])
    for dy, dx in [(0, 0), (1, -2), (-5, 4)]:
        source = bands[ys - dy, xs - dx]
        k = (5 + dy) * 11 + 5 + dx
        for b in range(len(ratio.BANDS)):
            hit = inside & (source == b)
            assert stats[k, :, b, 0].sum() == hit.sum()
            assert stats[k, :, b, 1].sum() == pytest.approx(values[ys, xs][hit].sum(), rel=1e-12, abs=1e-9)


@pytest.mark.parametrize('form', ['dip', 'ring', 'disk'])
def test_ratio_finds_each_form_of_the_pattern(form):
    """Fewer counts under the nuclei (the dip), more just around them (the ring), or more under the
    nuclei and their ring (the disk, with no dip): one measure finds the shift in each."""
    if form == 'dip':
        labels, counts, tissue = slide(shift=(-2, 1), seed=7)
    elif form == 'ring':
        labels, counts, tissue = lattice_slide(band=3, under=12, around=18)
    else:
        labels, counts, tissue = lattice_slide(band=3)
    result, surface = ratio.assess(labels > 0, counts, tissue, search=8, draws=60)
    assert result['status'] == 'usable' and result['method'] == 'correlation ratio'
    assert (result['dy'], result['dx']) == (-2, 1) and result['share_within_one'] >= 0.95
    assert result['eta2'] == pytest.approx(np.nanmax(surface[1:-1, 1:-1])) and result['eta2'] > result['eta2_at_zero']
    profile = result['profile']
    assert list(profile) == list(ratio.BANDS) + ['far']
    if form == 'dip':
        assert profile['I3'] < profile['O2']
    elif form == 'ring':
        assert profile['O2'] > profile['I3'] and profile['O2'] > profile['far']
    else:                         # within the disk the counts are level; the contrast is at its edge
        assert profile['O3'] > profile['far'] > profile['O4']


def test_noise_is_not_registered():
    rng = np.random.default_rng(5)
    width = 360
    labels = np.zeros((width, width), np.int32)
    for k, (r, c) in enumerate(rng.integers(20, width - 20, size=(900, 2)), start=1):
        labels[r - 2:r + 3, c - 2:c + 3] = k
    counts = rng.poisson(12.0, size=(width, width)).astype(float)
    result, _ = ratio.assess(labels > 0, counts, np.ones((width, width), bool), search=8, draws=100)
    assert not result['usable'] and result['dy'] is None and result['status'] == 'not registered'


def test_shift_on_the_search_edge_is_not_usable():
    labels, counts, tissue = slide(shift=(-8, 0))
    result, _ = ratio.assess(labels > 0, counts, tissue, search=8, draws=60)
    assert not result['usable'] and result['dy'] is None and (result['candidate_dy'], result['candidate_dx']) == (-8, 0)
    assert result['status'] == 'not registered' and result['reason'] == 'highest maximum on the edge of the search'


def test_assessment_is_deterministic_in_the_seed():
    labels, counts, tissue = slide(shift=(-2, 1), seed=7)
    a, _ = ratio.assess(labels > 0, counts, tissue, search=8, draws=60)
    b, _ = ratio.assess(labels > 0, counts, tissue, search=8, draws=60)
    c, _ = ratio.assess(labels > 0, counts, tissue, search=8, draws=60, seed=1)
    assert a == b
    assert a['share_exact'] <= a['share_within_one'] <= a['share_any_peak'] <= 1
    assert (c['dy'], c['dx']) == (-2, 1)


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
    rows, _ = ratio.regions(labels > 0, counts, tissue, divisions=5, minimum=1500, search=8, draws=100)
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
                     '--surface', str(tmp_path / 'surface.npy'), '--out', str(tmp_path / 'check.json')]) == 0
    result = json.loads((tmp_path / 'check.json').read_text())
    whole = result['whole']
    assert whole['status'] == 'usable' and whole['method'] == 'correlation ratio' and result['baseline'] == 2
    assert (whole['dy'], whole['dx']) == (-2, -1) and whole['share_within_one'] >= 0.95
    assert whole['offset_um'] == pytest.approx(2 * np.hypot(2, 1))
    assert np.load(tmp_path / 'surface.npy').shape == (19, 19)
    assert cli.main(['check', *args, '--divisions', '0', '--search', '8', '--draws', '100', '--baseline', '4',
                     '--out', str(tmp_path / 'check4.json')]) == 0
    result = json.loads((tmp_path / 'check4.json').read_text())
    assert result['baseline'] == 4 and (result['whole']['dy'], result['whole']['dx']) == (-2, -1)
    assert cli.main(['refine', *args, '--radius', '6', '--out', str(tmp_path / 'refine.json')]) == 0
    assert json.loads((tmp_path / 'refine.json').read_text())['integer_dy'] == -2
    assert cli.main(['apply', '--nuclei', str(tmp_path / 'nuclei.npz'), '--width', '360', '--shift=-2,-1',
                     '--out-grid', str(tmp_path / 'moved.npz'), '--out', str(tmp_path / 'apply.json')]) == 0
    moved = io.load_grid(tmp_path / 'moved.npz', 360)
    again, _ = ratio.assess(moved > 0, counts, tissue, search=8, draws=60)
    assert (again['dy'], again['dx']) == (0, 0)


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
