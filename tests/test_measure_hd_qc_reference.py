from pathlib import Path
import gzip
import json
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import measure_hd_qc_reference as qc


def direct(nuc, counts, tissue, radius):
    h, w = nuc.shape
    result = np.empty((2*radius+1, 2*radius+1))
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            ys, xs = slice(max(dy, 0), h+min(dy, 0)), slice(max(dx, 0), w+min(dx, 0))
            yo, xo = slice(max(-dy, 0), h+min(-dy, 0)), slice(max(-dx, 0), w+min(-dx, 0))
            mask = nuc[yo, xo] & tissue[ys, xs]
            result[dy+radius, dx+radius] = counts[ys, xs][mask].mean() if mask.any() else np.nan
    return result


def test_fft_matches_all_nonwrapping_direct_means_exactly():
    rng = np.random.default_rng(91)
    n = rng.random((65, 71)) > .75
    t = rng.random(n.shape) > .15
    c = rng.integers(0, 50, n.shape).astype(float)
    np.testing.assert_array_equal(qc.shift_surface(n, c, t), direct(n, c, t, 13))


@pytest.mark.parametrize('direction,label', [(-1, 'dip'), (1, 'peak')])
def test_planted_dip_and_peak_on_grid(direction, label):
    rng = np.random.default_rng(61)
    n = rng.random((150, 150)) > .8
    n[:15] = False; n[-15:] = False; n[:, :15] = False; n[:, -15:] = False
    shifted = np.roll(n, (2, -1), (0, 1))
    counts = 20 + direction*8*shifted.astype(float)
    surface = qc.shift_surface(n, counts, np.ones_like(n))
    f = qc.feature(surface)
    assert (f['dy'], f['dx'], f['sign']) == (2, -1, label)
    assert f['width_um'] == 2


def test_sloping_surface_feature_is_not_corner_minimum():
    yy, xx = np.mgrid[-13:14, -13:14]
    surface = 20 + .4*(yy-xx) - 3*np.exp(-(yy**2+xx**2)/2)
    assert np.unravel_index(np.argmin(surface[5:22, 5:22]), (17, 17)) == (0, 16)
    f = qc.feature(surface)
    assert (f['dy'], f['dx']) == (0, 0)
    assert f['contrast'] > .1


def test_no_nucleus_mask_still_has_off_tissue_measures():
    tissue = np.zeros((60, 60), bool); tissue[10:50, 10:50] = True
    counts = np.where(tissue, 10., 1.)
    summary, off, near = qc.edge_profiles(counts, tissue)
    assert summary['off_to_in_mean_ratio'] == .1
    assert summary['share_of_counts_off_tissue'] == 2000/18000
    assert len(off) == len(qc.legacy.OFF)
    assert near == []


def test_partial_counts_never_report_off_tissue_zero_as_measure():
    t = np.ones((20, 20), bool)
    summary, off, _ = qc.edge_profiles(np.ones(t.shape), t, complete=False)
    assert 'share_of_counts_off_tissue' not in summary and off == []
    assert 'q1_q2_unavailable_reason' in summary


def test_input_h5_npz_and_gzipped_parquet(tmp_path):
    names = np.array([f's_002um_{r:05}_{c:05}-1' for r in range(2) for c in range(2)], dtype='S')
    p = tmp_path/'raw_feature_bc_matrix.h5'
    with h5py.File(p, 'w') as h:
        m = h.create_group('matrix')
        m['data'] = [2, 3, 7]; m['indptr'] = [0, 2, 2, 3, 3]; m['barcodes'] = names
    expected = np.array([[5., 0], [7, 0]])
    np.testing.assert_array_equal(qc.load_counts({'kind':'h5', 'path':str(p)}, 2), expected)
    np.savez(tmp_path/'counts.npz', total=expected.ravel())
    np.testing.assert_array_equal(qc.load_counts({'kind':'npz', 'path':str(tmp_path/'counts.npz')}, 2), expected)
    tissue = pd.DataFrame({'barcode': names.astype(str), 'array_row':[0,0,1,1],
                          'array_col':[0,1,0,1], 'in_tissue':[0,1,1,0]})
    raw = tmp_path/'positions.parquet'; tissue.to_parquet(raw)
    gz = tmp_path/'positions.parquet.gz'
    with gzip.open(gz, 'wb') as h: h.write(raw.read_bytes())
    np.testing.assert_array_equal(qc.tissue_mask(gz, 2), [[False,True],[True,False]])
    mapping = pd.DataFrame({'square_002um':names.astype(str), 'in_nucleus':[True,None,False,True]})
    raw = tmp_path/'nuclei.parquet'; mapping.to_parquet(raw)
    gz = tmp_path/'nuclei.parquet.gz'
    with gzip.open(gz, 'wb') as h: h.write(raw.read_bytes())
    np.testing.assert_array_equal(qc.published_nuclei(gz, 2), [[True,False],[False,True]])
    tissue['barcode'] = tissue.barcode.str.replace('002um','008um'); tissue.to_parquet(raw)
    with pytest.raises(ValueError, match='invalid 2 um barcode'):
        qc.tissue_mask(raw, 2)


def test_zero_eligible_regions_is_unresolved():
    n = np.zeros((50,50), bool); n[20:30,20:30] = True
    counts = 20 - 8*n.astype(float); tissue = np.ones_like(n)
    f = qc.feature(qc.shift_surface(n, counts, tissue))
    regions, arrays, result = qc.region_features(n, counts, tissue, f)
    assert result['resolved'] is False and result['eligible_regions'] == 0
    assert result['resolved_position_squares'] is None
    assert len(regions) == 9 and arrays == {}


def test_cli_without_nucleus_is_complete_and_refuses_repeat(tmp_path, monkeypatch):
    width = 40
    rr, cc = np.indices((width,width))
    tissue = pd.DataFrame({'array_row':rr.ravel(),'array_col':cc.ravel(),
                          'in_tissue':((rr>5)&(rr<35)&(cc>5)&(cc<35)).astype(int).ravel()})
    tissue.to_parquet(tmp_path/'tissue.parquet')
    np.savez(tmp_path/'counts.npz', total=np.ones(width*width))
    config = {'slide':'synthetic','width':width,'tissue_positions':str(tmp_path/'tissue.parquet'),
              'counts':{'kind':'npz','path':str(tmp_path/'counts.npz')},'counts_complete':True}
    (tmp_path/'config.json').write_text(json.dumps(config))
    monkeypatch.setenv('OFF_BENCH_CPUS_USED','synthetic-test')
    monkeypatch.setattr(sys,'argv',['measure', '--config',str(tmp_path/'config.json'),'--out-dir',str(tmp_path/'out')])
    assert qc.main() == 0
    summary = json.loads((tmp_path/'out/summary.json').read_text())
    assert summary['maps'] == {} and summary['primary_nucleus_map'] is None
    assert summary['off_tissue_profile'] and len(summary['inputs']) == 3
    with pytest.raises(FileExistsError): qc.main()
