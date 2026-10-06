from pathlib import Path
import sys
import h5py
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_hd_same_run_offsets import h5_totals
from measure_hd_global_offset_fft import ncc_map, summarise
from build_hd_cell_matrix import shift_labels


def test_h5_totals_handles_empty_and_trailing_empty_columns(tmp_path):
    path = tmp_path / 'counts.h5'
    with h5py.File(path, 'w') as f:
        m = f.create_group('matrix')
        m['data'] = [2, 3, 7]
        m['indptr'] = [0, 2, 2, 3, 3]
        m['barcodes'] = np.array(['s_002um_00000_00000-1', 's_002um_00000_00001-1', 's_002um_00001_00000-1', 's_002um_00001_00001-1'], dtype='S')
    np.testing.assert_array_equal(h5_totals(path, 2), [5, 0, 7, 0])


def test_offset_sign_is_nucleus_to_count():
    n = (np.random.default_rng(7).random((100, 100)) > .8).astype(float)
    c = shift_labels(n, 3, -2)
    support = np.zeros_like(n, bool)
    support[10:-10, 10:-10] = True
    result = summarise(ncc_map(n, c, support))
    assert result['peak_dx_um'] == -4
    assert result['peak_dy_um'] == 6
