from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_spatch_qc_reference import mex_totals


def test_streamed_mex_uses_barcode_coordinates_and_keeps_zeros(tmp_path):
    (tmp_path/'barcodes.tsv').write_text('s_002um_1_1-1\ns_002um_0_0-1\ns_002um_0_1-1\n')
    (tmp_path/'matrix.mtx').write_text('%%MatrixMarket matrix coordinate integer general\n% example\n2 3 3\n1 1 7\n1 2 3\n2 2 2\n')
    counts, summary = mex_totals(tmp_path,width=2,block_bytes=5)
    np.testing.assert_array_equal(counts,[5,0,0,7])
    assert summary['entries']==3 and summary['molecules']==12
