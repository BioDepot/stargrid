from pathlib import Path
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_hd_cell_matrices import reciprocal_matches, profile_correlations


def test_reciprocal_overlap_and_tie_break():
    ours = np.array([1, 1, 1, 1, 2, 2, 3, 3, 0])
    theirs = np.array([8, 8, 9, 9, 9, 9, 7, 7, 6])
    matches = reciprocal_matches(ours, theirs)
    assert matches[['ours', 'theirs']].values.tolist() == [[1, 8], [3, 7]]


def test_sparse_correlation_matches_dense_and_handles_constant_profiles():
    a = np.array([[1., 0, 2, 8], [1, 0, 0, 0], [0, 0, 0, 0]])
    b = np.array([[0., 1, 4, 7], [0, 0, 0, 2], [1, 2, 3, 4]])
    got = profile_correlations(sparse.csr_matrix(a), sparse.csr_matrix(b))
    for i in range(2):
        assert np.isclose(got[i], np.corrcoef(np.log1p(a[i]), np.log1p(b[i]))[0, 1])
    assert np.isnan(got[2])
