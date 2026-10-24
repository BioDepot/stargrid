"""Physical centre conventions, including a minimally determined homography."""
import numpy as np
import pandas as pd
import pytest
from stargrid_register import transform


@pytest.mark.parametrize('bin_size', [2, 8])
@pytest.mark.parametrize('with_barcodes', [False, True])
def test_bin_centres_recover_the_same_two_micrometre_transform(bin_size, with_barcodes):
    rows, cols = np.mgrid[0:4, 0:5]
    factor = bin_size / 2
    grid = np.column_stack([cols.ravel(), rows.ravel()]) * factor + (factor - 1) / 2
    true = np.array([[3.1, .04, 400.], [-.03, 3.2, 300.], [1e-5, -2e-5, 1.]])
    image = transform.apply_homography(true, grid)
    t = pd.DataFrame(dict(array_row=rows.ravel(), array_col=cols.ravel(),
                          pxl_col_in_fullres=image[:, 0], pxl_row_in_fullres=image[:, 1]))
    if with_barcodes:
        t['barcode'] = [f's_{bin_size:03d}um_{r}_{c}-1' for r, c in zip(rows.ravel(), cols.ravel())]
    h, residual = transform.spaceranger_grid_to_image(t, bin_size_um=None if with_barcodes else bin_size)
    assert residual < 1e-9
    np.testing.assert_allclose(transform.apply_homography(h, [[0, 0], [3349, 3349]]),
                               transform.apply_homography(true, [[0, 0], [3349, 3349]]), atol=1e-5, rtol=0)


def test_four_points_retain_the_homography_null_vector():
    source = np.array([[0., 0.], [0., 10.], [10., 0.], [10., 10.]])
    true = np.array([[2., .1, 30.], [.2, 3., 40.], [.001, -.002, 1.]])
    target = transform.apply_homography(true, source)
    h, residual = transform.fit_homography(source, target)
    assert residual < 1e-10
    np.testing.assert_allclose(h, true, rtol=0, atol=1e-10)


@pytest.mark.parametrize('barcodes,explicit', [
    (['s_002um_0_0-1', 's_008um_0_1-1'], None),
    (['invalid', 's_002um_0_1-1'], None),
    (['s_016um_0_0-1', 's_016um_0_1-1'], None),
    (['s_008um_0_0-1', 's_008um_0_1-1'], 2),
])
def test_rejects_ambiguous_or_unsupported_position_scales(barcodes, explicit):
    t = pd.DataFrame(dict(barcode=barcodes, array_row=[0, 0], array_col=[0, 1],
                         pxl_col_in_fullres=[0., 1.], pxl_row_in_fullres=[0., 0.]))
    with pytest.raises(ValueError):
        transform.spaceranger_grid_to_image(t, bin_size_um=explicit)
