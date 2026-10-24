"""Boundary and coordinate-frame tests for the explicit archived convention."""
import json
import numpy as np
import pandas as pd
import pytest
from stargrid_register import cli, transform


def test_negative_and_half_pixel_boundaries_keep_default_nearest():
    labels = np.arange(1, 13).reshape(3, 4)
    h = [[1, 0, -.5], [0, 1, .5], [0, 0, 1]]
    np.testing.assert_array_equal(transform.sample_labels(labels, h, 3),
                                  [[1, 1, 3], [9, 9, 11], [9, 9, 11]])
    np.testing.assert_array_equal(transform.sample_labels(labels, h, 3, sampling='archived-floor'),
                                  [[0, 1, 2], [0, 5, 6], [0, 9, 10]])


def test_explicit_frame_preserves_pixel_centre_offset():
    labels = np.arange(1, 17).reshape(4, 4)
    frame = [[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]]
    h = [[2, 0, 1], [0, 2, 1], [0, 0, 1]]
    np.testing.assert_array_equal(
        transform.sample_labels(labels, h, 3, sampling='archived-floor', image_to_labels=frame),
        labels[:3, :3])
    with pytest.raises(ValueError, match='not both'):
        transform.sample_labels(labels, h, 3, .5, image_to_labels=frame)
    with pytest.raises(ValueError, match='sampling'):
        transform.sample_labels(labels, h, 3, sampling='unknown')
    with pytest.raises(ValueError, match='3 x 3'):
        transform.sample_labels(labels, h, 3, image_to_labels=np.eye(2))


@pytest.mark.parametrize('bin_size', [2, 8])
@pytest.mark.parametrize('sampling', ['nearest', 'archived-floor'])
def test_cli_placement_keeps_bin_centres_and_explicit_frame(tmp_path, bin_size, sampling):
    rows, cols = np.mgrid[:4, :4]
    factor = bin_size / 2
    grid = np.column_stack([cols.ravel(), rows.ravel()]) * factor + (factor - 1) / 2
    xy = transform.apply_homography([[2, 0, 11.5], [0, 2, 21.5], [0, 0, 1]], grid)
    positions = pd.DataFrame(dict(array_row=rows.ravel(), array_col=cols.ravel(),
        pxl_col_in_fullres=xy[:, 0], pxl_row_in_fullres=xy[:, 1],
        barcode=[f's_{bin_size:03d}um_{r}_{c}-1' for r, c in zip(rows.ravel(), cols.ravel())]))
    positions.to_parquet(tmp_path/'positions.parquet')
    labels = np.arange(1, 26).reshape(5, 5)
    np.save(tmp_path/'labels.npy', labels)
    frame = [[.5, 0, -5], [0, .5, -10], [0, 0, 1]]
    (tmp_path/'frame.json').write_text(json.dumps(frame))
    cli.main(['place', '--tissue-positions', str(tmp_path/'positions.parquet'),
              '--labels', str(tmp_path/'labels.npy'), '--image-to-labels', str(tmp_path/'frame.json'),
              '--sampling', sampling, '--width', '3', '--out-grid', str(tmp_path/'grid.npz'),
              '--out', str(tmp_path/'result.json')])
    start = 0 if sampling == 'archived-floor' else 1
    with np.load(tmp_path/'grid.npz') as result:
        np.testing.assert_array_equal(result['labels'], labels[start:start+3, start:start+3])
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['sampling'] == sampling
    assert result['image_to_labels'] == frame
