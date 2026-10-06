from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from segment_hd_he_stardist import histogram_percentile, NormalizedImage, project


def test_histogram_percentiles_equal_full_image_percentiles():
    image = np.random.default_rng(123).integers(0, 256, size=(73, 129, 3), dtype=np.uint8)
    hist = np.bincount(image.ravel(), minlength=256)
    for percentile in [0, 5, 50, 95, 100]:
        assert histogram_percentile(hist, percentile) == np.percentile(image, percentile)


def test_lazy_normalization_preserves_global_percentiles():
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    low, high = np.percentile(image, [5, 95])
    lazy = NormalizedImage(image, low, high)
    expected = (image.astype(np.float32) - np.float32(low)) / (np.float32(high) - np.float32(low) + np.float32(1e-20))
    np.testing.assert_array_equal(lazy[1:3, 2:4], expected[1:3, 2:4])


def test_project_retains_quarter_turn_and_translation():
    matrix = np.array([[0, -2, 100], [2, 0, 50], [0, 0, 1]])
    np.testing.assert_array_equal(project(matrix, np.array([[0, 0], [3, 4]])), [[100, 50], [92, 56]])
