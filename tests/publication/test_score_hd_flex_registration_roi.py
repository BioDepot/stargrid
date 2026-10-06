import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "score_hd_flex_registration_roi.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("score_hd_flex_registration_roi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def complete_geometry() -> object:
    indices = np.arange(64, dtype=np.int64)
    states = np.zeros(64, dtype=np.uint8)
    states[:32] = 1
    states[:16] = 2
    return MODULE.RoiGeometry(
        indices=indices,
        states=states,
        parent_ids=np.zeros(64, dtype=np.int64),
        complete_parent_mask=np.array([True]),
        audit={},
    )


def test_roi_score_preserves_raw_mass_and_mask_overlap():
    geometry = complete_geometry()
    field = np.zeros(64, dtype=np.float64)
    field[:16] = 2.0
    field[16:32] = 1.0
    row = MODULE.score_field("test", "whole_open_policy", "hard", field, geometry)
    assert row["raw_molecule_mass"] == 48.0
    assert row["in_cell_mass_fraction"] == 1.0
    assert row["in_nucleus_mass_fraction"] == 2.0 / 3.0
    assert row["complete_16um_parents"] == 1
    assert row["equal_mass_normalization_applied"] is False


def test_sparse_compatibility_and_increment_conserve_mass():
    star = sparse.csr_matrix(np.array([[3.0, 0.0], [0.0, 2.0]]))
    vendor = sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))
    shared, star_only, vendor_only = MODULE.compatibility_fields(star, vendor)
    assert shared.tolist() == [2.0, 2.0]
    assert star_only.tolist() == [1.0, 0.0]
    assert vendor_only.tolist() == [0.0, 2.0]
    assert shared.sum() + star_only.sum() == star.sum()
    assert shared.sum() + vendor_only.sum() == vendor.sum()

    policy = sparse.csr_matrix(np.array([[3.0, 1.0], [0.0, 2.0]]))
    increment = MODULE.increment_field(policy, star)
    assert increment.tolist() == [0.0, 1.0]


def test_vendor_mass_on_unobserved_star_rows_is_reported_not_dropped():
    vendor = sparse.csr_matrix(np.array([
        [3.0, 0.0],
        [0.0, 11.0],
        [0.0, 0.0],
    ]))
    missing, mass = MODULE.absent_star_feature_mass(
        ["g1"], {"g1": 0, "g2": 1, "g3": 2}, vendor,
    )
    assert missing == ["g2", "g3"]
    assert mass == 11.0


def test_increment_rejects_loss_of_strict_feature_bin_mass():
    strict = sparse.csr_matrix(np.array([[2.0, 0.0]]))
    policy = sparse.csr_matrix(np.array([[1.0, 0.0]]))
    with np.testing.assert_raises_regex(ValueError, "loses strict"):
        MODULE.increment_field(policy, strict)


def test_registration_to_segmentation_coordinate_tracks_pixel_centres():
    moving = np.array([0.0, 1.0, 10.5])
    # The frozen ROI registration image was an area-resized ds4 crop; Cellpose
    # is run on an area-resized ds2 crop of the same original pixel bounds.
    observed = MODULE.registration_to_segmentation_coordinate(
        moving,
        moving_downsample=4,
        moving_sampling="area",
        moving_source_origin=4608,
        segmentation_source_origin=4608,
        segmentation_downsample=2,
    )
    assert np.allclose(observed, [0.5, 2.5, 21.5])


def test_decimated_registration_coordinate_is_not_treated_as_area_resize():
    observed = MODULE.registration_to_segmentation_coordinate(
        np.array([0.0, 1.0]),
        moving_downsample=16,
        moving_sampling="decimate",
        moving_source_origin=0,
        segmentation_source_origin=0,
        segmentation_downsample=2,
    )
    assert np.allclose(observed, [-0.25, 7.75])


def test_independent_capture_grid_selects_row_major_centres():
    indices, audit = MODULE.select_native_capture_grid_roi(
        np.eye(3), np.eye(3), None,
        moving_shape_yx=(3, 4), fixed_shape_yx=(3, 4),
        width=4, height=3,
    )
    assert indices.tolist() == list(range(12))
    assert audit["geometry_source"] == "independent_capture_grid"
    assert audit["native_roi_2um_bins"] == 12


def test_independent_capture_grid_respects_projective_transform_and_extent():
    spot_to_cytassist = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    indices, _ = MODULE.select_native_capture_grid_roi(
        spot_to_cytassist, np.eye(3), None,
        moving_shape_yx=(3, 4), fixed_shape_yx=(10, 10),
        width=4, height=3,
    )
    assert indices.tolist() == [0, 1, 4, 5]


def test_publication_source_rejects_reference_scored_capture_grid():
    source = SCRIPT.read_text()
    assert "publication capture-grid JSON must be generated without a reference alignment" in source


def test_observed_star_axis_retains_vendor_only_reference_mass():
    source = SCRIPT.read_text()
    assert '"absent_feature_mass_is_retained_in_vendor_fields": True' in source
    assert "Space Ranger has {missing_mass} ROI mass on features absent from STAR" not in source
