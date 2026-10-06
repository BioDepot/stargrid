import importlib.util
import sys
from pathlib import Path

import numpy as np
import json
import pytest
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "score_hd_flex_registration_sensitivity.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("score_hd_flex_registration_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_transform_reuse_verifies_geometry_without_fitting(tmp_path):
    geometry = tmp_path / "geometry.json"
    geometry.write_text("{}")
    summary = tmp_path / "summary.json"
    identity = np.eye(3).tolist()
    summary.write_text(json.dumps({
        "inputs": {str(geometry): MODULE.sha256(geometry)},
        "transforms": {"grid_to_microscope": identity, "old_mask_grid_from_expression_grid": identity},
        "grid_to_microscope_fit": {"fit_points": 4},
    }))
    grid, fit, relative = MODULE.frozen_transforms(summary, (geometry,))
    np.testing.assert_array_equal(grid, np.eye(3))
    np.testing.assert_array_equal(relative, np.eye(3))
    assert fit["fit_points"] == 4
    geometry.write_text("changed")
    with pytest.raises(ValueError, match="input drift"):
        MODULE.frozen_transforms(summary, (geometry,))


def test_contrast_audit_measures_difference_of_differences():
    names = ["space_ranger", *(name for name, _ in MODULE.POLICIES)]
    def rows(value):
        return [dict(field=name, **{metric: value for metric, _ in MODULE.METRICS}) for name in names]
    baseline, native = rows(0.2), rows(0.3)
    assert all(row["absolute_change"] == 0 for row in MODULE.contrast_sensitivity(baseline, native))
    next(row for row in native if row["field"] == "postcollapse_hard")["in_cell_roc_auc"] = 0.4
    changes = MODULE.contrast_sensitivity(baseline, native)
    changed = [row for row in changes if row["absolute_change"] > 0]
    assert len(changed) == 1
    assert changed[0]["absolute_change"] == pytest.approx(0.1)


def test_non_nested_policies_preserve_fields_without_an_increment_cohort(monkeypatch):
    strict = {("g1", "s_002um_0_0"): 2.0}
    rescued = {("g1", "s_002um_0_0"): 1.0, ("g1", "s_002um_1_1"): 3.0}
    monkeypatch.setattr(MODULE, "load_space_ranger_h5", lambda *a: (strict, {"g1"}))
    monkeypatch.setattr(MODULE, "load_mex", lambda path, scale: strict if "strict" in path.parts else rescued)
    args = SimpleNamespace(vendor_h5=Path("vendor.h5"), policy_mex_root=Path("mex"),
                           width=2, height=2, skip_strict_increments=True)
    states = np.array([0, 1, 2, 2], dtype=np.uint8)
    rows = MODULE.build_fields(args, states)
    by_name = {row["field"]: row for row in rows}
    assert by_name["strict"]["raw_molecule_mass"] == 2.0
    assert by_name["postcollapse_hard"]["raw_molecule_mass"] == 4.0
    assert by_name["postcollapse_hard_star_only"]["raw_molecule_mass"] == 3.0
    assert by_name["postcollapse_hard_space_ranger_only"]["raw_molecule_mass"] == 1.0
    assert not any("increment" in name for name in by_name)
    contrasts = MODULE.contrast_sensitivity(rows, rows)
    assert all(row["absolute_change"] == 0 for row in contrasts)
    # The paper endpoint consumes whole-policy versus vendor contrasts only.
    extra = dict(rows[0], field="irrelevant_increment", raw_molecule_mass=123.0)
    assert MODULE.contrast_sensitivity(rows + [extra], rows + [extra]) == contrasts
    args.skip_strict_increments = False
    with pytest.raises(ValueError, match="strict mass is not a subset"):
        MODULE.build_fields(args, states)


def test_normalized_dlt_recovers_projective_grid_mapping():
    expected = np.array([
        [3.2, -0.1, 100.0],
        [0.2, 2.9, 50.0],
        [2e-5, -1e-5, 1.0],
    ])
    source = np.array([
        [0.0, 0.0], [100.0, 0.0], [0.0, 80.0], [100.0, 80.0],
        [20.0, 40.0], [70.0, 10.0], [55.0, 65.0], [5.0, 75.0],
    ])
    target = MODULE.project(expected, source)
    observed = MODULE.fit_homography(source, target)
    assert np.max(np.abs(MODULE.project(observed, source) - target)) < 1e-8


def test_relative_registration_reprojects_mask_without_changing_states():
    width, height = 5, 4
    # One grid bin equals two microscope pixels. The native image transform
    # moves raw x by +2 CytAssist pixels, so the old mask is sampled one grid
    # bin to the left for each fixed expression coordinate.
    grid_to_microscope = np.array([
        [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0],
    ])
    sr = np.eye(3)
    native = np.array([
        [1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    relative = MODULE.relative_mask_transform(
        grid_to_microscope, sr, native, native_moving_downsample=1.0,
    )
    states = np.array([
        0, 1, 2, 0, 0,
        0, 1, 2, 0, 0,
        0, 1, 2, 0, 0,
        0, 1, 2, 0, 0,
    ], dtype=np.uint8)
    remapped, metrics = MODULE.remap_compartments(
        states, relative, width=width, height=height,
    )
    assert remapped.reshape(height, width).tolist() == [
        [0, 0, 1, 2, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 1, 2, 0],
    ]
    assert metrics["median_displacement_2um_bins"] == 1.0
    assert metrics["out_of_bounds_bins"] == height


def test_identity_registration_is_exact_noop():
    states = np.array([0, 1, 2, 1, 0, 2], dtype=np.uint8)
    remapped, metrics = MODULE.remap_compartments(
        states, np.eye(3), width=3, height=2,
    )
    assert np.array_equal(remapped, states)
    assert metrics["changed_state_bins"] == 0
    assert metrics["maximum_displacement_um"] == 0.0


def test_sensitivity_rows_require_identical_mass_and_report_metric_delta():
    baseline = [{
        "field": "soft",
        "raw_molecule_mass": "10.0",
        **{metric: "0.25" for metric in MODULE.SENSITIVITY_METRICS},
    }]
    native = [{
        "field": "soft",
        "field_role": "whole_open_policy",
        "policy": "postcollapse_soft",
        "raw_molecule_mass": 10.0,
        **{metric: 0.5 for metric in MODULE.SENSITIVITY_METRICS},
    }]
    rows = MODULE.sensitivity_rows(baseline, native)
    assert len(rows) == len(MODULE.SENSITIVITY_METRICS)
    assert all(row["native_minus_space_ranger_registration"] == 0.25 for row in rows)

    native[0]["raw_molecule_mass"] = 10.5
    with np.testing.assert_raises_regex(ValueError, "changed expression mass"):
        MODULE.sensitivity_rows(baseline, native)
