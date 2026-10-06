import importlib.util
import argparse
import sys
import types
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "segment_hd_he_cellpose.py"
)
SPEC = importlib.util.spec_from_file_location("segment_hd_he_cellpose", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hematoxylin_contrast_prefers_blue_nucleus_like_pixel():
    image = np.array([[[70, 60, 125], [240, 200, 210]]], dtype=np.uint8)
    contrast = MODULE.hematoxylin_contrast(image)
    assert contrast.shape == (1, 2)
    assert contrast[0, 0] > contrast[0, 1]


def profile_args(profile: str, **overrides):
    values = {
        "profile": profile,
        "model": None,
        "model_file": Path("/tmp/cpsam"),
        "model_sha256": "expected",
        "downsample": None,
        "tile_size": None,
        "overlap": None,
        "nucleus_expansion_um": None,
        "flow_threshold": None,
        "cellprob_threshold": None,
        "min_size": None,
        "max_size_fraction": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_independent_profile_resolves_frozen_primary_parameters():
    args = MODULE.resolve_profile_args(profile_args("independent_v1"))
    assert args.model == "cpsam"
    assert args.flow_threshold == 0.4
    assert args.nucleus_expansion_um == 2.0
    assert args.tile_size == 2048
    assert args.overlap == 128


def test_sr_compat_profile_is_named_supplementary_parameter_set():
    args = MODULE.resolve_profile_args(profile_args("sr_compat_v1"))
    assert args.flow_threshold == 0.8
    assert args.nucleus_expansion_um == 8.0


def test_named_profile_rejects_parameter_drift():
    with np.testing.assert_raises_regex(ValueError, "conflicts with profile"):
        MODULE.resolve_profile_args(
            profile_args("independent_v1", flow_threshold=0.8)
        )


def test_local_cellpose_adapter_forwards_frozen_parameters(monkeypatch):
    observed = {}

    class FakeModel:
        def __init__(self, **kwargs):
            observed["constructor"] = kwargs

        def eval(self, image, **kwargs):
            observed["shape"] = image.shape
            observed["eval"] = kwargs
            return np.ones(image.shape, dtype=np.uint32), None, None

    fake_cellpose = types.ModuleType("cellpose")
    fake_cellpose.models = types.SimpleNamespace(CellposeModel=FakeModel)
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    predictor = MODULE.build_cellpose_predictor(
        "cpsam", diameter=None, flow_threshold=0.4,
        cellprob_threshold=0.0, min_size=15, max_size_fraction=0.4,
        invert=False, normalize=True, gpu=True,
    )
    result = predictor(np.zeros((5, 7), dtype=np.float32))
    assert result.shape == (5, 7)
    assert observed["constructor"] == {"gpu": True, "pretrained_model": "cpsam"}
    assert observed["eval"]["channel_axis"] is None
    assert observed["eval"]["flow_threshold"] == 0.4
    assert observed["eval"]["max_size_fraction"] == 0.4


def test_irregular_tile_interiors_cover_each_pixel_exactly_once():
    width, height = 2500, 1800
    coverage = np.zeros((height, width), dtype=np.uint8)
    rows = MODULE.tile_specs(width, height, tile_size=1000, overlap=100)
    for row in rows:
        interior = MODULE.tile_interior(row, width, height, overlap=100)
        coverage[
            interior["global_y0"]:interior["global_y1"],
            interior["global_x0"]:interior["global_x1"],
        ] += 1
    assert coverage.min() == 1
    assert coverage.max() == 1


def test_expansion_streaming_preserves_nucleus_hierarchy():
    source = np.zeros((25, 13), dtype=np.uint32)
    source[12, 6] = 7
    target = np.zeros_like(source)

    MODULE.expand_labels_streamed(source, target, 2.0)
    assert target[12, 6] == 7
    assert np.count_nonzero(target) == 13
    assert not np.any((source > 0) & (target == 0))
