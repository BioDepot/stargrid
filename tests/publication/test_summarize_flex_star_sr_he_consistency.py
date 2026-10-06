import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "summarize_flex_star_sr_he_consistency.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_flex_star_sr_he_consistency", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_dataset_requires_label_and_path():
    assert MODULE.parse_dataset("CRC=/tmp/crc") == ("CRC", Path("/tmp/crc"))
    try:
        MODULE.parse_dataset("/tmp/crc")
    except Exception as exc:
        assert "LABEL=COMPARISON_DIR" in str(exc)
    else:
        raise AssertionError("unlabeled dataset was accepted")


def test_segmentation_signature_ignores_slide_pixel_size():
    summary = {
        "input_contract": {"downsample": 2, "source_pixel_size_um": 0.25},
        "method": {
            "engine": "Cellpose",
            "engine_version": "4.1.1",
            "model": "cpsam",
            "channel": "hematoxylin",
            "nucleus_expansion_um": 2,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0,
            "min_size": 15,
            "max_size_fraction": 0.4,
            "tile_size": 2048,
            "overlap": 128,
            "cell_domain": "nearest-nucleus label expansion",
        },
    }
    signature = MODULE.segmentation_signature(summary)
    assert signature["downsample"] == 2
    assert "source_pixel_size_um" not in signature
