import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "compare_hd_cellpose_space_ranger_masks_roi.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "compare_hd_cellpose_space_ranger_masks_roi", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_binary_mask_metrics_reports_confusion_and_dice() -> None:
    metrics = MODULE.binary_mask_metrics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert metrics["true_positive_bins"] == 1
    assert metrics["false_positive_bins"] == 1
    assert metrics["false_negative_bins"] == 1
    assert metrics["true_negative_bins"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["dice"] == 0.5
    assert metrics["query_minus_reference_fraction"] == 0.0


def test_binary_mask_metrics_rejects_empty_support() -> None:
    with pytest.raises(ValueError, match="positive reference and query support"):
        MODULE.binary_mask_metrics(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))


def test_roi_manifest_is_optional_for_complete_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--barcode-mappings", "barcode_mappings.parquet",
            "--cellpose-segmentation", "cellpose",
            "--tissue-positions", "tissue_positions.parquet",
            "--space-ranger-alignment-json", "final_alignment.json",
            "--native-registration-json", "registration.json",
            "--out-dir", "comparison",
        ],
    )
    assert MODULE.parse_args().roi_manifest is None
