from __future__ import annotations

import numpy as np

from star_spatial.hd_natural_image_oracle import (
    NaturalImageMasks,
    binary_density_metrics,
    compare_natural_field,
)


def test_natural_image_metrics_favor_image_supported_mass() -> None:
    width = height = 16
    annotated = np.zeros(width * height, dtype=bool)
    annotated[:] = True
    outside = np.zeros(width * height, dtype=bool)
    outside.reshape(height, width)[:, 12:] = True
    compartments = np.zeros(width * height, dtype=np.uint8)
    compartments.reshape(height, width)[:, :4] = 1
    compartments.reshape(height, width)[:, :2] = 2
    masks = NaturalImageMasks(width, height, annotated, outside, compartments)

    supported = np.zeros(width * height, dtype=np.float64)
    supported.reshape(height, width)[:, :2] = 4.0
    unsupported = np.zeros(width * height, dtype=np.float64)
    unsupported.reshape(height, width)[:, 12:] = 4.0

    good = compare_natural_field(supported, masks)
    bad = compare_natural_field(unsupported, masks)
    assert good["tissue_support"]["pathologist_tissue"]["mass_fraction"] == 1.0
    assert bad["tissue_support"]["pathologist_tissue"]["mass_fraction"] == 0.0
    assert good["classification"]["pathologist_tissue"]["roc_auc"] > 0.5
    assert good["classification"]["in_cell"]["roc_auc"] > 0.5
    assert bad["classification"]["in_cell"]["roc_auc"] < 0.5
    assert good["parent_image_support"]["cell_supported_fine_scale"][
        "global_supported_mass_fraction"
    ] == 1.0
    assert bad["parent_image_support"]["cell_supported_fine_scale"][
        "global_supported_mass_fraction"
    ] == 0.0


def test_binary_density_metrics_are_tie_aware() -> None:
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([False, True, False, True])
    metrics = binary_density_metrics(scores, labels)
    assert metrics["roc_auc"] == 0.5
    assert metrics["average_precision"] == 0.5


def test_fine_scale_argmax_diagnostic_splits_tied_maxima() -> None:
    width = height = 8
    annotated = np.ones(width * height, dtype=bool)
    outside = np.zeros(width * height, dtype=bool)
    compartments = np.zeros(width * height, dtype=np.uint8)
    compartments[0] = 1
    masks = NaturalImageMasks(width, height, annotated, outside, compartments)
    field = np.zeros(width * height, dtype=np.float64)
    field[0] = 1.0
    field[1] = 1.0

    support = compare_natural_field(field, masks)["parent_image_support"][
        "cell_supported_fine_scale"
    ]
    assert support["parents_with_image_empty_unique_max_fraction"] == 0.0
    assert support["parents_image_empty_argmax_tie_fraction"] == 0.5
