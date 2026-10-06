"""Non-transcriptomic spatial-fit metrics for natural Visium HD fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


COMPARTMENT_NAMES = {
    0: "extracellular",
    1: "cell_non_nucleus",
    2: "nucleus",
}


@dataclass(frozen=True)
class NaturalImageMasks:
    width: int
    height: int
    annotated: np.ndarray
    explicit_outside: np.ndarray
    compartments: np.ndarray

    def validate(self) -> None:
        size = self.width * self.height
        for name, values in (
            ("annotated", self.annotated),
            ("explicit_outside", self.explicit_outside),
            ("compartments", self.compartments),
        ):
            if values.size != size:
                raise ValueError(f"{name} dimensions differ from the HD grid")
        if np.any(np.asarray(self.explicit_outside, dtype=bool) & ~np.asarray(
            self.annotated, dtype=bool
        )):
            raise ValueError("explicit outside annotations must be annotated")
        if np.any(np.asarray(self.compartments) > max(COMPARTMENT_NAMES)):
            raise ValueError("unknown image compartment state")


def compare_natural_field(field: np.ndarray, masks: NaturalImageMasks) -> dict:
    """Compare one natural molecule field with image-only spatial masks."""
    masks.validate()
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    if values.size != masks.width * masks.height:
        raise ValueError("field dimensions differ from the HD grid")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("field values must be finite and non-negative")

    annotated = np.asarray(masks.annotated, dtype=bool).reshape(-1)
    explicit_outside = np.asarray(masks.explicit_outside, dtype=bool).reshape(-1)
    states = np.asarray(masks.compartments, dtype=np.uint8).reshape(-1)
    in_cell = states > 0
    in_nucleus = states == 2
    pathologist_tissue = annotated & ~explicit_outside
    total = float(np.sum(values, dtype=np.float64))

    tissue = {
        "total_molecule_mass": total,
        "occupied_bins": int(np.count_nonzero(values)),
        "zero_bin_fraction": float(np.mean(values == 0.0)),
        "pathologist_tissue": _mask_summary(values, pathologist_tissue),
        "annotated_any": _mask_summary(values, annotated),
        "unannotated": _mask_summary(values, ~annotated),
        "explicit_outside": _mask_summary(values, explicit_outside),
    }
    compartments = {
        COMPARTMENT_NAMES[state]: _mask_summary(values, states == state)
        for state in sorted(COMPARTMENT_NAMES)
    }
    classification = {
        "pathologist_tissue": binary_density_metrics(values, pathologist_tissue),
        "in_cell": binary_density_metrics(values, in_cell),
        "in_nucleus": binary_density_metrics(values, in_nucleus),
    }
    distribution_fit = {
        "pathologist_tissue": distribution_distance(
            values, pathologist_tissue.astype(np.float64)
        ),
        "in_cell": distribution_distance(values, in_cell.astype(np.float64)),
        "in_nucleus": distribution_distance(
            values, in_nucleus.astype(np.float64)
        ),
    }
    parent_fit = compare_parent_image_support(
        values,
        annotated,
        in_cell,
        in_nucleus,
        width=masks.width,
        height=masks.height,
    )
    edge = tissue_edge_metrics(
        values,
        pathologist_tissue,
        width=masks.width,
        height=masks.height,
    )
    return {
        "tissue_support": tissue,
        "compartment_support": compartments,
        "classification": classification,
        "distribution_fit": distribution_fit,
        "parent_image_support": parent_fit,
        "tissue_edge_support": edge,
    }


def _mask_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    bins = int(np.count_nonzero(mask))
    total_bins = values.size
    mass = float(np.sum(values[mask], dtype=np.float64))
    total_mass = float(np.sum(values, dtype=np.float64))
    area_fraction = bins / total_bins if total_bins else 0.0
    mass_fraction = mass / total_mass if total_mass else 0.0
    return {
        "bins": bins,
        "area_fraction": area_fraction,
        "molecule_mass": mass,
        "mass_fraction": mass_fraction,
        "area_normalized_enrichment": (
            mass_fraction / area_fraction if area_fraction else 0.0
        ),
        "mean_molecule_density": mass / bins if bins else 0.0,
    }


def binary_density_metrics(scores: np.ndarray, target: np.ndarray) -> dict:
    """Tie-aware ROC AUC and average precision without sklearn."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    if values.size != labels.size:
        raise ValueError("score and target dimensions differ")
    positives = int(np.count_nonzero(labels))
    negatives = labels.size - positives
    prevalence = positives / labels.size if labels.size else 0.0
    if positives == 0 or negatives == 0:
        return {
            "bins": int(labels.size),
            "positive_bins": positives,
            "prevalence": prevalence,
            "roc_auc": 0.5,
            "average_precision": prevalence,
        }

    order = np.argsort(values, kind="stable")
    sorted_scores = values[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0.0) + 1]
    ends = np.r_[starts[1:], values.size]
    group_positive = np.add.reduceat(sorted_labels.astype(np.int64), starts)
    group_size = ends - starts
    group_negative = group_size - group_positive

    negatives_before = np.cumsum(group_negative) - group_negative
    favorable = np.sum(
        group_positive * (negatives_before + 0.5 * group_negative),
        dtype=np.float64,
    )
    auc = float(favorable / (positives * negatives))

    descending_positive = group_positive[::-1]
    descending_size = group_size[::-1]
    cumulative_positive = np.cumsum(descending_positive)
    cumulative_size = np.cumsum(descending_size)
    precision = cumulative_positive / cumulative_size
    recall_increment = descending_positive / positives
    average_precision = float(np.sum(recall_increment * precision))
    return {
        "bins": int(labels.size),
        "positive_bins": positives,
        "prevalence": prevalence,
        "roc_auc": auc,
        "average_precision": average_precision,
    }


def distribution_distance(values: np.ndarray, support: np.ndarray) -> dict:
    """Compare normalized molecule mass with normalized image-support area."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.asarray(support, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("distribution dimensions differ")
    x_mass = float(np.sum(x, dtype=np.float64))
    y_mass = float(np.sum(y, dtype=np.float64))
    px = x / x_mass if x_mass else np.zeros_like(x)
    py = y / y_mass if y_mass else np.zeros_like(y)
    tv = float(0.5 * np.sum(np.abs(px - py), dtype=np.float64))
    midpoint = 0.5 * (px + py)
    jsd = 0.0
    positive_x = px > 0.0
    positive_y = py > 0.0
    if np.any(positive_x):
        jsd += 0.5 * float(np.sum(
            px[positive_x] * np.log(px[positive_x] / midpoint[positive_x]),
            dtype=np.float64,
        ))
    if np.any(positive_y):
        jsd += 0.5 * float(np.sum(
            py[positive_y] * np.log(py[positive_y] / midpoint[positive_y]),
            dtype=np.float64,
        ))
    return {
        "molecule_mass": x_mass,
        "image_support_bins": int(y_mass),
        "normalized_total_variation": tv,
        "jensen_shannon_divergence": jsd,
    }


def compare_parent_image_support(
    values: np.ndarray,
    annotated: np.ndarray,
    in_cell: np.ndarray,
    in_nucleus: np.ndarray,
    *,
    width: int,
    height: int,
    parent_size: int = 8,
) -> dict:
    """Measure 2 um structure against image masks within 16 um parents."""
    field_blocks = _parent_blocks(values, width, height, parent_size)
    annotated_blocks = _parent_blocks(
        annotated.astype(np.float64), width, height, parent_size
    )
    cell_blocks = _parent_blocks(
        in_cell.astype(np.float64), width, height, parent_size
    )
    nucleus_blocks = _parent_blocks(
        in_nucleus.astype(np.float64), width, height, parent_size
    )
    parent_mass = np.sum(field_blocks, axis=1, dtype=np.float64)
    valid_children = _parent_blocks(
        np.ones(values.size, dtype=np.float64), width, height, parent_size
    )
    valid_count = np.sum(valid_children, axis=1, dtype=np.float64)
    annotated_fraction = np.sum(annotated_blocks, axis=1) / valid_count
    cell_fraction = np.sum(cell_blocks, axis=1) / valid_count
    nucleus_fraction = np.sum(nucleus_blocks, axis=1) / valid_count
    molecule_density = parent_mass / valid_count
    annotated_parent = annotated_fraction > 0.0
    return {
        "all_parents": {
            "parents": int(parent_mass.size),
            "cell_fraction_pearson": _pearson(molecule_density, cell_fraction),
            "cell_fraction_spearman": _spearman(molecule_density, cell_fraction),
            "nucleus_fraction_pearson": _pearson(
                molecule_density, nucleus_fraction
            ),
            "nucleus_fraction_spearman": _spearman(
                molecule_density, nucleus_fraction
            ),
        },
        "annotated_parents": {
            "parents": int(np.count_nonzero(annotated_parent)),
            "cell_fraction_pearson": _pearson(
                molecule_density[annotated_parent], cell_fraction[annotated_parent]
            ),
            "cell_fraction_spearman": _spearman(
                molecule_density[annotated_parent], cell_fraction[annotated_parent]
            ),
            "nucleus_fraction_pearson": _pearson(
                molecule_density[annotated_parent],
                nucleus_fraction[annotated_parent],
            ),
            "nucleus_fraction_spearman": _spearman(
                molecule_density[annotated_parent],
                nucleus_fraction[annotated_parent],
            ),
        },
        "cell_supported_fine_scale": _fine_scale_support(
            field_blocks, cell_blocks, valid_children
        ),
        "nucleus_supported_fine_scale": _fine_scale_support(
            field_blocks, nucleus_blocks, valid_children
        ),
    }


def _fine_scale_support(
    field_blocks: np.ndarray,
    mask_blocks: np.ndarray,
    valid_blocks: np.ndarray,
) -> dict:
    supported_count = np.sum(mask_blocks, axis=1, dtype=np.float64)
    empty_count = np.sum(valid_blocks, axis=1, dtype=np.float64) - supported_count
    mixed = (supported_count > 0.0) & (empty_count > 0.0)
    fields = field_blocks[mixed]
    masks = mask_blocks[mixed] > 0.0
    supported_count = supported_count[mixed]
    empty_count = empty_count[mixed]
    parent_mass = np.sum(fields, axis=1, dtype=np.float64)
    occupied = parent_mass > 0.0
    fields = fields[occupied]
    masks = masks[occupied]
    supported_count = supported_count[occupied]
    empty_count = empty_count[occupied]
    parent_mass = parent_mass[occupied]
    if parent_mass.size == 0:
        return {
            "mixed_image_parents": int(np.count_nonzero(mixed)),
            "occupied_mixed_image_parents": 0,
            "global_supported_mass_fraction": 0.0,
            "global_supported_squared_mass_fraction": 0.0,
            "parent_equal_mean_supported_mass_fraction": 0.0,
            "parent_equal_mean_supported_minus_empty_density": 0.0,
            "parents_with_image_empty_unique_max_fraction": 0.0,
            "parents_image_empty_argmax_tie_fraction": 0.0,
        }
    supported_mass = np.sum(fields * masks, axis=1, dtype=np.float64)
    squared = fields * fields
    supported_squared = np.sum(squared * masks, axis=1, dtype=np.float64)
    total_squared = np.sum(squared, axis=1, dtype=np.float64)
    shares = supported_mass / parent_mass
    density_delta = (
        supported_mass / supported_count
        - (parent_mass - supported_mass) / empty_count
    ) / parent_mass
    maxima = np.max(fields, axis=1)
    max_mask = fields == maxima[:, None]
    unique_max = np.sum(max_mask, axis=1) == 1
    max_on_empty = np.any(max_mask & ~masks, axis=1) & unique_max
    max_count = np.sum(max_mask, axis=1, dtype=np.float64)
    empty_argmax_share = np.sum(max_mask & ~masks, axis=1, dtype=np.float64) / max_count
    return {
        "mixed_image_parents": int(np.count_nonzero(mixed)),
        "occupied_mixed_image_parents": int(parent_mass.size),
        "global_supported_mass_fraction": float(
            np.sum(supported_mass) / np.sum(parent_mass)
        ),
        "global_supported_squared_mass_fraction": float(
            np.sum(supported_squared) / np.sum(total_squared)
            if np.sum(total_squared) else 0.0
        ),
        "parent_equal_mean_supported_mass_fraction": float(np.mean(shares)),
        "parent_equal_mean_supported_minus_empty_density": float(
            np.mean(density_delta)
        ),
        "parents_with_image_empty_unique_max_fraction": float(
            np.mean(max_on_empty)
        ),
        "parents_image_empty_argmax_tie_fraction": float(
            np.mean(empty_argmax_share)
        ),
    }


def tissue_edge_metrics(
    values: np.ndarray,
    tissue: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, dict[str, float | int]]:
    """Report molecule mass in fixed signed-distance bands from tissue support."""
    image = np.asarray(tissue, dtype=bool).reshape(height, width)
    inside_um = distance_transform_edt(image) * 2.0
    outside_um = distance_transform_edt(~image) * 2.0
    signed = np.where(image, inside_um, -outside_um).reshape(-1)
    bands = {
        "inside_ge_64um": signed >= 64.0,
        "inside_16_to_64um": (signed >= 16.0) & (signed < 64.0),
        "inside_0_to_16um": (signed > 0.0) & (signed < 16.0),
        "outside_0_to_16um": (signed <= 0.0) & (signed > -16.0),
        "outside_16_to_64um": (signed <= -16.0) & (signed > -64.0),
        "outside_ge_64um": signed <= -64.0,
    }
    return {name: _mask_summary(values, mask) for name, mask in bands.items()}


def _parent_blocks(
    values: np.ndarray, width: int, height: int, parent_size: int
) -> np.ndarray:
    image = np.asarray(values, dtype=np.float64).reshape(height, width)
    parent_height = (height + parent_size - 1) // parent_size
    parent_width = (width + parent_size - 1) // parent_size
    padded = np.zeros(
        (parent_height * parent_size, parent_width * parent_size),
        dtype=np.float64,
    )
    padded[:height, :width] = image
    return (
        padded.reshape(parent_height, parent_size, parent_width, parent_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, parent_size * parent_size)
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size != x.size:
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    sorted_values = x[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
    ends = np.r_[starts[1:], x.size]
    ranks = np.empty(x.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks
