#!/usr/bin/env python3
"""Score CRC Flex policies on a native-registration microscope ROI.

The ROI is defined in the original microscope image.  Expression-grid points
are projected through either the independent capture-grid geometry or the
legacy retained slide geometry and the independently fitted
microscope-to-CytAssist transform.  STAR and Space Ranger matrices are then
restricted to the identical set of 2 um bins.  Raw molecule mass is never
normalized between policies.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import io as scipy_io
from scipy import sparse
from scipy.stats import rankdata

from compare_hd_flex_policy_matrices import (
    load_space_ranger_sparse,
    mex_file,
    open_text,
    read_lines,
)
from score_hd_flex_full_compartments import (
    POLICIES,
    comparison_rows,
    write_rows,
)
from star_spatial.hd_natural_image_oracle import (
    binary_density_metrics,
    distribution_distance,
)


@dataclass(frozen=True)
class RoiGeometry:
    indices: np.ndarray
    states: np.ndarray
    parent_ids: np.ndarray
    complete_parent_mask: np.ndarray
    audit: dict[str, object]


def parquet_file(path: Path):
    """Import the optional retained-vendor geometry dependency only on use."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-mex-root", type=Path, required=True)
    parser.add_argument("--vendor-h5", type=Path, required=True)
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument("--barcode-mappings", type=Path)
    mask_group.add_argument(
        "--cellpose-segmentation", type=Path,
        help="Output root from segment_hd_he_cellpose.py.",
    )
    geometry_group = parser.add_mutually_exclusive_group(required=True)
    geometry_group.add_argument(
        "--capture-grid-json", type=Path,
        help="Independent visium_hd_capture_grid output; preferred publication path.",
    )
    geometry_group.add_argument(
        "--tissue-positions", type=Path,
        help="Legacy retained-Space-Ranger geometry path for controlled sensitivity only.",
    )
    parser.add_argument(
        "--space-ranger-alignment-json", type=Path,
        help="Required only with --tissue-positions.",
    )
    parser.add_argument("--native-registration-json", type=Path, required=True)
    parser.add_argument(
        "--roi-manifest", type=Path,
        help="Optional frozen microscope ROI manifest; omit to score the full registered image extent.",
    )
    parser.add_argument(
        "--registration-moving-source-downsample", type=int, default=4,
        help="Original-image downsample used to build the native registration moving image.",
    )
    parser.add_argument(
        "--registration-moving-sampling", choices=("area", "decimate"), default="area",
        help="Pixel-coordinate convention used to build the native moving image.",
    )
    parser.add_argument("--umi-mode", choices=("1mm_cr", "exact"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3350)
    parser.add_argument("--height", type=int, default=3350)
    parser.add_argument("--parent-size", type=int, default=8)
    parser.add_argument("--slide", default="H1-GMHFWPH")
    parser.add_argument("--area", default="D1")
    parser.add_argument(
        "--omit-membership", action="store_true",
        help="Do not materialize the diagnostic per-bin membership TSV (recommended at full-slide scale).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def matrix_from_json(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    value: object = json.loads(path.read_text())
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing JSON matrix key {'.'.join(keys)} in {path}")
        value = value[key]
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid 3x3 matrix at {'.'.join(keys)} in {path}")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError(f"singular matrix at {'.'.join(keys)} in {path}")
    return matrix / matrix[2, 2]


def project(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack((x, y, np.ones(len(x), dtype=np.float64)))
    mapped = matrix @ points
    if np.any(np.abs(mapped[2]) < 1e-12):
        raise ValueError("transform projects an expression-grid point to infinity")
    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def select_native_roi(
    tissue_positions: Path,
    sr_microscope_to_cytassist: np.ndarray,
    native_moving_to_fixed: np.ndarray,
    roi_manifest: dict[str, object] | None,
    moving_shape_yx: tuple[int, int],
    fixed_shape_yx: tuple[int, int],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if roi_manifest is None:
        fixed_height, fixed_width = fixed_shape_yx
        cyt_bounds = (0.0, 0.0, float(fixed_width), float(fixed_height))
        raw_bounds = None
    else:
        geometry = roi_manifest["geometry"]
        if not isinstance(geometry, dict):
            raise ValueError("ROI manifest geometry is not an object")
        cyt_bounds = tuple(float(value) for value in geometry["cytassist_roi_bounds_xyxy"])
        raw_bounds = tuple(float(value) for value in geometry["microscope_roi_fullres_bounds_xyxy"])
    if len(cyt_bounds) != 4 or (raw_bounds is not None and len(raw_bounds) != 4):
        raise ValueError("ROI bounds must each contain four values")
    cyt_x0, cyt_y0, cyt_x1, cyt_y1 = cyt_bounds
    if raw_bounds is not None:
        raw_x0, raw_y0, raw_x1, raw_y1 = raw_bounds
    moving_height, moving_width = moving_shape_yx
    if cyt_x1 <= cyt_x0 or cyt_y1 <= cyt_y0 or moving_width <= 0 or moving_height <= 0:
        raise ValueError("ROI geometry is empty")

    inverse_native = np.linalg.inv(native_moving_to_fixed)
    native_parts: list[np.ndarray] = []
    sr_parts: list[np.ndarray] = []
    offset = 0
    parquet = parquet_file(tissue_positions)
    required = {
        "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres",
    }
    if not required.issubset(parquet.schema.names):
        raise ValueError("tissue positions lack expression-grid geometry columns")
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=200_000):
        values = batch.to_pydict()
        count = len(values["array_row"])
        flat = np.arange(offset, offset + count, dtype=np.int64)
        rows = np.asarray(values["array_row"], dtype=np.int64)
        columns = np.asarray(values["array_col"], dtype=np.int64)
        if not np.array_equal(rows, flat // width) or not np.array_equal(columns, flat % width):
            raise ValueError("tissue positions are not the complete row-major grid")
        raw_x = np.asarray(values["pxl_col_in_fullres"], dtype=np.float64)
        raw_y = np.asarray(values["pxl_row_in_fullres"], dtype=np.float64)
        cyt_x, cyt_y = project(sr_microscope_to_cytassist, raw_x, raw_y)
        moving_x, moving_y = project(
            inverse_native, cyt_x - cyt_x0, cyt_y - cyt_y0,
        )
        native = (
            (moving_x >= 0.0) & (moving_x < moving_width)
            & (moving_y >= 0.0) & (moving_y < moving_height)
        )
        sr_crop = (
            (
                (raw_x >= raw_x0) & (raw_x < raw_x1)
                & (raw_y >= raw_y0) & (raw_y < raw_y1)
            )
            if raw_bounds is not None else np.ones(count, dtype=bool)
        )
        native_parts.append(flat[native])
        sr_parts.append(flat[sr_crop])
        offset += count
    if offset != width * height:
        raise ValueError(f"unexpected expression-grid size: {offset}")
    native_indices = np.concatenate(native_parts)
    sr_indices = np.concatenate(sr_parts)
    if len(native_indices) == 0 or np.any(np.diff(native_indices) <= 0):
        raise ValueError("native ROI selection is empty or not strictly ordered")
    intersection = np.intersect1d(native_indices, sr_indices, assume_unique=True)
    audit = {
        "native_roi_2um_bins": int(len(native_indices)),
        "selection_scope": "frozen_roi" if roi_manifest is not None else "full_registered_extent",
        "space_ranger_raw_crop_2um_bins": int(len(sr_indices)),
        "membership_intersection_2um_bins": int(len(intersection)),
        "native_only_boundary_bins": int(len(native_indices) - len(intersection)),
        "space_ranger_only_boundary_bins": int(len(sr_indices) - len(intersection)),
        "membership_jaccard_geometry_only": float(
            len(intersection) / (len(native_indices) + len(sr_indices) - len(intersection))
        ),
        "native_row_min": int(native_indices.min() // width),
        "native_row_max": int(native_indices.max() // width),
        "native_col_min": int(np.min(native_indices % width)),
        "native_col_max": int(np.max(native_indices % width)),
    }
    return native_indices, audit


def select_native_capture_grid_roi(
    spot_colrow_to_cytassist: np.ndarray,
    native_moving_to_fixed: np.ndarray,
    roi_manifest: dict[str, object] | None,
    moving_shape_yx: tuple[int, int],
    fixed_shape_yx: tuple[int, int],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select grid centers using only the native capture-grid transform."""
    if roi_manifest is None:
        fixed_height, fixed_width = fixed_shape_yx
        cyt_bounds = (0.0, 0.0, float(fixed_width), float(fixed_height))
    else:
        geometry = roi_manifest["geometry"]
        if not isinstance(geometry, dict):
            raise ValueError("ROI manifest geometry is not an object")
        cyt_bounds = tuple(float(value) for value in geometry["cytassist_roi_bounds_xyxy"])
    if len(cyt_bounds) != 4:
        raise ValueError("CytAssist ROI bounds must contain four values")
    cyt_x0, cyt_y0, cyt_x1, cyt_y1 = cyt_bounds
    moving_height, moving_width = moving_shape_yx
    if cyt_x1 <= cyt_x0 or cyt_y1 <= cyt_y0 or moving_width <= 0 or moving_height <= 0:
        raise ValueError("ROI geometry is empty")

    inverse_native = np.linalg.inv(native_moving_to_fixed)
    selected: list[np.ndarray] = []
    total = width * height
    for offset in range(0, total, 200_000):
        flat = np.arange(offset, min(offset + 200_000, total), dtype=np.int64)
        rows = flat // width
        columns = flat % width
        cyt_x, cyt_y = project(
            spot_colrow_to_cytassist,
            columns.astype(np.float64), rows.astype(np.float64),
        )
        moving_x, moving_y = project(inverse_native, cyt_x - cyt_x0, cyt_y - cyt_y0)
        inside = (
            (moving_x >= 0.0) & (moving_x < moving_width)
            & (moving_y >= 0.0) & (moving_y < moving_height)
        )
        selected.append(flat[inside])
    indices = np.concatenate(selected)
    if len(indices) == 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("native capture-grid ROI selection is empty or not strictly ordered")
    return indices, {
        "native_roi_2um_bins": int(len(indices)),
        "selection_scope": "frozen_roi" if roi_manifest is not None else "full_registered_extent",
        "geometry_source": "independent_capture_grid",
        "native_row_min": int(indices.min() // width),
        "native_row_max": int(indices.max() // width),
        "native_col_min": int(np.min(indices % width)),
        "native_col_max": int(np.max(indices % width)),
    }


def load_roi_states(
    path: Path, roi_indices: np.ndarray, *, width: int, height: int,
) -> np.ndarray:
    result = np.empty(len(roi_indices), dtype=np.uint8)
    selected_offset = 0
    grid_offset = 0
    parquet = parquet_file(path)
    required = {"square_002um", "in_cell", "in_nucleus"}
    if not required.issubset(parquet.schema.names):
        raise ValueError("barcode mappings lack cell/nucleus compartments")
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=100_000):
        values = batch.to_pydict()
        count = len(values["square_002um"])
        first = values["square_002um"][0]
        last = values["square_002um"][-1]
        expected_first = f"s_002um_{grid_offset // width:05d}_{grid_offset % width:05d}-1"
        last_index = grid_offset + count - 1
        expected_last = f"s_002um_{last_index // width:05d}_{last_index % width:05d}-1"
        if first != expected_first or last != expected_last:
            raise ValueError("barcode mappings are not the declared row-major grid")
        begin = int(np.searchsorted(roi_indices, grid_offset, side="left"))
        end = int(np.searchsorted(roi_indices, grid_offset + count, side="left"))
        if end > begin:
            local = roi_indices[begin:end] - grid_offset
            in_cell = np.asarray(values["in_cell"], dtype=bool)[local]
            in_nucleus = np.asarray(values["in_nucleus"], dtype=bool)[local]
            if np.any(in_nucleus & ~in_cell):
                raise ValueError("nucleus ROI bin is not contained by the cell mask")
            result[selected_offset:selected_offset + len(local)] = np.where(
                in_nucleus, 2, np.where(in_cell, 1, 0),
            )
            selected_offset += len(local)
        grid_offset += count
    if grid_offset != width * height or selected_offset != len(roi_indices):
        raise ValueError("barcode mappings or ROI compartment selection are incomplete")
    return result


def registration_to_segmentation_coordinate(
    moving: np.ndarray,
    *,
    moving_downsample: int,
    moving_sampling: str,
    moving_source_origin: float,
    segmentation_source_origin: float,
    segmentation_downsample: int,
) -> np.ndarray:
    """Map native-registration pixel coordinates to segmentation pixels.

    ``area`` follows OpenCV's pixel-centre resize convention. ``decimate``
    follows the direct ``source[::downsample]`` convention used by the frozen
    full-field registration fixture.
    """
    moving = np.asarray(moving, dtype=np.float64)
    if moving_downsample < 1 or segmentation_downsample < 1:
        raise ValueError("image downsample factors must be positive")
    if moving_sampling == "area":
        raw_local = (moving + 0.5) * moving_downsample - 0.5
    elif moving_sampling == "decimate":
        raw_local = moving * moving_downsample
    else:
        raise ValueError(f"unknown registration moving sampling: {moving_sampling}")
    raw_global = moving_source_origin + raw_local
    return (
        (raw_global - segmentation_source_origin + 0.5)
        / segmentation_downsample
        - 0.5
    )


def load_cellpose_states(
    segmentation_root: Path,
    tissue_positions: Path,
    roi_indices: np.ndarray,
    sr_microscope_to_cytassist: np.ndarray,
    native_moving_to_fixed: np.ndarray,
    roi_manifest: dict[str, object] | None,
    *,
    width: int,
    height: int,
    moving_downsample: int,
    moving_sampling: str,
) -> tuple[np.ndarray, dict[str, object]]:
    import zarr

    summary_path = segmentation_root / "summary.json"
    nucleus_path = segmentation_root / "masks" / "nucleus_labels.zarr"
    cell_path = segmentation_root / "masks" / "cell_labels.zarr"
    for path in (summary_path, nucleus_path, cell_path):
        if not path.exists():
            raise ValueError(f"incomplete Cellpose segmentation: {path}")
    summary = json.loads(summary_path.read_text())
    contract = summary["input_contract"]
    if bool(contract.get("expression_or_vendor_segmentation_loaded")):
        raise ValueError("Cellpose segmentation declares expression or vendor-segmentation ancestry")
    if int(summary["counts"].get("nucleus_outside_cell_pixels", -1)) != 0:
        raise ValueError("Cellpose summary declares a nucleus/cell hierarchy failure")
    segmentation_roi = tuple(float(value) for value in contract["source_roi_xyxy"])
    segmentation_downsample = int(contract["downsample"])
    segmentation_shape = tuple(int(value) for value in contract["segmentation_shape_yx"])
    nucleus = zarr.open(nucleus_path, mode="r")
    cell = zarr.open(cell_path, mode="r")
    nucleus_hash = tree_sha256(nucleus_path)
    cell_hash = tree_sha256(cell_path)
    if nucleus_hash != summary["outputs"]["nucleus_labels.zarr"]:
        raise ValueError("Cellpose nucleus label tree differs from its declared hash")
    if cell_hash != summary["outputs"]["cell_labels.zarr"]:
        raise ValueError("Cellpose cell label tree differs from its declared hash")
    if tuple(nucleus.shape) != segmentation_shape or tuple(cell.shape) != segmentation_shape:
        raise ValueError("Cellpose label shapes disagree with their declared segmentation geometry")

    if roi_manifest is None:
        cyt_x0 = cyt_y0 = moving_x0 = moving_y0 = 0.0
    else:
        geometry = roi_manifest["geometry"]
        cyt_x0, cyt_y0, _, _ = (float(value) for value in geometry["cytassist_roi_bounds_xyxy"])
        moving_x0, moving_y0, _, _ = (
            float(value) for value in geometry["microscope_roi_fullres_bounds_xyxy"]
        )
    inverse_native = np.linalg.inv(native_moving_to_fixed)
    states = np.empty(len(roi_indices), dtype=np.uint8)
    supported = np.empty(len(roi_indices), dtype=bool)
    selected_offset = 0
    grid_offset = 0
    parquet = parquet_file(tissue_positions)
    required = {"array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"}
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=200_000):
        values = batch.to_pydict()
        count = len(values["array_row"])
        begin = int(np.searchsorted(roi_indices, grid_offset, side="left"))
        end = int(np.searchsorted(roi_indices, grid_offset + count, side="left"))
        if end > begin:
            local = roi_indices[begin:end] - grid_offset
            raw_x = np.asarray(values["pxl_col_in_fullres"], dtype=np.float64)[local]
            raw_y = np.asarray(values["pxl_row_in_fullres"], dtype=np.float64)[local]
            cyt_x, cyt_y = project(sr_microscope_to_cytassist, raw_x, raw_y)
            moving_x, moving_y = project(inverse_native, cyt_x - cyt_x0, cyt_y - cyt_y0)
            segmentation_x = registration_to_segmentation_coordinate(
                moving_x,
                moving_downsample=moving_downsample,
                moving_sampling=moving_sampling,
                moving_source_origin=moving_x0,
                segmentation_source_origin=segmentation_roi[0],
                segmentation_downsample=segmentation_downsample,
            )
            segmentation_y = registration_to_segmentation_coordinate(
                moving_y,
                moving_downsample=moving_downsample,
                moving_sampling=moving_sampling,
                moving_source_origin=moving_y0,
                segmentation_source_origin=segmentation_roi[1],
                segmentation_downsample=segmentation_downsample,
            )
            px = np.floor(segmentation_x).astype(np.int64)
            py = np.floor(segmentation_y).astype(np.int64)
            inside = (
                (px >= 0) & (px < segmentation_shape[1])
                & (py >= 0) & (py < segmentation_shape[0])
            )
            nucleus_values = np.zeros(len(local), dtype=bool)
            cell_values = np.zeros(len(local), dtype=bool)
            nucleus_values[inside] = np.asarray(nucleus.vindex[py[inside], px[inside]]) > 0
            cell_values[inside] = np.asarray(cell.vindex[py[inside], px[inside]]) > 0
            if np.any(nucleus_values & ~cell_values):
                raise ValueError("Cellpose nucleus bin is outside its expanded cell domain")
            states[selected_offset:selected_offset + len(local)] = np.where(
                nucleus_values, 2, np.where(cell_values, 1, 0),
            )
            supported[selected_offset:selected_offset + len(local)] = inside
            selected_offset += len(local)
        grid_offset += count
    if grid_offset != width * height or selected_offset != len(roi_indices):
        raise ValueError("Cellpose ROI projection is incomplete")
    audit = {
        "segmentation_root": str(segmentation_root.resolve()),
        "segmentation_summary_sha256": sha256(summary_path),
        "nucleus_labels_tree_sha256": nucleus_hash,
        "cell_labels_tree_sha256": cell_hash,
        "segmentation_shape_yx": list(segmentation_shape),
        "segmentation_downsample": segmentation_downsample,
        "registration_moving_source_downsample": moving_downsample,
        "registration_moving_sampling": moving_sampling,
        "coordinate_resampling": "explicit source-pixel and pixel-centre transform",
        "native_roi_bins_outside_cellpose_pixel_support": int(np.count_nonzero(~supported)),
    }
    return states, supported, audit


def load_cellpose_states_from_capture_grid(
    segmentation_root: Path,
    spot_colrow_to_cytassist: np.ndarray,
    roi_indices: np.ndarray,
    native_moving_to_fixed: np.ndarray,
    roi_manifest: dict[str, object] | None,
    *,
    width: int,
    height: int,
    moving_downsample: int,
    moving_sampling: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Project independent grid centers into independent Cellpose labels."""
    import zarr

    summary_path = segmentation_root / "summary.json"
    nucleus_path = segmentation_root / "masks" / "nucleus_labels.zarr"
    cell_path = segmentation_root / "masks" / "cell_labels.zarr"
    for path in (summary_path, nucleus_path, cell_path):
        if not path.exists():
            raise ValueError(f"incomplete Cellpose segmentation: {path}")
    summary = json.loads(summary_path.read_text())
    contract = summary["input_contract"]
    if bool(contract.get("expression_or_vendor_segmentation_loaded")):
        raise ValueError("Cellpose segmentation declares expression or vendor-segmentation ancestry")
    if int(summary["counts"].get("nucleus_outside_cell_pixels", -1)) != 0:
        raise ValueError("Cellpose summary declares a nucleus/cell hierarchy failure")
    segmentation_roi = tuple(float(value) for value in contract["source_roi_xyxy"])
    segmentation_downsample = int(contract["downsample"])
    segmentation_shape = tuple(int(value) for value in contract["segmentation_shape_yx"])
    nucleus = zarr.open(nucleus_path, mode="r")
    cell = zarr.open(cell_path, mode="r")
    nucleus_hash = tree_sha256(nucleus_path)
    cell_hash = tree_sha256(cell_path)
    if nucleus_hash != summary["outputs"]["nucleus_labels.zarr"]:
        raise ValueError("Cellpose nucleus label tree differs from its declared hash")
    if cell_hash != summary["outputs"]["cell_labels.zarr"]:
        raise ValueError("Cellpose cell label tree differs from its declared hash")
    if tuple(nucleus.shape) != segmentation_shape or tuple(cell.shape) != segmentation_shape:
        raise ValueError("Cellpose label shapes disagree with their declared segmentation geometry")

    if roi_manifest is None:
        cyt_x0 = cyt_y0 = moving_x0 = moving_y0 = 0.0
    else:
        geometry = roi_manifest["geometry"]
        cyt_x0, cyt_y0, _, _ = (float(value) for value in geometry["cytassist_roi_bounds_xyxy"])
        moving_x0, moving_y0, _, _ = (
            float(value) for value in geometry["microscope_roi_fullres_bounds_xyxy"]
        )
    inverse_native = np.linalg.inv(native_moving_to_fixed)
    states = np.empty(len(roi_indices), dtype=np.uint8)
    supported = np.empty(len(roi_indices), dtype=bool)
    for begin in range(0, len(roi_indices), 200_000):
        end = min(begin + 200_000, len(roi_indices))
        flat = roi_indices[begin:end]
        rows = flat // width
        columns = flat % width
        cyt_x, cyt_y = project(
            spot_colrow_to_cytassist,
            columns.astype(np.float64), rows.astype(np.float64),
        )
        moving_x, moving_y = project(inverse_native, cyt_x - cyt_x0, cyt_y - cyt_y0)
        segmentation_x = registration_to_segmentation_coordinate(
            moving_x,
            moving_downsample=moving_downsample,
            moving_sampling=moving_sampling,
            moving_source_origin=moving_x0,
            segmentation_source_origin=segmentation_roi[0],
            segmentation_downsample=segmentation_downsample,
        )
        segmentation_y = registration_to_segmentation_coordinate(
            moving_y,
            moving_downsample=moving_downsample,
            moving_sampling=moving_sampling,
            moving_source_origin=moving_y0,
            segmentation_source_origin=segmentation_roi[1],
            segmentation_downsample=segmentation_downsample,
        )
        px = np.floor(segmentation_x).astype(np.int64)
        py = np.floor(segmentation_y).astype(np.int64)
        inside = (
            (px >= 0) & (px < segmentation_shape[1])
            & (py >= 0) & (py < segmentation_shape[0])
        )
        nucleus_values = np.zeros(len(flat), dtype=bool)
        cell_values = np.zeros(len(flat), dtype=bool)
        nucleus_values[inside] = np.asarray(nucleus.vindex[py[inside], px[inside]]) > 0
        cell_values[inside] = np.asarray(cell.vindex[py[inside], px[inside]]) > 0
        if np.any(nucleus_values & ~cell_values):
            raise ValueError("Cellpose nucleus bin is outside its expanded cell domain")
        states[begin:end] = np.where(nucleus_values, 2, np.where(cell_values, 1, 0))
        supported[begin:end] = inside
    return states, supported, {
        "segmentation_root": str(segmentation_root.resolve()),
        "segmentation_summary_sha256": sha256(summary_path),
        "nucleus_labels_tree_sha256": nucleus_hash,
        "cell_labels_tree_sha256": cell_hash,
        "segmentation_shape_yx": list(segmentation_shape),
        "segmentation_downsample": segmentation_downsample,
        "registration_moving_source_downsample": moving_downsample,
        "registration_moving_sampling": moving_sampling,
        "coordinate_resampling": "explicit source-pixel and pixel-centre transform",
        "grid_geometry": "independent_capture_grid",
        "native_roi_bins_outside_cellpose_pixel_support": int(np.count_nonzero(~supported)),
    }


def build_roi_geometry(args: argparse.Namespace) -> RoiGeometry:
    native_json = json.loads(args.native_registration_json.read_text())
    moving_shape = tuple(int(value) for value in native_json["moving_shape_yx"])
    fixed_shape = tuple(int(value) for value in native_json["fixed_shape_yx"])
    if len(moving_shape) != 2 or len(fixed_shape) != 2:
        raise ValueError("native registration image shapes must each have two dimensions")
    roi_manifest = json.loads(args.roi_manifest.read_text()) if args.roi_manifest else None
    native_transform = matrix_from_json(args.native_registration_json, ("moving_to_fixed",))
    if args.capture_grid_json is not None:
        spot_to_cytassist = matrix_from_json(
            args.capture_grid_json, ("spot_colrow_to_cytassist",),
        )
        indices, audit = select_native_capture_grid_roi(
            spot_to_cytassist, native_transform, roi_manifest,
            moving_shape, fixed_shape, width=args.width, height=args.height,
        )
    else:
        indices, audit = select_native_roi(
            args.tissue_positions,
            matrix_from_json(args.space_ranger_alignment_json, ("cytAssistInfo", "transformImages")),
            native_transform, roi_manifest, moving_shape, fixed_shape,
            width=args.width, height=args.height,
        )
    if args.cellpose_segmentation is not None:
        if args.capture_grid_json is not None:
            states, supported, segmentation_audit = load_cellpose_states_from_capture_grid(
                args.cellpose_segmentation, spot_to_cytassist, indices,
                native_transform, roi_manifest, width=args.width, height=args.height,
                moving_downsample=args.registration_moving_source_downsample,
                moving_sampling=args.registration_moving_sampling,
            )
        else:
            states, supported, segmentation_audit = load_cellpose_states(
                args.cellpose_segmentation, args.tissue_positions, indices,
                matrix_from_json(args.space_ranger_alignment_json, ("cytAssistInfo", "transformImages")),
                native_transform, roi_manifest, width=args.width, height=args.height,
                moving_downsample=args.registration_moving_source_downsample,
                moving_sampling=args.registration_moving_sampling,
            )
        audit["compartment_source"] = "independent_he_cellpose"
        audit["cellpose_projection"] = segmentation_audit
        indices = indices[supported]
        states = states[supported]
        audit["cellpose_supported_roi_2um_bins"] = int(len(indices))
    else:
        states = load_roi_states(
            args.barcode_mappings, indices, width=args.width, height=args.height,
        )
        audit["compartment_source"] = "retained_space_ranger_segmentation"
    rows, columns = indices // args.width, indices % args.width
    parent_width = math.ceil(args.width / args.parent_size)
    parent_ids = (rows // args.parent_size) * parent_width + columns // args.parent_size
    parent_counts = np.bincount(parent_ids)
    complete = parent_counts == args.parent_size ** 2
    audit.update({
        "cell_mask_bins": int(np.count_nonzero(states > 0)),
        "nucleus_mask_bins": int(np.count_nonzero(states == 2)),
        "extracellular_mask_bins": int(np.count_nonzero(states == 0)),
        "complete_16um_parents": int(np.count_nonzero(complete)),
    })
    return RoiGeometry(indices, states, parent_ids, complete, audit)


def build_mex_axis(
    root: Path,
    target_index: dict[str, int],
    roi_indices: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    features = [line.split("\t", 1)[0] for line in read_lines(mex_file(root, "features.tsv"))]
    if len(features) != len(set(features)):
        raise ValueError("duplicate STAR feature axis")
    unexpected = sorted(set(features) - set(target_index))
    if unexpected:
        raise ValueError(f"STAR matrix has {len(unexpected)} off-target features")
    mapped_rows = np.asarray([target_index[name] for name in features], dtype=np.int32)
    full_to_roi = np.full(width * height, -1, dtype=np.int32)
    full_to_roi[roi_indices] = np.arange(len(roi_indices), dtype=np.int32)
    mapped_columns: list[int] = []
    seen = np.zeros(width * height, dtype=bool)
    with open_text(mex_file(root, "barcodes.tsv")) as handle:
        for line in handle:
            unit = line.rstrip("\n").split("\t", 1)[0]
            parts = unit.split("_")
            if len(parts) != 4 or parts[0] != "s" or parts[1] != "002um":
                raise ValueError(f"invalid STAR 2 um barcode: {unit}")
            row = int(parts[2])
            column = int(parts[3].split("-", 1)[0])
            if row >= height or column >= width:
                raise ValueError(f"STAR barcode outside expression grid: {unit}")
            flat = row * width + column
            if seen[flat]:
                raise ValueError(f"duplicate STAR barcode: {unit}")
            seen[flat] = True
            mapped_columns.append(int(full_to_roi[flat]))
    if not mapped_columns:
        raise ValueError("empty STAR barcode axis")
    return features, mapped_rows, np.asarray(mapped_columns, dtype=np.int32)


def absent_star_feature_mass(
    star_features: list[str],
    target_index: dict[str, int],
    vendor: sparse.csr_matrix,
) -> tuple[list[str], float]:
    """Return vendor mass retained on rows absent from an observed STAR axis."""
    missing = sorted(set(target_index) - set(star_features))
    rows = [target_index[name] for name in missing]
    mass = float(vendor[rows, :].sum()) if rows else 0.0
    if not math.isfinite(mass) or mass < 0.0:
        raise ValueError("invalid vendor mass on features absent from STAR")
    return missing, mass


def load_mex_roi_sparse(
    root: Path,
    mapped_rows: np.ndarray,
    mapped_columns: np.ndarray,
    *,
    target_features: int,
    roi_bins: int,
) -> sparse.csr_matrix:
    raw = scipy_io.mmread(str(mex_file(root, "matrix.mtx"))).tocoo()
    if raw.shape != (len(mapped_rows), len(mapped_columns)):
        raise ValueError(f"STAR MEX axes disagree with matrix dimensions in {root}")
    if np.any(~np.isfinite(raw.data)) or np.any(raw.data < 0.0):
        raise ValueError(f"invalid STAR MEX values in {root}")
    local_columns = mapped_columns[raw.col]
    selected = (local_columns >= 0) & (raw.data != 0.0)
    selected_count = int(np.count_nonzero(selected))
    result = sparse.coo_matrix(
        (
            np.asarray(raw.data[selected], dtype=np.float64),
            (mapped_rows[raw.row[selected]], local_columns[selected]),
        ),
        shape=(target_features, roi_bins),
    ).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    if result.nnz != selected_count:
        raise ValueError(f"duplicate STAR feature-bin entries in {root}")
    del raw, local_columns, selected
    gc.collect()
    return result


def correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool) -> float:
    if len(left) == 0:
        return 1.0
    x = rankdata(left, method="average") if ranked else left
    y = rankdata(right, method="average") if ranked else right
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)))
    if denominator:
        return float(np.dot(centered_x, centered_y) / denominator)
    return 1.0 if np.array_equal(x, y) else 0.0


def parent_metrics(field: np.ndarray, geometry: RoiGeometry) -> dict[str, float | int]:
    count = len(geometry.complete_parent_mask)
    mass = np.bincount(geometry.parent_ids, weights=field, minlength=count)
    cell = np.bincount(
        geometry.parent_ids, weights=(geometry.states > 0), minlength=count,
    )
    nucleus = np.bincount(
        geometry.parent_ids, weights=(geometry.states == 2), minlength=count,
    )
    selected = geometry.complete_parent_mask
    density = mass[selected] / 64.0
    cell_fraction = cell[selected] / 64.0
    nucleus_fraction = nucleus[selected] / 64.0
    return {
        "complete_16um_parents": int(np.count_nonzero(selected)),
        "cell_fraction_pearson_16um": correlation(density, cell_fraction, ranked=False),
        "cell_fraction_spearman_16um": correlation(density, cell_fraction, ranked=True),
        "nucleus_fraction_pearson_16um": correlation(density, nucleus_fraction, ranked=False),
        "nucleus_fraction_spearman_16um": correlation(density, nucleus_fraction, ranked=True),
    }


def mask_summary(field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = float(np.sum(field, dtype=np.float64))
    mass = float(np.sum(field[mask], dtype=np.float64))
    area_fraction = float(np.mean(mask))
    mass_fraction = mass / total if total else 0.0
    return {
        "mass": mass,
        "mass_fraction": mass_fraction,
        "area_fraction": area_fraction,
        "area_normalized_enrichment": mass_fraction / area_fraction if area_fraction else 0.0,
    }


def score_field(
    name: str,
    role: str,
    policy: str,
    field: np.ndarray,
    geometry: RoiGeometry,
) -> dict[str, object]:
    field = np.asarray(field, dtype=np.float64).reshape(-1)
    if len(field) != len(geometry.indices) or np.any(~np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError(f"invalid ROI molecule field: {name}")
    in_cell = geometry.states > 0
    in_nucleus = geometry.states == 2
    extracellular = geometry.states == 0
    cell = mask_summary(field, in_cell)
    nucleus = mask_summary(field, in_nucleus)
    outside = mask_summary(field, extracellular)
    cell_classifier = binary_density_metrics(field, in_cell)
    nucleus_classifier = binary_density_metrics(field, in_nucleus)
    cell_tv = distribution_distance(field, in_cell.astype(np.float64))
    nucleus_tv = distribution_distance(field, in_nucleus.astype(np.float64))
    parent = parent_metrics(field, geometry)
    return {
        "field": name,
        "field_role": role,
        "policy": policy,
        "raw_molecule_mass": float(np.sum(field, dtype=np.float64)),
        "occupied_2um_bins": int(np.count_nonzero(field)),
        "roi_2um_bins": int(len(field)),
        "complete_16um_parents": parent["complete_16um_parents"],
        "in_cell_mass": cell["mass"],
        "in_cell_mass_fraction": cell["mass_fraction"],
        "in_cell_area_fraction": cell["area_fraction"],
        "in_cell_area_normalized_enrichment": cell["area_normalized_enrichment"],
        "in_nucleus_mass": nucleus["mass"],
        "in_nucleus_mass_fraction": nucleus["mass_fraction"],
        "in_nucleus_area_fraction": nucleus["area_fraction"],
        "in_nucleus_area_normalized_enrichment": nucleus["area_normalized_enrichment"],
        "extracellular_mass": outside["mass"],
        "extracellular_mass_fraction": outside["mass_fraction"],
        "extracellular_area_fraction": outside["area_fraction"],
        "extracellular_area_normalized_enrichment": outside["area_normalized_enrichment"],
        "in_cell_roc_auc": cell_classifier["roc_auc"],
        "in_cell_average_precision": cell_classifier["average_precision"],
        "in_nucleus_roc_auc": nucleus_classifier["roc_auc"],
        "in_nucleus_average_precision": nucleus_classifier["average_precision"],
        "cell_fraction_pearson_16um": parent["cell_fraction_pearson_16um"],
        "cell_fraction_spearman_16um": parent["cell_fraction_spearman_16um"],
        "nucleus_fraction_pearson_16um": parent["nucleus_fraction_pearson_16um"],
        "nucleus_fraction_spearman_16um": parent["nucleus_fraction_spearman_16um"],
        "in_cell_normalized_tv": cell_tv["normalized_total_variation"],
        "in_nucleus_normalized_tv": nucleus_tv["normalized_total_variation"],
        "equal_mass_normalization_applied": False,
    }


def matrix_field(values: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(values.sum(axis=0), dtype=np.float64).ravel()


def compatibility_fields(
    star: sparse.csr_matrix,
    vendor: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = star.minimum(vendor).tocsr()
    shared.eliminate_zeros()
    star_only = (star - shared).tocsr()
    vendor_only = (vendor - shared).tocsr()
    star_only.eliminate_zeros()
    vendor_only.eliminate_zeros()
    if np.any(star_only.data < -1e-12) or np.any(vendor_only.data < -1e-12):
        raise ValueError("negative compatibility residual")
    return matrix_field(shared), matrix_field(star_only), matrix_field(vendor_only)


def increment_field(policy: sparse.csr_matrix, strict: sparse.csr_matrix) -> np.ndarray:
    difference = (policy - strict).tocsr()
    difference.eliminate_zeros()
    if np.any(difference.data < -1e-9):
        raise ValueError("policy loses strict feature-by-bin mass in the ROI")
    difference.data[difference.data < 0.0] = 0.0
    difference.eliminate_zeros()
    return matrix_field(difference)


def main() -> int:
    args = parse_args()
    if args.tissue_positions is not None and args.space_ranger_alignment_json is None:
        raise SystemExit("--space-ranger-alignment-json is required with --tissue-positions")
    if args.capture_grid_json is not None and args.space_ranger_alignment_json is not None:
        raise SystemExit("--space-ranger-alignment-json is forbidden with --capture-grid-json")
    capture_grid_lineage = None
    if args.capture_grid_json is not None:
        capture_payload = json.loads(args.capture_grid_json.read_text())
        if capture_payload.get("slide") != args.slide or capture_payload.get("area") != args.area:
            raise SystemExit("capture-grid slide/area differs from the declared comparison")
        if capture_payload.get("grid_shape_row_col") != [args.height, args.width]:
            raise SystemExit("capture-grid shape differs from the declared comparison")
        reference_audit = capture_payload.get("reference_audit")
        if not isinstance(reference_audit, dict) or reference_audit.get("enabled") is not False:
            raise SystemExit(
                "publication capture-grid JSON must be generated without a reference alignment"
            )
        capture_inputs = capture_payload.get("inputs")
        if not isinstance(capture_inputs, dict) or not capture_inputs:
            raise SystemExit("capture-grid JSON lacks its raw input lineage")
        raw_paths = {name: Path(path) for name, path in capture_inputs.items()}
        for path in raw_paths.values():
            if not path.is_file():
                raise SystemExit(f"missing transitive capture-grid input: {path}")
        capture_grid_lineage = {
            "schema": capture_payload.get("schema"),
            "fit": capture_payload.get("fit"),
            "raw_inputs": {
                name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in raw_paths.items()
            },
        }
    inputs = (args.vendor_h5, args.native_registration_json)
    inputs += (
        (args.capture_grid_json,)
        if args.capture_grid_json is not None else
        (args.tissue_positions, args.space_ranger_alignment_json)
    )
    if args.roi_manifest is not None:
        inputs += (args.roi_manifest,)
    inputs += (
        (args.barcode_mappings,)
        if args.barcode_mappings is not None
        else (args.cellpose_segmentation / "summary.json",)
    )
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    if args.parent_size != 8:
        raise SystemExit("this publication scorer currently requires 8x8 children per 16 um parent")
    args.out_dir.mkdir(parents=True)

    geometry = build_roi_geometry(args)
    vendor_full, target_index, grid_width = load_space_ranger_sparse(args.vendor_h5, 2)
    if grid_width != args.width or vendor_full.shape[1] != args.width * args.height:
        raise SystemExit("Space Ranger matrix grid differs from declared geometry")
    vendor = vendor_full[:, geometry.indices].tocsr()
    del vendor_full
    gc.collect()

    strict_root = args.policy_mex_root / "strict" / "square_002um"
    features, mapped_rows, mapped_columns = build_mex_axis(
        strict_root, target_index, geometry.indices,
        width=args.width, height=args.height,
    )
    missing_features, missing_mass = absent_star_feature_mass(
        features, target_index, vendor,
    )

    strict = load_mex_roi_sparse(
        strict_root, mapped_rows, mapped_columns,
        target_features=len(target_index), roi_bins=len(geometry.indices),
    )
    rows = [score_field(
        "space_ranger", "vendor_compatibility_context", "space_ranger",
        matrix_field(vendor), geometry,
    )]
    matrix_inputs: dict[str, dict[str, str]] = {}
    for policy, directory in POLICIES:
        root = args.policy_mex_root / directory / "square_002um"
        if not mex_file(root, "features.tsv").samefile(mex_file(strict_root, "features.tsv")):
            if read_lines(mex_file(root, "features.tsv")) != features:
                raise SystemExit(f"{policy} feature axis differs from strict")
        if not mex_file(root, "barcodes.tsv").samefile(mex_file(strict_root, "barcodes.tsv")):
            if sha256(mex_file(root, "barcodes.tsv")) != sha256(mex_file(strict_root, "barcodes.tsv")):
                raise SystemExit(f"{policy} barcode axis differs from strict")
        values = strict if policy == "strict" else load_mex_roi_sparse(
            root, mapped_rows, mapped_columns,
            target_features=len(target_index), roi_bins=len(geometry.indices),
        )
        whole = matrix_field(values)
        rows.append(score_field(policy, "whole_open_policy", policy, whole, geometry))
        shared, star_only, vendor_only = compatibility_fields(values, vendor)
        rows.extend((
            score_field(
                f"{policy}_shared_with_space_ranger", "shared_feature_bin_mass",
                policy, shared, geometry,
            ),
            score_field(
                f"{policy}_star_only", "open_additional_feature_bin_mass",
                policy, star_only, geometry,
            ),
            score_field(
                f"{policy}_space_ranger_only", "vendor_only_feature_bin_mass",
                policy, vendor_only, geometry,
            ),
        ))
        if not math.isclose(
            float(np.sum(shared) + np.sum(star_only)), float(np.sum(whole)),
            rel_tol=1e-12, abs_tol=1e-6,
        ):
            raise SystemExit(f"STAR ROI compatibility mass does not reconcile for {policy}")
        if not math.isclose(
            float(np.sum(shared) + np.sum(vendor_only)), float(vendor.sum()),
            rel_tol=1e-12, abs_tol=1e-6,
        ):
            raise SystemExit(f"Space Ranger ROI compatibility mass does not reconcile for {policy}")
        if policy != "strict":
            rows.append(score_field(
                f"{policy}_increment_over_strict", "ambiguous_increment_over_strict",
                policy, increment_field(values, strict), geometry,
            ))
        matrix_inputs[policy] = {
            name: sha256(mex_file(root, name))
            for name in ("matrix.mtx", "features.tsv", "barcodes.tsv")
        }
        if policy != "strict":
            del values
            gc.collect()

    suffix = "_roi" if args.roi_manifest is not None else ""
    fields_path = args.out_dir / f"field_biology{suffix}.tsv"
    comparisons_path = args.out_dir / f"comparison_deltas{suffix}.tsv"
    membership_path = args.out_dir / "roi_membership.tsv"
    write_rows(fields_path, rows)
    write_rows(comparisons_path, comparison_rows(rows))
    output_paths = [fields_path, comparisons_path]
    if not args.omit_membership:
        with membership_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("flat_index", "array_row", "array_col", "compartment"))
            labels = np.asarray(("extracellular", "cell_non_nucleus", "nucleus"))
            for index, state in zip(geometry.indices, geometry.states, strict=True):
                writer.writerow((int(index), int(index // args.width), int(index % args.width), labels[state]))
        output_paths.append(membership_path)

    summary = {
        "schema": (
            "visium_hd_processing.flex_native_capture_grid_roi_biology.v1"
            if args.capture_grid_json is not None and args.roi_manifest is not None else
            "visium_hd_processing.flex_native_capture_grid_biology.v1"
            if args.capture_grid_json is not None else
            "visium_hd_processing.flex_native_registration_roi_biology.v1"
            if args.roi_manifest is not None else
            "visium_hd_processing.flex_native_registration_biology.v1"
        ),
        "slide": args.slide,
        "area": args.area,
        "umi_mode": args.umi_mode,
        "roi_definition": (
            "2um grid centers whose inverse native microscope-to-CytAssist projection "
            + (
                "falls inside the frozen microscope ROI"
                if args.roi_manifest is not None else
                "falls inside the complete microscope image"
            )
        ),
        "roi_geometry_audit": geometry.audit,
        "feature_axis_audit": {
            "space_ranger_target_features": len(target_index),
            "star_features": len(features),
            "space_ranger_features_absent_from_star": missing_features,
            "space_ranger_mass_on_features_absent_from_star": missing_mass,
            "space_ranger_mass_fraction_on_features_absent_from_star": (
                missing_mass / float(vendor.sum()) if vendor.nnz else 0.0
            ),
            "absent_feature_mass_is_retained_in_vendor_fields": True,
        },
        "capture_grid_lineage": capture_grid_lineage,
        "inputs": {
            str(path.resolve()): sha256(path) for path in inputs
        } | {"open_policy_mex": matrix_inputs},
        "outputs": {
            path.name: sha256(path)
            for path in output_paths
        },
        "invariants": {
            "identical_roi_bins_for_star_and_space_ranger": True,
            "raw_mass_not_normalized": True,
            "strict_and_ambiguous_increment_reported_separately": True,
            "space_ranger_assignment_not_used_to_select_roi": True,
            "space_ranger_slide_geometry_held_fixed_for_assignment_comparison": (
                args.capture_grid_json is None
            ),
            "independent_capture_grid_used": args.capture_grid_json is not None,
            "space_ranger_alignment_excluded_from_geometry_ancestry": (
                args.capture_grid_json is not None
            ),
            "space_ranger_tissue_positions_excluded_from_geometry_ancestry": (
                args.capture_grid_json is not None
            ),
            "cell_nucleus_masks_not_used_as_assignment_priors": True,
            "normalized_tv_is_secondary": True,
            "occupancy_jaccard_not_computed": True,
        },
        "limitations": [
            (
                "Cell and nucleus masks are independently derived from the H&E with Cellpose; "
                "the H&E remains a noisy morphology oracle."
                if args.cellpose_segmentation is not None else
                "Cell and nucleus masks are retained Space Ranger segmentation and are a noisy morphology oracle."
            ),
            (
                "Slide-to-CytAssist and microscope-to-CytAssist geometry are independently derived from raw image inputs."
                if args.capture_grid_json is not None else
                "The slide-to-CytAssist geometry is held at the retained Space Ranger solution; only microscope-to-CytAssist registration is native."
            ),
            "Shared/additional fields are feature-by-bin mass decompositions, not exact read-member-set matches.",
        ],
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "checksums.sha256").write_text("".join(
        f"{sha256(path)}  {path.name}\n"
        for path in (*output_paths, summary_path)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
