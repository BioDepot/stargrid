#!/usr/bin/env python3
"""Segment Visium HD H&E nuclei with external Cellpose and expand to cell domains.

Cellpose remains an external dependency.  This publication adapter supplies the
Visium-specific H&E optical-density channel, bounded source-image reads, exact
overlap ownership, deterministic label namespacing, and streamed cell-domain
expansion.  It never reads expression counts or Space Ranger segmentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import tifffile
import zarr


CELLPOSE_PROFILES = {
    "independent_v1": {
        "model": "cpsam",
        "downsample": 2,
        "tile_size": 2048,
        "overlap": 128,
        "nucleus_expansion_um": 2.0,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,
        "max_size_fraction": 0.4,
    },
    "sr_compat_v1": {
        "model": "cpsam",
        "downsample": 2,
        "tile_size": 2048,
        "overlap": 128,
        "nucleus_expansion_um": 8.0,
        "flow_threshold": 0.8,
        "cellprob_threshold": 0.0,
        "min_size": 15,
        "max_size_fraction": 0.4,
    },
}
DEFAULT_PARAMETERS = CELLPOSE_PROFILES["independent_v1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-pixel-size-um", type=float, required=True)
    parser.add_argument(
        "--profile", choices=("custom", *CELLPOSE_PROFILES), default="custom"
    )
    parser.add_argument("--downsample", type=int)
    parser.add_argument(
        "--source-roi-xyxy", type=int, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional half-open ROI in original source-image pixels.",
    )
    parser.add_argument("--model")
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--nucleus-expansion-um", type=float)
    parser.add_argument("--flow-threshold", type=float)
    parser.add_argument("--cellprob-threshold", type=float)
    parser.add_argument("--min-size", type=int)
    parser.add_argument("--max-size-fraction", type=float)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--preview-max-side", type=int, default=2048)
    return parser.parse_args()


def resolve_profile_args(args: argparse.Namespace) -> argparse.Namespace:
    expected = (
        CELLPOSE_PROFILES[args.profile]
        if args.profile != "custom"
        else DEFAULT_PARAMETERS
    )
    for name, default in expected.items():
        observed = getattr(args, name)
        if observed is None:
            setattr(args, name, default)
        elif args.profile != "custom" and observed != default:
            raise ValueError(
                f"--{name.replace('_', '-')}={observed} conflicts with "
                f"profile {args.profile} ({default})"
            )
    if args.profile != "custom":
        if args.model_file is None or args.model_sha256 is None:
            raise ValueError(
                f"profile {args.profile} requires --model-file and --model-sha256"
            )
    return args


def package_versions() -> dict[str, str]:
    packages = (
        "cellpose", "numpy", "scipy", "tifffile", "zarr", "torch", "pillow"
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    versions["opencv"] = cv2.__version__
    return versions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
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


class BoundedImage:
    """Tile-addressable RGB image with an optional crop and integer downsample."""

    def __init__(self, path: Path, source_roi_xyxy: tuple[int, int, int, int] | None, downsample: int):
        self.path = path
        self.store = tifffile.imread(path, aszarr=True)
        self.array = zarr.open(self.store, mode="r")
        if len(self.array.shape) != 3 or self.array.shape[2] < 3:
            raise ValueError(f"expected YXS RGB source, observed {self.array.shape}")
        source_height, source_width = int(self.array.shape[0]), int(self.array.shape[1])
        if source_roi_xyxy is None:
            source_roi_xyxy = (0, 0, source_width, source_height)
        x0, y0, x1, y1 = source_roi_xyxy
        if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
            raise ValueError("source ROI is outside the source image")
        if downsample < 1:
            raise ValueError("downsample must be positive")
        self.source_shape_yx = (source_height, source_width)
        self.roi_xyxy = (x0, y0, x1, y1)
        self.downsample = downsample
        self.height = math.ceil((y1 - y0) / downsample)
        self.width = math.ceil((x1 - x0) / downsample)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()

    def read_target_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        x0, y0, x1, y1 = self.roi_xyxy
        raw_x0 = x0 + x * self.downsample
        raw_y0 = y0 + y * self.downsample
        raw_x1 = min(x1, x0 + (x + width) * self.downsample)
        raw_y1 = min(y1, y0 + (y + height) * self.downsample)
        raw = np.asarray(self.array[raw_y0:raw_y1, raw_x0:raw_x1, :3])
        if raw.shape[:2] == (height, width):
            return raw
        return cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)

    def preview(self, max_side: int) -> tuple[np.ndarray, int]:
        step = max(1, math.ceil(max(self.height, self.width) / max_side))
        x0, y0, x1, y1 = self.roi_xyxy
        raw_step = step * self.downsample
        image = np.asarray(self.array[y0:y1:raw_step, x0:x1:raw_step, :3])
        return image, step


def hematoxylin_contrast(rgb: np.ndarray) -> np.ndarray:
    """Ruifrok-style H&E hematoxylin optical density (nuclei are bright)."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("hematoxylin contrast requires RGB input")
    optical_density = -np.log((image[..., :3] + 1.0) / 256.0)
    # Same H stain direction as skimage.color.rgb2hed, expressed in RGB order.
    result = (
        0.650 * optical_density[..., 0]
        + 0.704 * optical_density[..., 1]
        + 0.290 * optical_density[..., 2]
    )
    return np.asarray(result, dtype=np.float32)


def build_cellpose_predictor(
    model_name: str,
    *,
    diameter: float | None,
    flow_threshold: float,
    cellprob_threshold: float,
    min_size: int,
    max_size_fraction: float,
    invert: bool,
    normalize: bool,
    gpu: bool,
) -> Callable[[Any], Any]:
    """Build the small Cellpose adapter required by this Visium workflow."""
    try:
        from cellpose import models
    except ImportError as exc:
        raise RuntimeError(
            "H&E segmentation requires cellpose>=4 in the execution environment"
        ) from exc

    model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)

    def predict(image: Any) -> Any:
        array = np.asarray(image)
        channel_axis = -1 if array.ndim == 3 else None
        masks, _flows, _styles = model.eval(
            array,
            channel_axis=channel_axis,
            diameter=diameter,
            normalize=normalize,
            invert=invert,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            max_size_fraction=max_size_fraction,
        )
        return masks

    return predict


def tile_specs(width: int, height: int, tile_size: int, overlap: int) -> list[dict[str, int | str]]:
    if tile_size < 1 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile size must be positive and overlap smaller than the tile")
    step = tile_size - overlap

    def origins(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = [0]
        while values[-1] + tile_size < length:
            candidate = min(values[-1] + step, length - tile_size)
            if candidate <= values[-1]:
                break
            values.append(candidate)
        return values

    rows: list[dict[str, int | str]] = []
    for y in origins(height):
        for x in origins(width):
            rows.append({
                "tile_id": f"tile_{len(rows):06d}",
                "x": x,
                "y": y,
                "width": min(tile_size, width - x),
                "height": min(tile_size, height - y),
            })
    return rows


def axis_interior(start: int, size: int, slide_size: int, overlap: int) -> tuple[int, int]:
    margin_left = overlap // 2
    margin_right = overlap - margin_left
    step = max(1, size - overlap)
    local0 = 0 if start == 0 else min(margin_left, size)
    local1 = size if start + size >= slide_size else max(local0, size - margin_right)
    if start > 0 and start + size >= slide_size:
        previous_regular_start = ((start - 1) // step) * step
        previous_gap = start - previous_regular_start
        if 0 < previous_gap < step:
            local0 = min(size, max(0, (size - previous_gap) // 2))
    remaining = slide_size - (start + size)
    if 0 < remaining < step:
        local1 = max(local0, min(size, (size + remaining) // 2))
    return local0, local1


def tile_interior(row: dict[str, Any], width: int, height: int, overlap: int) -> dict[str, int]:
    x, y = int(row["x"]), int(row["y"])
    tile_width, tile_height = int(row["width"]), int(row["height"])
    local_x0, local_x1 = axis_interior(x, tile_width, width, overlap)
    local_y0, local_y1 = axis_interior(y, tile_height, height, overlap)
    return {
        "local_x0": local_x0, "local_x1": local_x1,
        "local_y0": local_y0, "local_y1": local_y1,
        "global_x0": x + local_x0, "global_x1": x + local_x1,
        "global_y0": y + local_y0, "global_y1": y + local_y1,
    }


def iter_blocks(height: int, width: int, block_pixels: int = 16_000_000) -> Iterator[tuple[int, int]]:
    rows = max(1, block_pixels // max(1, width))
    for y0 in range(0, height, rows):
        yield y0, min(height, y0 + rows)


def expand_labels_within(labels: Any, distance_px: float) -> np.ndarray:
    """Grow labels into background by a bounded Euclidean distance."""
    from scipy import ndimage

    array = np.asarray(labels)
    background = array == 0
    if distance_px <= 0 or not background.any():
        return array
    distances, (nearest_y, nearest_x) = ndimage.distance_transform_edt(
        background, return_indices=True,
    )
    expanded = array.copy()
    within = background & (distances <= distance_px)
    expanded[within] = array[nearest_y[within], nearest_x[within]]
    return expanded


def expand_labels_streamed(source: Any, target: Any, distance_px: float) -> None:

    height, width = (int(value) for value in source.shape)
    halo = math.ceil(distance_px) + 1
    for y0, y1 in iter_blocks(height, width):
        read_y0 = max(0, y0 - halo)
        read_y1 = min(height, y1 + halo)
        block = np.asarray(source[read_y0:read_y1, :])
        expanded = expand_labels_within(block, distance_px)
        target[y0:y1, :] = expanded[y0 - read_y0:y1 - read_y0]


def label_audit(nucleus: Any, cell: Any) -> dict[str, int]:
    height, width = (int(value) for value in nucleus.shape)
    labels: set[int] = set()
    nucleus_pixels = 0
    cell_pixels = 0
    violations = 0
    for y0, y1 in iter_blocks(height, width):
        n = np.asarray(nucleus[y0:y1, :])
        c = np.asarray(cell[y0:y1, :])
        nucleus_pixels += int(np.count_nonzero(n))
        cell_pixels += int(np.count_nonzero(c))
        violations += int(np.count_nonzero((n > 0) & (c == 0)))
        labels.update(int(value) for value in np.unique(n) if value)
    return {
        "retained_nucleus_labels": len(labels),
        "nucleus_pixels": nucleus_pixels,
        "cell_pixels": cell_pixels,
        "nucleus_outside_cell_pixels": violations,
    }


def write_preview(path: Path, image: np.ndarray, nucleus: Any, cell: Any, step: int) -> None:
    labels_n = np.asarray(nucleus[::step, ::step])
    labels_c = np.asarray(cell[::step, ::step])
    height = min(image.shape[0], labels_n.shape[0])
    width = min(image.shape[1], labels_n.shape[1])
    rgb = np.asarray(image[:height, :width, :3], dtype=np.uint8).copy()
    labels_n = labels_n[:height, :width]
    labels_c = labels_c[:height, :width]
    kernel = np.ones((3, 3), np.uint8)
    nucleus_edge = cv2.morphologyEx((labels_n > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    cell_edge = cv2.morphologyEx((labels_c > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    rgb[cell_edge] = (30, 144, 255)
    rgb[nucleus_edge] = (255, 40, 40)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> int:
    try:
        args = resolve_profile_args(parse_args())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.image.is_file():
        raise SystemExit(f"missing image: {args.image}")
    model_file_sha256 = None
    if args.model_file is not None:
        if not args.model_file.is_file():
            raise SystemExit(f"missing Cellpose model file: {args.model_file}")
        model_file_sha256 = sha256(args.model_file)
        if args.model_sha256 and model_file_sha256 != args.model_sha256:
            raise SystemExit(
                f"Cellpose model hash drift: {model_file_sha256} != "
                f"{args.model_sha256}"
            )
    if args.source_pixel_size_um <= 0 or args.nucleus_expansion_um < 0:
        raise SystemExit("pixel size must be positive and expansion nonnegative")
    target_pixel_size_um = args.source_pixel_size_um * args.downsample
    if target_pixel_size_um > 1.0:
        raise SystemExit(
            f"Cellpose target resolution {target_pixel_size_um:.6g} um/px exceeds the 1 um/px guardrail"
        )
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    source = BoundedImage(
        args.image,
        tuple(args.source_roi_xyxy) if args.source_roi_xyxy else None,
        args.downsample,
    )
    started = time.monotonic()
    try:
        predictor = build_cellpose_predictor(
            model_name=str(args.model_file) if args.model_file else args.model,
            diameter=None,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            max_size_fraction=args.max_size_fraction,
            invert=False,
            normalize=True,
            gpu=not args.cpu,
        )
        masks = args.out_dir / "masks"
        masks.mkdir()
        nucleus_path = masks / "nucleus_labels.zarr"
        cell_path = masks / "cell_labels.zarr"
        nucleus = zarr.open(
            nucleus_path, mode="w", shape=(source.height, source.width),
            chunks=(1024, 1024), dtype="uint32", fill_value=0,
        )
        tile_rows = tile_specs(source.width, source.height, args.tile_size, args.overlap)
        offset = 0
        for index, row in enumerate(tile_rows):
            tile_started = time.monotonic()
            rgb = source.read_target_tile(
                int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"]),
            )
            contrast = hematoxylin_contrast(rgb)
            local = np.asarray(predictor(contrast), dtype=np.uint32)
            if local.shape != contrast.shape:
                raise RuntimeError(f"Cellpose changed tile shape {contrast.shape} to {local.shape}")
            tile_max = int(local.max()) if local.size else 0
            if tile_max:
                local = np.where(local > 0, local + offset, 0).astype(np.uint32)
            interior = tile_interior(row, source.width, source.height, args.overlap)
            nucleus[
                interior["global_y0"]:interior["global_y1"],
                interior["global_x0"]:interior["global_x1"],
            ] = local[
                interior["local_y0"]:interior["local_y1"],
                interior["local_x0"]:interior["local_x1"],
            ]
            offset += tile_max
            row.update({
                "cellpose_labels_before_overlap_trim": tile_max,
                "elapsed_seconds": time.monotonic() - tile_started,
                "status": "succeeded",
            })
            print(
                f"[{index + 1}/{len(tile_rows)}] {row['tile_id']} labels={tile_max} "
                f"elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )

        cell = zarr.open(
            cell_path, mode="w", shape=(source.height, source.width),
            chunks=(1024, 1024), dtype="uint32", fill_value=0,
        )
        expansion_px = args.nucleus_expansion_um / target_pixel_size_um
        expand_labels_streamed(nucleus, cell, expansion_px)
        audit = label_audit(nucleus, cell)
        if audit["retained_nucleus_labels"] == 0:
            raise RuntimeError("Cellpose produced no retained nuclei")
        if audit["nucleus_outside_cell_pixels"] != 0:
            raise RuntimeError("nucleus/cell hierarchy failure")
        preview_image, preview_step = source.preview(args.preview_max_side)
        preview_path = args.out_dir / "cellpose_overlay.png"
        write_preview(preview_path, preview_image, nucleus, cell, preview_step)
    finally:
        source.close()

    tile_path = args.out_dir / "tile_status.tsv"
    columns = []
    for row in tile_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with tile_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tile_rows)

    summary = {
        "schema": "visium_hd_processing.he_cellpose_segmentation.v1",
        "input_contract": {
            "image": str(args.image.resolve()),
            "image_sha256": sha256(args.image),
            "source_shape_yx": list(source.source_shape_yx),
            "source_roi_xyxy": list(source.roi_xyxy),
            "source_pixel_size_um": args.source_pixel_size_um,
            "downsample": args.downsample,
            "segmentation_shape_yx": [source.height, source.width],
            "segmentation_pixel_size_um": target_pixel_size_um,
            "expression_or_vendor_segmentation_loaded": False,
        },
        "method": {
            "engine": "Cellpose",
            "engine_version": __import__("cellpose").version,
            "profile": args.profile,
            "model": args.model,
            "model_file": str(args.model_file.resolve()) if args.model_file else None,
            "model_file_sha256": model_file_sha256,
            "channel": "Ruifrok-style hematoxylin optical density",
            "diameter": None,
            "flow_threshold": args.flow_threshold,
            "cellprob_threshold": args.cellprob_threshold,
            "min_size": args.min_size,
            "max_size_fraction": args.max_size_fraction,
            "gpu": not args.cpu,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "overlap_policy": "half-overlap deterministic interior ownership",
            "cell_domain": "nearest-nucleus label expansion",
            "nucleus_expansion_um": args.nucleus_expansion_um,
            "nucleus_expansion_px": expansion_px,
        },
        "counts": audit | {
            "image_pixels": source.height * source.width,
            "nucleus_area_fraction": audit["nucleus_pixels"] / (source.height * source.width),
            "cell_area_fraction": audit["cell_pixels"] / (source.height * source.width),
            "tiles": len(tile_rows),
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "packages": package_versions(),
            "container_image_id": os.environ.get("VISIUM_HD_CELLPOSE_IMAGE_ID"),
        },
        "outputs": {
            "nucleus_labels.zarr": tree_sha256(nucleus_path),
            "cell_labels.zarr": tree_sha256(cell_path),
            preview_path.name: sha256(preview_path),
            tile_path.name: sha256(tile_path),
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "checksums.sha256").write_text(
        f"{sha256(summary_path)}  summary.json\n"
        f"{sha256(tile_path)}  tile_status.tsv\n"
        f"{sha256(preview_path)}  cellpose_overlay.png\n"
        f"{tree_sha256(nucleus_path)}  masks/nucleus_labels.zarr\n"
        f"{tree_sha256(cell_path)}  masks/cell_labels.zarr\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
