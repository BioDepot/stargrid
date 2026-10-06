#!/usr/bin/env python3
"""Bounded-memory full-capture StarDist using the selected ENACT parameters.

prepare: image-only capture crop, resampling and exact global percentiles.
predict: frozen StarDist model and block geometry, with disk-backed labels.
project: independent native registration maps square centres into these labels.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import numpy as np

MODEL_HASHES = {
    'config.json': '47531372d6436172c94be44882c4c74f8c8edeb4843dba47a89a7782673daccd',
    'thresholds.json': '05a10c871bb32735e86a1ed2ff5c756e7d207beb1142b6d3c470198b41e93fa6',
    'weights_best.h5': 'ce23a21f09511132c51e2c7b077054355b6b0c1cbece61066dd280054ee7178f',
}
PARAMETERS = {'block_size': 4096, 'prob_thresh': .005, 'nms_thresh': .001,
              'min_overlap': 28, 'context': 128, 'n_tiles': (4, 4, 1)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def project(matrix, xy):
    p = np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(matrix).T
    if np.any(np.abs(p[:, 2]) < 1e-12):
        raise ValueError('projective pole in capture area')
    return p[:, :2] / p[:, 2, None]


def histogram_percentile(hist, percentile):
    """NumPy linear percentile of all uint8 values without loading the image."""
    if not hist.sum():
        raise ValueError('empty image')
    position = (int(hist.sum()) - 1) * percentile / 100
    cumulative = np.cumsum(hist)
    low, high = int(np.floor(position)), int(np.ceil(position))
    a, b = np.searchsorted(cumulative, [low, high], side='right')
    return float(a + (b - a) * (position - low))


class NormalizedImage:
    def __init__(self, image, low, high):
        self.image, self.low, self.high = image, np.float32(low), np.float32(high)
        self.shape, self.ndim, self.dtype = image.shape, image.ndim, np.dtype('float32')

    def __getitem__(self, item):
        return (np.asarray(self.image[item], np.float32) - self.low) / (self.high - self.low + np.float32(1e-20))


def prepare(a):
    import cv2
    import tifffile
    import zarr
    a.out_dir.mkdir(parents=True, exist_ok=False)
    transform_record = json.loads(a.grid_transform.read_text())
    transform = np.asarray(transform_record['grid_to_fullres'])
    corners = project(transform, np.array([[0, 0], [3349, 0], [0, 3349], [3349, 3349]]))
    store = tifffile.imread(a.image, aszarr=True)
    source = zarr.open(store, mode='r')
    if source.ndim != 3 or source.shape[2] < 3 or source.dtype != np.uint8:
        raise ValueError('expected uint8 YXC RGB H&E image')
    margin_px = a.margin_um / a.source_pixel_um
    x0, y0 = np.maximum(0, np.floor(corners.min(0) - margin_px)).astype(int)
    x1, y1 = np.minimum([source.shape[1], source.shape[0]], np.ceil(corners.max(0) + margin_px)).astype(int)
    scale = a.source_pixel_um / a.target_pixel_um
    shape = (int(math.ceil((y1 - y0) * scale)), int(math.ceil((x1 - x0) * scale)), 3)
    if min(shape) < 1:
        raise ValueError('capture crop misses the image')
    rgb = np.lib.format.open_memmap(a.out_dir / 'rgb.npy', mode='w+', dtype=np.uint8, shape=shape)
    hist = np.zeros(256, np.int64)
    for yy in range(0, shape[0], 512):
        ye = min(yy + 512, shape[0])
        for xx in range(0, shape[1], 4096):
            xe = min(xx + 4096, shape[1])
            sx = (np.arange(xx, xe) + .5) / scale - .5 + x0
            sy = (np.arange(yy, ye) + .5) / scale - .5 + y0
            left, top = max(0, int(np.floor(sx.min()))), max(0, int(np.floor(sy.min())))
            right = min(source.shape[1], int(np.ceil(sx.max())) + 2)
            bottom = min(source.shape[0], int(np.ceil(sy.max())) + 2)
            raw = np.asarray(source[top:bottom, left:right, :3])
            mapx, mapy = np.meshgrid((sx - left).astype(np.float32), (sy - top).astype(np.float32))
            block = cv2.remap(raw, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rgb[yy:ye, xx:xe] = block
            hist += np.bincount(block.ravel(), minlength=256)
        print(f'prepared rows {ye}/{shape[0]}', flush=True)
    rgb.flush()
    low, high = [histogram_percentile(hist, p) for p in (5, 95)]
    if high <= low:
        raise ValueError('degenerate 5th/95th percentiles')
    summary = {'schema': 'visium_hd_processing.stardist_prepared_image.v1',
        'image': str(a.image), 'image_sha256': sha256(a.image), 'source_shape_yxc': list(source.shape),
        'source_pixel_um': a.source_pixel_um, 'target_pixel_um': a.target_pixel_um,
        'source_roi_xyxy': [int(x0), int(y0), int(x1), int(y1)], 'target_shape_yxc': shape,
        'source_to_target_scale': scale, 'margin_um': a.margin_um,
        'resampling': 'OpenCV INTER_LINEAR, pixel-centre mapping; replicated image boundary',
        'normalization': {'percentiles': [5, 95], 'low': low, 'high': high, 'axis': None},
        'grid_to_fullres': transform.tolist(), 'grid_transform_sha256': sha256(a.grid_transform),
        'expression_or_vendor_segmentation_loaded': False, 'rgb_sha256': sha256(a.out_dir / 'rgb.npy'),
        'script_sha256': sha256(Path(__file__))}
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    store.close()


def predict(a):
    for name, expected in MODEL_HASHES.items():
        if sha256(a.model_dir / name) != expected:
            raise ValueError(f'model hash mismatch: {name}')
    import tensorflow as tf
    from stardist.models import StarDist2D
    devices = tf.config.list_physical_devices('GPU')
    if not devices and not a.cpu:
        raise RuntimeError('GPU unavailable; complete environment preflight before segmentation')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    prepared = json.loads((a.prepared / 'summary.json').read_text())
    if sha256(a.prepared / 'rgb.npy') != prepared['rgb_sha256']:
        raise ValueError('prepared RGB checksum mismatch')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    rgb = np.load(a.prepared / 'rgb.npy', mmap_mode='r')
    norm = prepared['normalization']
    image = NormalizedImage(rgb, norm['low'], norm['high'])
    labels = np.lib.format.open_memmap(a.out_dir / 'nucleus_labels.npy', mode='w+', dtype=np.int32, shape=rgb.shape[:2])
    model = StarDist2D(None, name=a.model_dir.name, basedir=str(a.model_dir.parent))
    start = time.monotonic()
    _, polygons = model.predict_instances_big(image, axes='YXC', labels_out=labels,
                                              normalizer=None, **PARAMETERS)
    labels.flush()
    np.savez_compressed(a.out_dir / 'polygons.npz', **polygons)
    versions = {p: importlib.metadata.version(p) for p in ['stardist', 'csbdeep', 'tensorflow', 'numpy']}
    summary = {'schema': 'visium_hd_processing.stardist_segmentation.v1', 'method': PARAMETERS,
        'model': '2D_versatile_he', 'model_sha256': MODEL_HASHES, 'versions': versions,
        'gpu': bool(devices), 'seconds': time.monotonic() - start,
        'nuclei': len(polygons['prob']), 'prepared_summary_sha256': sha256(a.prepared / 'summary.json'),
        'outputs': {p.name: sha256(p) for p in a.out_dir.glob('*.np*')},
        'script_sha256': sha256(Path(__file__))}
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def grid(a):
    info = json.loads((a.prepared / 'summary.json').read_text())
    accepted = json.loads((a.segmentation / 'summary.json').read_text())
    if sha256(a.segmentation / 'nucleus_labels.npy') != accepted['outputs']['nucleus_labels.npy']:
        raise ValueError('nucleus labels checksum mismatch')
    labels = np.load(a.segmentation / 'nucleus_labels.npy', mmap_mode='r')
    matrix = np.asarray(info['grid_to_fullres'])
    nucleus = np.zeros(3350 ** 2, np.int64)
    supported = np.zeros(len(nucleus), bool)
    x0, y0, _, _ = info['source_roi_xyxy']
    for start in range(0, len(nucleus), 100000):
        idx = np.arange(start, min(start + 100000, len(nucleus)))
        xy = project(matrix, np.column_stack([idx % 3350, idx // 3350]))
        pixels = np.floor((xy - [x0, y0]) * info['source_to_target_scale']).astype(np.int64)
        good = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0) & (pixels[:, 0] < labels.shape[1]) & (pixels[:, 1] < labels.shape[0])
        nucleus[idx[good]] = labels[pixels[good, 1], pixels[good, 0]]
        supported[idx[good]] = True
    a.out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.out_dir / 'grid_labels.npz', label=nucleus, nucleus_id=nucleus, supported=supported, nx=3350, ny=3350)
    summary = {'nuclei_on_grid': int(len(np.unique(nucleus[nucleus > 0]))), 'nucleus_squares': int((nucleus > 0).sum()),
        'supported_squares': int(supported.sum()), 'grid_to_fullres': matrix.tolist(),
        'prepared_summary_sha256': sha256(a.prepared / 'summary.json'),
        'segmentation_summary_sha256': sha256(a.segmentation / 'summary.json'),
        'grid_labels_sha256': sha256(a.out_dir / 'grid_labels.npz'), 'script_sha256': sha256(Path(__file__))}
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--image', type=Path, required=True)
    prep.add_argument('--source-pixel-um', type=float, required=True)
    prep.add_argument('--target-pixel-um', type=float, default=.1369)
    prep.add_argument('--margin-um', type=float, default=32)
    prep.add_argument('--grid-transform', type=Path, required=True)
    pred = sub.add_parser('predict')
    pred.add_argument('--prepared', type=Path, required=True)
    pred.add_argument('--model-dir', type=Path, default=Path.home() / '.keras/models/StarDist2D/2D_versatile_he')
    pred.add_argument('--cpu', action='store_true')
    proj = sub.add_parser('project')
    proj.add_argument('--prepared', type=Path, required=True)
    proj.add_argument('--segmentation', type=Path, required=True)
    for parser in (prep, pred, proj):
        parser.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    {'prepare': prepare, 'predict': predict, 'project': grid}[args.stage](args)


if __name__ == '__main__': main()
