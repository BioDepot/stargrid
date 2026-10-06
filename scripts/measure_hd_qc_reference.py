#!/usr/bin/env python3
"""Exploratory Visium HD QC survey, using output files only (2026-10-04).

Frozen definitions: docs/the corresponding analysis record.
Run through run_off_benchmark_cores.sh; one slide/source per output directory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import fft, ndimage

import measure_hd_nucleus_dip_and_edge_profiles as legacy
from prepare_hd_same_run_offsets import h5_totals, square_indices

RADIUS = 13
SEARCH = 8
REGION_MIN = 50_000


def parquet(path, columns=None):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            return pd.read_parquet(handle, columns=columns)
    return pd.read_parquet(path, columns=columns)


def tissue_mask(path, width):
    t = parquet(path)
    if 'barcode' in t:
        square_indices(t.barcode, width)  # Reject 8 um tables even if coordinates fit.
    rows, cols = t.array_row.to_numpy(), t.array_col.to_numpy()
    idx = rows * width + cols
    if len(idx) != width * width or len(np.unique(idx)) != width * width:
        raise ValueError('tissue positions must cover every 2 um capture position')
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= width) or np.any(cols >= width):
        raise ValueError('tissue position outside grid')
    if not np.isin(t.in_tissue, [0, 1]).all():
        raise ValueError('invalid tissue flag')
    result = np.zeros(width * width, bool)
    result[idx] = t.in_tissue.to_numpy() == 1
    return result.reshape(width, width)


def published_nuclei(path, width):
    t = parquet(path, ['square_002um', 'in_nucleus'])
    idx = square_indices(t.square_002um, width)
    if len(np.unique(idx)) != len(idx):
        raise ValueError('duplicate nucleus-map square')
    out = np.zeros(width * width, bool)
    out[idx] = t.in_nucleus.fillna(False).to_numpy(bool)
    return out.reshape(width, width)


def load_counts(spec, width):
    path = Path(spec['path'])
    if spec['kind'] == 'npz':
        with np.load(path) as a:
            counts = a[spec.get('key', 'total')].reshape(width, width).astype(float)
    elif spec['kind'] == 'h5':
        counts = h5_totals(path, width).reshape(width, width)
    else:
        raise ValueError('counts kind must be npz or h5')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('counts must be finite and nonnegative')
    if not np.equal(counts, np.rint(counts)).all():
        raise ValueError('this survey freezes integer hard/deposited counts')
    return counts


def shift_surface(nuc, counts, support, radius=RADIUS):
    """Exact integer sums via padded FFT; no wrap at the capture-area edge."""
    shape = tuple(fft.next_fast_len(n + radius) for n in nuc.shape)
    nf = fft.rfft2(nuc.astype(float), s=shape)
    at = np.arange(-radius, radius + 1)
    ix = np.ix_(at % shape[0], at % shape[1])
    values = []
    for target in (counts * support, support.astype(float)):
        corr = fft.irfft2(np.conj(nf) * fft.rfft2(target, s=shape), s=shape)[ix]
        rounded = np.rint(corr)
        if np.max(np.abs(corr - rounded)) > 0.01:
            raise ValueError('FFT integer-sum residual exceeds 0.01')
        values.append(rounded)
    out = np.full(values[0].shape, np.nan)
    np.divide(values[0], values[1], out=out, where=values[1] > 0)
    return out


def contrast(surface, dy, dx, radius=RADIUS):
    yy, xx = np.mgrid[-5:6, -5:6]
    ring = np.maximum(np.abs(yy), np.abs(xx))
    values = surface[radius + dy + yy, radius + dx + xx]
    ref = np.median(values[(ring >= 3) & (ring <= 5)])
    centre = surface[radius + dy, radius + dx]
    return float(1 - centre / ref) if np.isfinite(ref) and ref > 0 else None


def feature(surface, radius=RADIUS, search=SEARCH):
    candidates = []
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            c = contrast(surface, dy, dx, radius)
            if c is not None and np.isfinite(c):
                candidates.append((c, dy, dx))
    if not candidates:
        return {'dy': None, 'dx': None, 'contrast': None, 'sign': 'unresolved',
                'border': False, 'width_um': None, 'ring_profile': []}
    c, dy, dx = min(candidates, key=lambda p: (-abs(p[0]), max(abs(p[1]), abs(p[2])), p[1], p[2]))
    centre = float(surface[radius + dy, radius + dx])
    yy, xx = np.mgrid[-5:6, -5:6]
    rr = np.maximum(np.abs(yy), np.abs(xx))
    patch = surface[radius + dy + yy, radius + dx + xx]
    reference = float(np.median(patch[(rr >= 3) & (rr <= 5)]))
    half = (centre + reference) / 2
    profile = [{'radius_squares': r, 'radius_um': 2*r, 'median_count': float(np.median(patch[rr == r]))}
               for r in range(6)]
    crossing = [p['radius_um'] for p in profile[1:]
                if (reference > centre and p['median_count'] >= half)
                or (reference < centre and p['median_count'] <= half)]
    return {'dy': dy, 'dx': dx, 'dy_um': 2*dy, 'dx_um': 2*dx,
            'contrast': c, 'sign': 'dip' if c > 0 else 'peak' if c < 0 else 'flat',
            'border': max(abs(dy), abs(dx)) == search, 'mean_count': centre,
            'ring_median': reference, 'width_um': min(crossing) if crossing else None,
            'ring_profile': profile}


def region_features(nuc, counts, tissue, whole, minimum=REGION_MIN):
    edges_y = np.linspace(0, nuc.shape[0], 4).astype(int)
    edges_x = np.linspace(0, nuc.shape[1], 4).astype(int)
    regions, surfaces = {}, {}
    for iy in range(3):
        for ix in range(3):
            y0, y1 = edges_y[iy:iy+2]; x0, x1 = edges_x[ix:ix+2]
            name = f'r{iy}c{ix}'
            n = int((nuc[y0:y1, x0:x1] & tissue[y0:y1, x0:x1]).sum())
            row = {'nucleus_squares_in_tissue': n, 'eligible': n >= minimum, 'agrees': False}
            if n >= minimum:
                a, b = max(0, y0-RADIUS), min(nuc.shape[0], y1+RADIUS)
                c, d = max(0, x0-RADIUS), min(nuc.shape[1], x1+RADIUS)
                local = np.zeros((b-a, d-c), bool)
                local[y0-a:y1-a, x0-c:x1-c] = nuc[y0:y1, x0:x1]
                s = shift_surface(local, counts[a:b, c:d], tissue[a:b, c:d])
                f = feature(s)
                row['feature'] = f
                row['contrast_at_zero'] = contrast(s, 0, 0)
                row['agrees'] = bool(f['dy'] is not None and whole['dy'] is not None
                    and f['sign'] == whole['sign'] and f['sign'] in ('dip', 'peak')
                    and max(abs(f['dy']-whole['dy']), abs(f['dx']-whole['dx'])) <= 1)
                surfaces[name] = s
            else:
                row['reason'] = 'fewer than 50000 nucleus positions in tissue'
            regions[name] = row
    eligible = sum(r['eligible'] for r in regions.values())
    agrees = sum(r['agrees'] for r in regions.values())
    fraction = agrees / eligible if eligible else None
    resolved = bool(whole['contrast'] is not None and abs(whole['contrast']) >= .03
                    and not whole['border'] and eligible and fraction >= .75)
    return regions, surfaces, {'eligible_regions': eligible, 'agreeing_regions': agrees,
        'regional_agreement': fraction, 'resolved': resolved,
        'resolved_position_squares': [whole['dy'], whole['dx']] if resolved else None}


def edge_profiles(counts, tissue, nucleus=None, complete=True):
    summary = {'tissue_share_of_squares': float(tissue.mean()),
               'mean_count_in_tissue': float(counts[tissue].mean()) if tissue.any() else None}
    off, near = [], []
    if complete:
        outside = ~tissue
        inside_mean = summary['mean_count_in_tissue']
        off_mean = float(counts[outside].mean()) if outside.any() else None
        total = float(counts.sum())
        summary.update(mean_count_off_tissue=off_mean,
            share_of_counts_off_tissue=float(counts[outside].sum()/total) if total else None,
            off_to_in_mean_ratio=off_mean/inside_mean if off_mean is not None and inside_mean else None)
        d_off = ndimage.distance_transform_edt(outside) * 2
        d_in = ndimage.distance_transform_edt(tissue) * 2
        ref_mask = tissue & (d_in >= 20) & (d_in < 60)
        ref = float(counts[ref_mask].mean()) if ref_mask.any() else None
        summary['reference_mean_count_20_60um_inside_edge'] = ref
        for lo, hi in legacy.OFF:
            m = outside & (d_off > lo) & (d_off <= hi)
            mean = float(counts[m].mean()) if m.any() else None
            off.append({'bin': f'{lo:.0f}-{hi:.0f} um' if hi < 1e8 else 'beyond 500 um',
                'lower_exclusive_um': lo, 'upper_inclusive_um': hi if hi < 1e8 else None,
                'mean_count': mean, 'relative_to_inside': mean/ref if mean is not None and ref else None,
                'squares': int(m.sum())})
        del d_off, d_in
    else:
        summary['q1_q2_unavailable_reason'] = 'counts do not cover off-tissue positions'
    if nucleus is not None:
        d_out = ndimage.distance_transform_edt(~nucleus) * 2
        d_in = ndimage.distance_transform_edt(nucleus) * 2
        masks = [('nucleus, at least 4 um inside', nucleus & (d_in >= 4)),
                 ('nucleus, 2-4 um inside', nucleus & (d_in >= 2.1) & (d_in < 4)),
                 ('nucleus, edge square', nucleus & (d_in < 2.1))]
        for lo, hi in legacy.NEAR:
            masks.append((f'outside nucleus, {lo:.0f}-{hi:.0f} um' if hi < 1e8 else 'outside nucleus, beyond 50 um',
                          (~nucleus) & (d_out > lo) & (d_out <= hi)))
        for name, m in masks:
            m &= tissue
            near.append({'bin': name, 'mean_count': float(counts[m].mean()) if m.any() else None,
                         'squares': int(m.sum())})
    return summary, off, near


def legacy_gate(old_dir, summary, surfaces, off, near):
    old_dir = Path(old_dir)
    old = json.loads((old_dir/'summary.json').read_text())
    checked = []
    for key, value in old.items():
        if key in ('status', 'script_sha256', 'maps', 'inputs'):
            continue
        if summary[key] != value:
            raise ValueError(f'gate mismatch: {key}: {summary[key]} != {value}')
        checked.append(key)
    for path, sha in old['inputs'].items():
        if summary['inputs'].get(path) != sha:
            raise ValueError(f'gate input mismatch: {path}')
    for name, fields in old['maps'].items():
        for key, value in fields.items():
            if summary['maps'][name][key] != value:
                raise ValueError(f'gate map mismatch: {name}.{key}')
            checked.append(f'maps.{name}.{key}')
    with (old_dir/'shift_surface.tsv').open() as h:
        for row in csv.DictReader(h, delimiter='\t'):
            got = surfaces[row['nucleus_map']][int(row['dy'])+RADIUS, int(row['dx'])+RADIUS]
            if got != float(row['mean_count']):
                raise ValueError(f'gate surface mismatch: {row}')
    for filename, rows in [('off_tissue_profile.tsv', off), ('nucleus_distance_profile.tsv', near)]:
        lookup = {r['bin']: r for r in rows}
        with (old_dir/filename).open() as h:
            for row in csv.DictReader(h, delimiter='\t'):
                for k, value in row.items():
                    if k not in ('slide', 'bin') and float(value) != lookup[row['bin']][k]:
                        raise ValueError(f'gate profile mismatch: {filename} {row["bin"]} {k}')
    expected = {'published': (0, 0, '0.1496'), 'native_cellpose': (-3, -3, '0.1402'),
                'native_stardist': (-3, -3, '0.1500')} if 'ovarian' in old['slide'] else {
                'published': (0, 0, '0.0574'), 'native_stardist': (0, 0, '0.0586')}
    controls = {}
    for name, (dy, dx, value) in expected.items():
        f = feature(surfaces[name], search=3)
        if (f['dy'], f['dx'], f'{f["contrast"]:.4f}') != (dy, dx, value):
            raise ValueError(f'gate contrast mismatch: {name}: {f}')
        controls[name] = f
    return {'passed': True, 'shared_summary_fields': checked,
            'original_surfaces_and_profiles_match_exactly': True, 'legacy_search_controls': controls,
            'accepted_summary_sha256': legacy.sha256(old_dir/'summary.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):
        raise SystemExit('run through scripts/run_off_benchmark_cores.sh')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    c = json.loads(a.config.read_text())
    width = c.get('width', 3350)
    inputs = {}
    def record(path):
        path = str(path)
        inputs[path] = legacy.sha256(Path(path))
    record(a.config)
    record(c['counts']['path']); record(c['tissue_positions'])
    tissue = tissue_mask(c['tissue_positions'], width)
    counts = load_counts(c['counts'], width)
    maps = {}
    if c.get('barcode_mappings'):
        record(c['barcode_mappings'])
        maps['published'] = published_nuclei(c['barcode_mappings'], width)
    for name, spec in c.get('extra_nuclei', {}).items():
        record(spec['path'])
        with np.load(spec['path']) as data:
            maps[name] = data[spec['key']].reshape(width, width) > 0
    for path in c.get('supporting_records', []):
        record(path)
    primary = c.get('primary_nucleus_map', 'published' if 'published' in maps else None)
    if primary is not None and primary not in maps:
        raise ValueError('primary nucleus map missing')
    summary, off, near = edge_profiles(counts, tissue, maps.get(primary), c['counts_complete'])
    summary.update(schema='visium_hd_processing.qc_reference.v1', slide=c['slide'],
        metadata=c, maps={}, inputs=inputs, script_sha256=legacy.sha256(Path(__file__)),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        command=sys.argv, cpus=os.environ['OFF_BENCH_CPUS_USED'],
        status='exploratory survey; provisional definitions; not a paper result',
        count_total=float(counts.sum()), primary_nucleus_map=primary,
        surface_radius_squares=RADIUS, feature_search_squares=SEARCH,
        off_tissue_profile=off, nucleus_distance_profile=near)
    surfaces, regional_arrays, rows, regional_rows = {}, {}, [], []
    for name, nuc in maps.items():
        print(f'measuring {c["slide"]}: {name}', flush=True)
        s = shift_surface(nuc, counts, tissue)
        surfaces[name] = s
        legacy_s = s[RADIUS-8:RADIUS+9, RADIUS-8:RADIUS+9]
        mini = np.unravel_index(np.nanargmin(legacy_s), legacy_s.shape)
        maxi = np.unravel_index(np.nanargmax(legacy_s), legacy_s.shape)
        f = feature(s)
        regions, region_arrays, stability = region_features(nuc, counts, tissue, f)
        m = {'nucleus_squares_in_tissue': int((nuc & tissue).sum()),
             'nucleus_share_of_tissue_squares': float((nuc & tissue).sum()/tissue.sum()),
             'mean_count_at_zero_shift': float(s[RADIUS, RADIUS]),
             'minimum': {'dy': int(mini[0]-8), 'dx': int(mini[1]-8), 'mean_count': float(legacy_s[mini])},
             'maximum': {'dy': int(maxi[0]-8), 'dx': int(maxi[1]-8), 'mean_count': float(legacy_s[maxi])},
             'legacy_extrema_search_squares': 8, 'contrast_at_zero': contrast(s, 0, 0),
             'feature': f, 'regions': regions, **stability}
        if name != 'published' and 'published' in maps:
            m['displacement_of_published_map'] = legacy.map_displacement(nuc, maps['published'])
        summary['maps'][name] = m
        for dy in range(-RADIUS, RADIUS+1):
            for dx in range(-RADIUS, RADIUS+1):
                rows.append({'slide': c['slide'], 'nucleus_map': name, 'dy': dy, 'dx': dx,
                             'mean_count': s[dy+RADIUS, dx+RADIUS]})
        for region, r in regions.items():
            regional_rows.append({'nucleus_map': name, 'region': region,
                **{k:v for k,v in r.items() if k != 'feature'},
                **{k:v for k,v in r.get('feature', {}).items() if k != 'ring_profile'}})
        for region, array in region_arrays.items():
            regional_arrays[f'{name}_{region}'] = array
        print(json.dumps({'map': name, 'feature': {k:v for k,v in f.items() if k != 'ring_profile'}, **stability}), flush=True)
    if c.get('legacy_gate_dir'):
        summary['gate'] = legacy_gate(c['legacy_gate_dir'], summary, surfaces, off, near)
        print('legacy gate: exact match', flush=True)
    findings = []
    if 'published' in summary['maps']:
        m = summary['maps']['published']
        if m['resolved'] and max(abs(m['feature']['dy']), abs(m['feature']['dx'])) > 1:
            findings.append('resolved deposited nucleus feature more than one square from zero; stop and report')
    summary['stop_findings'] = findings
    for filename, records in [('shift_surface.tsv', rows), ('regional_features.tsv', regional_rows),
        ('off_tissue_profile.tsv', off), ('nucleus_distance_profile.tsv', near)]:
        if records:
            pd.DataFrame(records).to_csv(a.out_dir/filename, sep='\t', index=False)
    np.savez_compressed(a.out_dir/'surfaces.npz', **surfaces, **regional_arrays)
    summary['outputs'] = {p.name: legacy.sha256(p) for p in a.out_dir.iterdir() if p.is_file()}
    (a.out_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if findings:
        print('\n'.join(findings), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
