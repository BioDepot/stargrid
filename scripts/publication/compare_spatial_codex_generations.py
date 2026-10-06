#!/usr/bin/env python3
"""Compare accepted CODEX generations without rescoring or selecting a scale."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


POLICIES = ('strict', 'soft_expected', 'hard', 'gated_hard')
METRICS = ('macro_roc_auc', 'median_roc_auc', 'direct_panel_rna_mass', 'method_evaluation_mass')
COHORT = ('genes', 'mapping', 'reference_marker_available', 'evaluated_bins',
          'protein_positive_bins', 'protein_positive_threshold')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in out:
            raise ValueError(f'duplicate key: {key}')
        out[key] = row
    return out


def load(item):
    root = Path(item['root'])
    summary_path = root / 'summary.json'
    if sha(summary_path) != item['summary_sha256']:
        raise ValueError('summary checksum drift')
    summary = json.loads(summary_path.read_text())
    tables = {}
    for name in ('method_summary.tsv', 'marker_metrics.tsv'):
        path = root / name
        if sha(path) != summary['outputs'][name]['sha256']:
            raise ValueError(f'accepted table checksum drift: {path}')
        with path.open() as stream:
            tables[name] = list(csv.DictReader(stream, delimiter='\t'))
    return summary, tables


def compare(old, new, old_tables, new_tables, old_prefix, new_prefix, arm):
    for key in ('parameters', 'coordinate_serialization'):
        if old[key] != new[key]:
            raise ValueError(f'frozen {key} differs')
    for key in ('codex_h5ad', 'probe_csv', 'space_ranger_h5ad', 'space_ranger_filtered_probe_set'):
        if old['inputs'][key]['sha256'] != new['inputs'][key]['sha256']:
            raise ValueError(f'frozen input differs: {key}')
    old_methods = index(old_tables['method_summary.tsv'], ('method', 'bin_size_um'))
    new_methods = index(new_tables['method_summary.tsv'], ('method', 'bin_size_um'))
    old_markers = index(old_tables['marker_metrics.tsv'], ('method', 'bin_size_um', 'protein'))
    new_markers = index(new_tables['marker_metrics.tsv'], ('method', 'bin_size_um', 'protein'))
    rows = []
    for scale in old['parameters']['bin_sizes_um']:
        scale = str(scale)
        vendor_key = ('space_ranger_2020a', scale)
        if old_methods[vendor_key] != new_methods[vendor_key]:
            raise ValueError(f'vendor result differs: {scale}')
        for policy in POLICIES:
            a, b = (old_prefix + policy, scale), (new_prefix + policy, scale)
            before, after = old_methods[a], new_methods[b]
            if before['markers_scored'] != after['markers_scored']:
                raise ValueError('marker count differs')
            prior = {k[2]: v for k, v in old_markers.items() if k[:2] == a}
            current = {k[2]: v for k, v in new_markers.items() if k[:2] == b}
            if prior.keys() != current.keys():
                raise ValueError('marker identity differs')
            for protein in prior:
                if any(prior[protein][k] != current[protein][k] for k in COHORT):
                    raise ValueError(f'protein/cohort differs: {protein}, {scale}')
            for metric in METRICS:
                left, right = float(before[metric]), float(after[metric])
                if not math.isfinite(left) or not math.isfinite(right):
                    raise ValueError(f'nonfinite metric: {metric}')
                rows.append(dict(reference_arm=arm, policy=policy, bin_size_um=scale,
                                 metric=metric, old_value=left, new_value=right,
                                 absolute_change=right-left,
                                 relative_change_percent=100*(right-left)/abs(left) if left else 'not_applicable',
                                 old_method=a[0], new_method=b[0]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refusing existing output')
    manifest = json.loads(args.manifest.read_text())
    current, current_tables = load(manifest['current'])
    rows = []
    for arm in manifest['historical_arms']:
        prior, prior_tables = load(arm)
        rows.extend(compare(prior, current, prior_tables, current_tables,
                            arm['method_prefix'], arm['current_method_prefix'], arm['reference_arm']))
    args.out.mkdir(parents=True)
    target = args.out / 'changes.tsv'
    with target.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(schema='visium_hd_processing.codex_generation_comparison.v1',
                   result_id=manifest['result_id'], inputs=manifest,
                   rescored=False, all_policies_and_frozen_scales_included=True,
                   outputs={'changes.tsv': sha(target)})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
