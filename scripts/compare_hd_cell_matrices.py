#!/usr/bin/env python3
"""Descriptive comparison to Space Ranger's segmented output (not ground truth)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_limits
from build_hd_cell_matrix import sha256
from prepare_hd_same_run_offsets import square_indices


def reciprocal_matches(ours, theirs):
    keep = (ours > 0) & (theirs > 0)
    if not keep.any():
        return pd.DataFrame(columns=['ours', 'theirs', 'overlap_squares'])
    pairs, counts = np.unique(np.column_stack([ours[keep], theirs[keep]]), axis=0, return_counts=True)
    table = pd.DataFrame({'ours': pairs[:, 0], 'theirs': pairs[:, 1], 'overlap_squares': counts})
    # Greatest overlap, ties to the smaller identifier on the other side.
    a = table.sort_values(['ours', 'overlap_squares', 'theirs'], ascending=[True, False, True]).drop_duplicates('ours')
    b = table.sort_values(['theirs', 'overlap_squares', 'ours'], ascending=[True, False, True]).drop_duplicates('theirs')
    return a.merge(b, on=['ours', 'theirs', 'overlap_squares']).sort_values('ours').reset_index(drop=True)


def profile_correlations(a, b):
    if a.shape != b.shape or a.shape[1] < 2:
        raise ValueError('matched profile matrices must have equal shapes and at least two genes')
    a, b = a.astype(float).tocsr(), b.astype(float).tocsr()
    a.data, b.data = np.log1p(a.data), np.log1p(b.data)
    n = a.shape[1]
    sa, sb = np.asarray(a.sum(1)).ravel(), np.asarray(b.sum(1)).ravel()
    va = np.asarray(a.multiply(a).sum(1)).ravel() - sa * sa / n
    vb = np.asarray(b.multiply(b).sum(1)).ravel() - sb * sb / n
    cov = np.asarray(a.multiply(b).sum(1)).ravel() - sa * sb / n
    valid = (va > 1e-12) & (vb > 1e-12)
    result = np.full(len(sa), np.nan)
    result[valid] = np.clip(cov[valid] / np.sqrt(va[valid] * vb[valid]), -1, 1)
    return result


def load_h5(path):
    with h5py.File(path) as f:
        m = f['matrix']
        matrix = sparse.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:])).T.tocsr()
        return matrix, m['features/id'][:].astype(str), m['barcodes'][:].astype(str)


def describe(matrix, total):
    counts = np.asarray(matrix.sum(1)).ravel()
    return {'cells': matrix.shape[0], 'zero_count_cells': int((counts == 0).sum()),
            'molecules': float(counts.sum()), 'molecules_per_cell_mean': float(counts.mean()),
            'molecules_per_cell_median': float(np.median(counts)),
            'molecules_per_cell_p10_p90': np.quantile(counts, [.1, .9]).tolist(),
            'assigned_fraction': float(counts.sum() / total) if total else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--our-matrix', type=Path, required=True)
    p.add_argument('--sr-cell-h5', type=Path, required=True)
    p.add_argument('--sr-mappings', type=Path, required=True)
    p.add_argument('--same-run-offset-summary', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    if a.out_dir.exists():
        raise FileExistsError(a.out_dir)
    own_summary = json.loads((a.our_matrix / 'summary.json').read_text())
    offsets = json.loads(a.same_run_offset_summary.read_text())
    expected_mapping = offsets['inputs'].get(str(a.sr_mappings))
    if expected_mapping != sha256(a.sr_mappings):
        raise ValueError('nucleus map is not the same-run comparator used for count totals')
    with threadpool_limits(limits=4):
        our = io.mmread(a.our_matrix / 'cell_by_gene/matrix.mtx').T.tocsr()
    sr, sr_genes, sr_ids = load_h5(a.sr_cell_h5)
    our_genes = pd.read_csv(a.our_matrix / 'cell_by_gene/features.tsv', sep='\t', header=None)[0].astype(str).to_numpy()
    cells = pd.read_csv(a.our_matrix / 'cells.tsv', sep='\t')
    our_ids = cells.nucleus_id.to_numpy()
    if len(our_ids) != our.shape[0]:
        raise ValueError('our cell identifiers do not match matrix')
    mapping = pd.read_parquet(a.our_matrix / 'square_to_cell.parquet', columns=['square', 'nucleus_id'])
    if not np.array_equal(mapping.square, np.arange(len(mapping))):
        raise ValueError('our grid is not in row-major order')
    width = int(own_summary['grid_shape'][1])
    bm = pd.read_parquet(a.sr_mappings, columns=['square_002um', 'cell_id', 'in_nucleus'])
    nuc = bm.in_nucleus.fillna(False).to_numpy(bool)
    sr_label = np.zeros(len(mapping), np.int64)
    bc_index = pd.Index(sr_ids).get_indexer(bm.loc[nuc, 'cell_id'])
    coords = square_indices(bm.loc[nuc, 'square_002um'].to_numpy(), width)
    sr_label[coords] = bc_index + 1
    matches = reciprocal_matches(mapping.nucleus_id.to_numpy(), sr_label)
    common = sorted(set(our_genes) & set(sr_genes))
    if len(common) < 2 or len(np.unique(our_genes)) != len(our_genes) or len(np.unique(sr_genes)) != len(sr_genes):
        raise ValueError('ambiguous or insufficient shared feature IDs')
    oi = pd.Index(our_genes).get_indexer(common)
    si = pd.Index(sr_genes).get_indexer(common)
    rows = pd.Index(our_ids).get_indexer(matches.ours)
    if np.any(rows < 0):
        raise ValueError('nucleus missing from our cell matrix')
    matches['profile_log1p_pearson_r'] = profile_correlations(our[rows][:, oi], sr[matches.theirs.to_numpy(int) - 1][:, si])
    matches['our_cell_id'] = cells.cell_id.to_numpy()[rows]
    matches['sr_cell_id'] = sr_ids[matches.theirs.to_numpy(int) - 1]
    a.out_dir.mkdir(parents=True, exist_ok=False)
    matches.to_csv(a.out_dir / 'matched_cells.tsv', sep='\t', index=False)
    own_tot = pd.Series(np.asarray(our.sum(0)).ravel(), index=our_genes, name='our_molecules')
    sr_tot = pd.Series(np.asarray(sr.sum(0)).ravel(), index=sr_genes, name='sr_molecules')
    pd.concat([own_tot, sr_tot], axis=1).rename_axis('feature_id').to_csv(a.out_dir / 'gene_totals.tsv', sep='\t')
    pd.DataFrame({'cell_id': sr_ids, 'molecules': np.asarray(sr.sum(1)).ravel()}).to_csv(a.out_dir / 'sr_cells.tsv', sep='\t', index=False)
    r = matches.profile_log1p_pearson_r.dropna().to_numpy()
    summary = {'schema': 'visium_hd_processing.cell_matrix_comparison.v1',
        'comparator_is_truth': False, 'ours': describe(our, own_summary['input_molecules']),
        'space_ranger': describe(sr, offsets['count_totals']['space_ranger_same_run']),
        'shared_feature_ids': len(common), 'matched_nuclei': len(matches),
        'our_match_fraction': len(matches) / our.shape[0], 'sr_match_fraction': len(matches) / sr.shape[0],
        'sr_nucleus_squares_with_cell_missing_from_raw_matrix': int((bc_index < 0).sum()),
        'profile_constant_pairs': int(len(matches) - len(r)),
        'profile_r_median': float(np.median(r)) if len(r) else None,
        'profile_r_mean': float(np.mean(r)) if len(r) else None,
        'inputs': {str(q): sha256(q) for q in [a.our_matrix / 'summary.json', a.sr_cell_h5, a.sr_mappings, a.same_run_offset_summary]},
        'outputs': {q.name: sha256(q) for q in a.out_dir.glob('*.tsv')}, 'script_sha256': sha256(Path(__file__))}
    (a.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
