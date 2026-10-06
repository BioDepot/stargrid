#!/usr/bin/env python3
"""Execute the new matrix writer once against an accepted known-truth assignment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_hd_cell_matrix import build, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--matrix-dir', type=Path, required=True)
    p.add_argument('--accepted-assignment', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    geo = np.load(args.truth / 'tile0_geometry.npz')
    source = args.out_dir / 'true_nuclei.npz'
    np.savez_compressed(source, label=np.where(geo['nucleus_label_full'] > 0, geo['nucleus_label_full'], 0), nx=geo['nx'], ny=geo['ny'])
    build(argparse.Namespace(matrix_dir=args.matrix_dir, nuclei=source, label_key='label', grid_width=None,
                            growth_um=8, shift_dy=0, shift_dx=0, out_dir=args.out_dir / 'matrix'))
    got = pd.read_parquet(args.out_dir / 'matrix/square_to_cell.parquet').cell_id.to_numpy()
    ref = np.load(args.accepted_assignment)
    want = np.zeros(len(got), np.int64)
    want[ref['square']] = ref['cell']
    np.testing.assert_array_equal(got, want)
    (args.out_dir / 'validation.json').write_text(json.dumps({'exact_assignment_match': True,
        'squares_compared': len(got), 'assignment_sha256': sha256(args.accepted_assignment),
        'truth_geometry_sha256': sha256(args.truth / 'tile0_geometry.npz'),
        'matrix_summary_sha256': sha256(args.out_dir / 'matrix/summary.json')}, indent=2) + '\n')
    print('Exact M1_expand8 assignment match:', len(got), 'squares', flush=True)


if __name__ == '__main__': main()
