#!/usr/bin/env python3
"""Extract only the raw 2 um matrix from a published binned-output archive."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from build_hd_cell_matrix import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--member-suffix', default='square_002um/raw_feature_bc_matrix.h5')
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    target = a.out_dir / Path(a.member_suffix).name
    found = None
    with tarfile.open(a.archive, mode='r|gz') as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(a.member_suffix):
                found = member.name
                with tar.extractfile(member) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, length=1 << 24)
                break
    if not found:
        raise ValueError(f'archive matrix not found: {a.member_suffix}')
    (a.out_dir / 'summary.json').write_text(json.dumps({'archive': str(a.archive),
        'archive_sha256': sha256(a.archive), 'member': found, 'output_sha256': sha256(target)}, indent=2) + '\n')
    print(target, flush=True)


if __name__ == '__main__': main()
