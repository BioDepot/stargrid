#!/usr/bin/env python3
"""Report coarse bins lacking corrected H&E support without scoring them."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from score_hd_flex_registration_roi import sha256


def unsupported(support,coordinates,scale_um):
    h,w=support.shape;factor=scale_um//2;cw=math.ceil(w/factor)
    yy,xx=np.nonzero(support)
    children=np.bincount((yy//factor)*cw+xx//factor,minlength=math.ceil(h/factor)*cw)
    return children[coordinates[:,0]*cw+coordinates[:,1]]==0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',required=True,type=Path)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    prepared=json.loads((a.prepared/'summary.json').read_text())
    with np.load(a.geometry) as f:support=f['supported']
    with np.load(a.prepared/'coarse_fields.npz') as f:
        coords=f['space_ranger_coordinates'];missing=unsupported(support,coords,prepared['comparison_scale_um'])
        rows={key:dict(missing_mass=float(f[key][missing].sum()),total=float(f[key].sum()))
              for key in ['space_ranger_bin_totals']+[m+'_star_common_bin_totals' for m in prepared['methods']]}
        result=dict(common_axis_bins=len(coords),unsupported_bins=int(missing.sum()),
                    unsupported_coordinates=coords[missing].tolist(),common_axis_mass=rows)
        for m in prepared['methods']:
            coords=f[m+'_star_full_coordinates'];missing=unsupported(support,coords,prepared['comparison_scale_um'])
            field=f[m+'_star_full_bin_totals']
            result[m+'_full_axis']=dict(bins=len(coords),unsupported_bins=int(missing.sum()),
                                      missing_mass=float(field[missing].sum()),total=float(field.sum()))
    result.update(inputs={str(f):sha256(f) for f in [a.geometry,a.prepared/'summary.json',a.prepared/'coarse_fields.npz']},
                  generator_sha256=sha256(Path(__file__)),command=sys.argv,
                  source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (a.out_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
