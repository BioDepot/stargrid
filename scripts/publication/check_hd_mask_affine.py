#!/usr/bin/env python3
"""Gate A on exported paper geometry with the scorer's shared affine helper."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import measure_hd_qc_reference as qc
from fit_hd_mask_affine import measure_regions
from score_hd_flex_registration_roi import RoiGeometry
from hd_mask_shift import correct_geometry,geometry_grids,export_geometry
import hd_mask_shift


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mask-affine-json',type=Path,required=True)
    p.add_argument('--reason',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    c=json.loads(a.config.read_text());w=c.get('width',3350)
    spec=c['extra_nuclei'][c['primary_nucleus_map']]
    paths=[a.config,a.mask_affine_json,Path(spec['path']),Path(c['counts']['path']),Path(c['tissue_positions'])]
    inputs={str(f):qc.legacy.sha256(f) for f in paths}
    with np.load(spec['path']) as f:
        indices=np.flatnonzero(f['supported']);states=f['states'].ravel()[indices]
    parents=(indices//w//8)*math.ceil(w/8)+indices%w//8
    base=RoiGeometry(indices,states,parents,np.bincount(parents,minlength=math.ceil(w/8)**2)==64,
                     {'source':'accepted campaign build_roi_geometry export','path':spec['path']})
    corrected=correct_geometry(base,width=w,height=w,affine=a.mask_affine_json)
    export_geometry(a.out_dir/'scoring_geometry.npz',corrected,width=w,height=w)
    nuc=geometry_grids(corrected,width=w,height=w)['nucleus']
    counts=qc.load_counts(c['counts'],w);tissue=qc.tissue_mask(c['tissue_positions'],w)
    surface=qc.shift_surface(nuc,counts,tissue)
    f=qc.feature(surface)
    regions,arrays,stability=qc.region_features(nuc,counts,tissue,f)
    residuals,residual_arrays=measure_regions(nuc,counts,tissue)
    gate=bool(stability['resolved'] and (f['dy'],f['dx'])==(0,0) and f['sign']=='dip')
    pd.DataFrame(residuals).to_csv(a.out_dir/'regional_residuals_5x5.tsv',sep='\t',index=False)
    np.savez_compressed(a.out_dir/'surfaces.npz',whole=surface,**arrays,
                        **{'five_'+k:v for k,v in residual_arrays.items()})
    summary=dict(schema='visium_hd_processing.mask_affine_gate_a.v1',gate_a=gate,
        feature=f,regions=regions,stability=stability,residuals_5x5=residuals,
        geometry_audit=corrected.audit,reason=a.reason,inputs=inputs,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        generator_sha256=qc.legacy.sha256(Path(__file__)),
        helper_sha256=qc.legacy.sha256(Path(hd_mask_shift.__file__)),
        contrast_helper_sha256=qc.legacy.sha256(Path(qc.__file__)),
        command=sys.argv,cpus=os.environ['OFF_BENCH_CPUS_USED'],
        outputs={f.name:qc.legacy.sha256(f) for f in a.out_dir.iterdir() if f.is_file()})
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(gate_a=gate,feature=f,stability=stability),indent=2))


if __name__=='__main__':main()
