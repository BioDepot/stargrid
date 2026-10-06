#!/usr/bin/env python3
"""Measure 5x5 mask displacements and test a frozen affine correction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure_hd_qc_reference as qc


def measure_regions(nuc, counts, tissue, *, divisions=5, minimum=30000,
                    radius=17, search=12):
    ey = np.linspace(0, nuc.shape[0], divisions+1).astype(int)
    ex = np.linspace(0, nuc.shape[1], divisions+1).astype(int)
    rows, surfaces = [], {}
    for iy in range(divisions):
        for ix in range(divisions):
            y0, y1 = ey[iy:iy+2]; x0, x1 = ex[ix:ix+2]
            yy, xx = np.nonzero(nuc[y0:y1, x0:x1] & tissue[y0:y1, x0:x1])
            row = dict(region=f'r{iy}c{ix}', row=iy, col=ix,
                       nucleus_squares_in_tissue=len(yy), usable=False,
                       centroid_y=float(yy.mean()+y0) if len(yy) else None,
                       centroid_x=float(xx.mean()+x0) if len(xx) else None)
            if len(yy) >= minimum:
                a,b = max(0,y0-radius), min(nuc.shape[0],y1+radius)
                c,d = max(0,x0-radius), min(nuc.shape[1],x1+radius)
                local = np.zeros((b-a,d-c),bool)
                local[y0-a:y1-a,x0-c:x1-c] = nuc[y0:y1,x0:x1]
                surface = qc.shift_surface(local,counts[a:b,c:d],tissue[a:b,c:d],radius)
                f = qc.feature(surface,radius,search)
                row.update({k:v for k,v in f.items() if k!='ring_profile'})
                row['contrast_at_zero'] = qc.contrast(surface,0,0,radius)
                row['usable'] = bool(f['sign']=='dip' and f['contrast'] >= .03 and not f['border'])
                row['reason'] = 'usable' if row['usable'] else 'not an interior dip of contrast at least 0.03'
                surfaces[row['region']] = surface
            else:
                row['reason'] = f'fewer than {minimum} nucleus squares in tissue'
            rows.append(row)
    return rows,surfaces


def design(rows):
    return np.array([[1.,r['centroid_y'],r['centroid_x']] for r in rows])


def fit_once_with_rejection(rows):
    if len(rows)<3 or np.linalg.matrix_rank(design(rows))<3:
        raise ValueError('affine design has fewer than three independent positions')
    x=design(rows); y=np.array([[r['dy'],r['dx']] for r in rows],float)
    first=np.linalg.lstsq(x,y,rcond=None)[0]
    residual=y-x@first
    keep=np.max(np.abs(residual),axis=1)<=2
    if keep.sum()<3 or np.linalg.matrix_rank(x[keep])<3:
        raise ValueError('outlier rejection leaves fewer than three independent positions')
    fitted=np.linalg.lstsq(x[keep],y[keep],rcond=None)[0]
    return dict(coefficients=fitted.tolist(),first_coefficients=first.tolist(),
                first_residuals=residual.tolist(),
                retained=[r['region'] for r,k in zip(rows,keep) if k],
                rejected=[r['region'] for r,k in zip(rows,keep) if not k])


def fit_regions(rows, *, minimum_final=12):
    usable=[r for r in rows if r['usable']]
    train=[r for r in usable if (r['row']+r['col'])%2==0]
    held=[r for r in usable if (r['row']+r['col'])%2==1]
    out=dict(usable_regions=len(usable),training_regions=len(train),held_out_total=len(held),
             held_out_pass=0,accepted=False,
             model='[dy,dx] = [1,source_row,source_col] @ coefficients',
             centroid_definition='nucleus squares in tissue, zero-based global grid coordinates',
             outlier_rule='one rejection of training residuals >2 on either axis, then one refit; held-out rows never rejected',
             held_out=[],final_fit=None)
    try:
        fit=fit_once_with_rejection(train)
    except ValueError as exc:
        out['stop_reason']=str(exc);return out
    out['training_fit']=fit
    if held:
        pred=design(held)@np.array(fit['coefficients'])
        for r,p in zip(held,pred):
            residual=np.array([r['dy'],r['dx']])-p
            within=bool(np.max(np.abs(residual))<=1)
            out['held_out'].append(dict(region=r['region'],prediction=p.tolist(),
                residual=residual.tolist(),within_one_square=within))
        out['held_out_pass']=sum(r['within_one_square'] for r in out['held_out'])
    out['held_out_fraction']=out['held_out_pass']/len(held) if held else None
    if not held or out['held_out_fraction']<.75:
        out['stop_reason']='held-out prediction fraction below 0.75';return out
    try:
        final=fit_once_with_rejection(usable)
    except ValueError as exc:
        out['stop_reason']=str(exc);return out
    out['final_fit']=final
    if len(final['retained'])<minimum_final:
        out['stop_reason']=f'fewer than {minimum_final} regions remain';return out
    out['accepted']=True
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):
        raise SystemExit('use scripts/run_off_benchmark_cores.sh')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    c=json.loads(a.config.read_text());width=c.get('width',3350)
    name=c['primary_nucleus_map']; spec=c['extra_nuclei'][name]
    paths=[a.config,Path(c['counts']['path']),Path(c['tissue_positions']),Path(spec['path'])]
    paths += [Path(s) for s in c.get('supporting_records',[])]
    inputs={str(f):qc.legacy.sha256(f) for f in paths}
    counts=qc.load_counts(c['counts'],width); tissue=qc.tissue_mask(c['tissue_positions'],width)
    with np.load(spec['path']) as f: nuc=f[spec['key']].reshape(width,width)>0
    rows,surfaces=measure_regions(nuc,counts,tissue)
    whole_surface=qc.shift_surface(nuc,counts,tissue,17)
    whole=qc.feature(whole_surface,17,12)
    fit=fit_regions(rows)
    for row in rows:
        if row.get('dy') is not None:
            row['single_shift_residual_dy']=row['dy']-whole['dy']
            row['single_shift_residual_dx']=row['dx']-whole['dx']
    pd.DataFrame(rows).to_csv(a.out_dir/'regional_displacements.tsv',sep='\t',index=False)
    pd.DataFrame(fit['held_out']).to_csv(a.out_dir/'held_out_predictions.tsv',sep='\t',index=False)
    np.savez_compressed(a.out_dir/'surfaces.npz',whole=whole_surface,**surfaces)
    model=fit.get('final_fit') or fit.get('training_fit')
    locations={}
    if model:
        coeff=np.array(model['coefficients'])
        for label,(y,x) in {'centre':((width-1)/2,(width-1)/2),
                'top_left':(0,0),'top_right':(0,width-1),
                'bottom_left':(width-1,0),'bottom_right':(width-1,width-1)}.items():
            vector=np.array([1,y,x])@coeff
            locations[label]=dict(displacement_squares=vector.tolist(),
                displacement_um=(2*vector).tolist(),magnitude_um=float(2*np.linalg.norm(vector)))
    result=dict(schema='visium_hd_processing.mask_affine_fit.v1',slide=c['slide'],
        status='accepted_fit' if fit['accepted'] else 'stopped_before_affine_correction',
        counts_source='STAR hard totals only; no 2 um Space Ranger totals for SPATCH',
        command=sys.argv,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cpus=os.environ['OFF_BENCH_CPUS_USED'],inputs=inputs,
        generator_sha256=qc.legacy.sha256(Path(__file__)),
        contrast_helper_sha256=qc.legacy.sha256(Path(qc.__file__)),
        parameters=dict(divisions=5,surface_radius=17,search=12,minimum_nucleus_squares=30000,
                        minimum_contrast=.03,maximum_outlier_residual=2,held_out_tolerance=1,
                        minimum_held_out_fraction=.75,minimum_final_regions=12),
        whole=whole,regions=rows,fit=fit,
        reported_model='final' if fit.get('final_fit') else 'training only; not accepted',
        model_displacement_at_locations=locations,
        outputs={f.name:qc.legacy.sha256(f) for f in a.out_dir.iterdir() if f.is_file()})
    (a.out_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],fit=fit),indent=2),flush=True)
    return 0 # A scientific stop is a completed measurement, not a retryable crash.


if __name__=='__main__':
    raise SystemExit(main())
