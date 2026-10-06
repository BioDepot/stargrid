#!/usr/bin/env python3
"""Tabulate accepted slide QC outputs without repeating any slide measurement."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from measure_hd_nucleus_dip_and_edge_profiles import sha256


GROUP_KEYS = ['chemistry', 'preservation', 'count_source']


def capture_key(value):
    parts = str(value).upper().split('/')
    if len(parts) != 2:
        raise ValueError(f'capture identity must be serial_number/area: {value!r}')
    serial, area = (p.strip() for p in parts)
    if not re.fullmatch(r'H[0-9]+-[A-Z0-9]+', serial) or not re.fullmatch(r'[A-Z][0-9]*', area):
        raise ValueError(f'invalid capture identity: {value!r}')
    return f'{serial}/{area}'


def validate_rows(rows):
    result = rows.copy()
    result['slide_id'] = result.slide_id.map(capture_key)
    if result.duplicated(['slide_id', 'count_source']).any():
        raise ValueError('multiple observations per capture area and count source')
    for _, group in result.groupby('slide_id'):
        if any(group[k].nunique(dropna=False) != 1 for k in ['chemistry', 'preservation', 'tissue', 'species'] if k in group):
            raise ValueError('conflicting metadata for one capture area')
    return result


def row(summary, name):
    c = summary['metadata']
    m = summary['maps'].get(name, {})
    f = m.get('feature', {})
    r = {k:c.get(k) for k in ['slide_id','chemistry','preservation','tissue','species','count_source',
                              'pipeline_version','tissue_mask_origin','nucleus_mask_origin','reference_eligible']}
    r['slide_id'] = capture_key(r['slide_id'])
    r['serial_number'], r['area'] = r['slide_id'].split('/')
    r.update(slide=summary['slide'], nucleus_map=name, counts_complete=c['counts_complete'],
        tissue_share_of_squares=summary['tissue_share_of_squares'],
        mean_count_in_tissue=summary['mean_count_in_tissue'],
        count_total=summary['count_total'],
        nucleus_share_of_tissue_squares=m.get('nucleus_share_of_tissue_squares'),
        q1_count_share_off_tissue=summary.get('share_of_counts_off_tissue'),
        q1_off_to_in_mean_ratio=summary.get('off_to_in_mean_ratio'),
        q3_contrast_at_zero=m.get('contrast_at_zero'),
        q4_raw_dy_squares=f.get('dy'), q4_raw_dx_squares=f.get('dx'),
        q4_feature_contrast=f.get('contrast'), q4_feature_sign=f.get('sign'),
        q4_resolved=m.get('resolved'), q4_on_search_border=f.get('border'),
        q4_dy_um=f.get('dy_um') if m.get('resolved') else None,
        q4_dx_um=f.get('dx_um') if m.get('resolved') else None,
        feature_half_recovery_radius_um=f.get('width_um'),
        eligible_regions=m.get('eligible_regions'), agreeing_regions=m.get('agreeing_regions'),
        regional_agreement=m.get('regional_agreement'))
    for b in summary['off_tissue_profile']:
        key = 'q2_' + re.sub('[^a-z0-9]+','_',b['bin']).strip('_')
        r[key] = b['relative_to_inside']
    return r


def grouped(rows):
    rows = validate_rows(rows)
    eligible = rows[rows.reference_eligible.eq(True)]
    keys = GROUP_KEYS
    fields = [x for x in rows.columns if x.startswith(('q1_','q2_'))] + [
        'q3_contrast_at_zero','q4_feature_contrast','q4_dy_um','q4_dx_um',
        'feature_half_recovery_radius_um','mean_count_in_tissue','tissue_share_of_squares',
        'nucleus_share_of_tissue_squares']
    output = []
    for key, group in eligible.groupby(keys, dropna=False):
        for field in fields:
            values = pd.to_numeric(group[field],errors='coerce').dropna()
            item = dict(zip(keys,key), measure=field, slides_in_group=len(group), slides_measured=len(values),
                median=float(values.median()) if len(values) else None,
                minimum=float(values.min()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)
            if len(values) >= 10:
                item.update(mean=float(values.mean()), standard_deviation=float(values.std(ddof=1)))
            output.append(item)
    return pd.DataFrame(output)


def range_flags(rows):
    rows = validate_rows(rows)
    fields = ['q1_off_to_in_mean_ratio'] + [k for k in rows if k.startswith('q2_')] + ['q3_contrast_at_zero']
    output = []
    for _, group in rows.groupby(GROUP_KEYS, dropna=False):
        for _, target in group.iterrows():
            others = group[group.reference_eligible.eq(True) & group.slide_id.ne(target.slide_id)]
            for field in fields:
                value = pd.to_numeric(pd.Series([target.get(field)]), errors='coerce').iloc[0]
                values = pd.to_numeric(others[field], errors='coerce') if field in others else pd.Series(dtype=float)
                values = values[np.isfinite(values)]
                low, high = (float(values.min()), float(values.max())) if len(values) else (None, None)
                flag, reason = 'no reference', ''
                if not target.reference_eligible:
                    reason = 'auxiliary observation excluded from reference groups'
                elif not np.isfinite(value):
                    reason = 'target measure unavailable'
                elif len(values) < 5:
                    reason = 'fewer than five other capture areas with this measure'
                else:
                    flag = 'outside' if value < low or value > high else 'inside'
                output.append({**{k:target[k] for k in ['slide_id'] + GROUP_KEYS},
                    'measure':field, 'value':float(value) if np.isfinite(value) else None,
                    'other_capture_areas':len(values), 'other_minimum':low, 'other_maximum':high,
                    'flag':flag, 'reason':reason})
    return pd.DataFrame(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'): raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    rows,controls,inputs=[],[],{}
    for path in a.run_dir:
        source=path/'summary.json'; s=json.loads(source.read_text())
        if s['stop_findings']: raise ValueError(f'unreviewed stop finding in {source}')
        inputs[str(source)]=sha256(source)
        r=row(s,s['primary_nucleus_map']);r['run_directory']=str(path);rows.append(r)
        for name in s['maps']:
            if name!=s['primary_nucleus_map']:
                r=row(s,name);r['run_directory']=str(path);r['reference_eligible']=False
                r['nucleus_mask_origin']='independent native registration control';controls.append(r)
    frame=validate_rows(pd.DataFrame(rows))
    groups=grouped(frame)
    flags=range_flags(frame)
    frame.to_csv(a.out_dir/'qc_reference_values.tsv',sep='\t',index=False)
    groups.to_csv(a.out_dir/'qc_group_summary.tsv',sep='\t',index=False)
    flags.to_csv(a.out_dir/'qc_range_flags.tsv',sep='\t',index=False)
    pd.DataFrame(controls).to_csv(a.out_dir/'qc_controls.tsv',sep='\t',index=False)
    summary={'status':'exploratory pilot; flags prompt biological review',
        'slide_count':int(frame.slide_id.nunique()),'slide_source_rows':len(frame),
        'deposited_mask_reference_slides':int(frame[frame.reference_eligible].slide_id.nunique()),
        'grouping':['chemistry','preservation','count_source'],
        'flag_rule':'Amendment 2: strict comparison against other capture areas; five finite others required',
        'flag_counts':{str(k):int(v) for k,v in flags.flag.value_counts().items()},
        'range_probability':'For exchangeable continuous values: 2/(k+1) outside the k-other range; ties may reduce it',
        'small_groups':'median/min/max only; no mean/SD below ten independent slides; no z-scores',
        'cpus':os.environ['OFF_BENCH_CPUS_USED'],
        'inputs':inputs,'script_sha256':sha256(Path(__file__)),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'outputs':{p.name:sha256(p) for p in a.out_dir.iterdir() if p.is_file()}}
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(frame.to_string(index=False))


if __name__=='__main__': main()
