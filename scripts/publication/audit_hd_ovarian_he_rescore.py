#!/usr/bin/env python3
"""Audit paper-mask placement and unchanged rows after ovarian H&E rescoring."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from score_hd_flex_registration_roi import sha256


def check_placement(summary, expected, require_all_regions=False):
    m=summary['maps']['paper_cellpose']; f=m['feature']
    if not m['resolved'] or (f['dy'],f['dx'])!=tuple(expected) or f['sign']!='dip':
        raise ValueError(f'placement differs from required resolved dip at {expected}: {f}')
    if require_all_regions and any(r['eligible'] and not r['agrees'] for r in m['regions'].values()):
        raise ValueError('eligible ovarian regions do not all agree within one square')
    return {'slide':summary['slide'],'count_source':summary['metadata']['count_source'],
        'feature':f,'eligible_regions':m['eligible_regions'],'agreeing_regions':m['agreeing_regions'],
        'resolved':m['resolved']}


def table_rows(path):
    with path.open(newline='') as h:
        reader=csv.DictReader(h,delimiter='\t'); rows=list(reader)
        return reader.fieldnames,rows


def compare_tables(old_dir,new_dir):
    old_files={p.name for p in old_dir.glob('*.tsv')}
    new_files={p.name for p in new_dir.glob('*.tsv')}
    if old_files!=new_files or 'he_shared_excess.tsv' not in old_files:
        raise ValueError('summary table inventory differs')
    result={}
    for name in sorted(old_files):
        old,new=old_dir/name,new_dir/name
        if name!='he_shared_excess.tsv':
            if old.read_bytes()!=new.read_bytes(): raise ValueError(f'non-H&E table changed: {name}')
            result[name]={'byte_identical':True,'unchanged_rows':len(table_rows(old)[1])}
            continue
        old_header,old_rows=table_rows(old);new_header,new_rows=table_rows(new)
        if old_header!=new_header or len(old_rows)!=len(new_rows): raise ValueError('H&E table shape changed')
        changed=[];unchanged=0
        for before,after in zip(old_rows,new_rows):
            keys=['dataset','method','component']
            if any(before[k]!=after[k] for k in keys): raise ValueError('H&E row identity/order changed')
            if before['dataset']!='OVARIAN_GEX':
                if before!=after: raise ValueError('non-ovarian H&E row changed')
                unchanged+=1
            elif before!=after:
                changed.append({k:before[k] for k in keys})
        result[name]={'non_ovarian_rows_identical':True,'unchanged_non_ovarian_rows':unchanged,
                      'changed_ovarian_rows':changed}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--placement',action='append',default=[],metavar='SLIDE=SUMMARY_JSON')
    p.add_argument('--shifted',action='store_true')
    p.add_argument('--old-summary-dir',type=Path)
    p.add_argument('--new-summary-dir',type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'): raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    inputs={};checks=[]
    for item in a.placement:
        slide,filename=item.split('=',1); path=Path(filename);s=json.loads(path.read_text())
        inputs[str(path)]=sha256(path)
        expected=(0,0) if a.shifted or slide=='crc' else (-3,-3)
        if slide not in ('crc','ovarian'): raise ValueError('SPATCH is reported without a required position')
        checks.append(check_placement(s,expected,require_all_regions=slide=='ovarian'))
    tables=None
    if a.old_summary_dir or a.new_summary_dir:
        if not (a.old_summary_dir and a.new_summary_dir): raise ValueError('both table directories required')
        tables=compare_tables(a.old_summary_dir,a.new_summary_dir)
        for directory in [a.old_summary_dir,a.new_summary_dir]:
            for path in directory.glob('*.tsv'): inputs[str(path)]=sha256(path)
    if not checks and tables is None: raise ValueError('no audit requested')
    summary={'status':'complete','placement_checks':checks,'table_checks':tables,'inputs':inputs,
        'script_sha256':sha256(Path(__file__)),'command':sys.argv,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'cpus':os.environ['OFF_BENCH_CPUS_USED']}
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
