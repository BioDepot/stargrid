#!/usr/bin/env python3
"""Audit all rescored SPATCH tables, preserving non-H&E fields exactly."""
import csv,json,argparse
from pathlib import Path
from run_hd_spatch_rescore_step import STEPS,destination,OLD,RUNS
from summarize_sealed_spatial_primary import sha256


def he_column(step, name):
    if step == 'spatch_morphology':
        return name not in {'field','field_role','policy','equal_mass_normalization_applied','scope'}
    return any(x in name for x in ('cell_', 'nucleus_', '_cell', '_nucleus', '_in_he_domain'))


def compare(step, name, before, after):
    old=list(csv.DictReader(before.splitlines(),delimiter='\t'))
    new=list(csv.DictReader(after.splitlines(),delimiter='\t'))
    failures=[]; changed=[]; same=0
    if len(old)!=len(new):return {'failures':['row count changed']}
    for i,(a,b) in enumerate(zip(old,new)):
        if set(a)!=set(b):failures.append(f'row {i}: columns differ');continue
        spatch=(a.get('dataset')=='SPATCH' or step in ('spatch_morphology','spatch_gene_bin'))
        for key in a:
            permitted=(spatch and he_column(step,key) and name not in ('mass_reconciliation.tsv','count_gain.tsv','cross_slide_gene_consistency.tsv'))
            if a[key]!=b[key]:
                detail=dict(row=i,column=key,old=a[key],new=b[key])
                if permitted:changed.append(detail)
                else:failures.append(detail)
            else:same+=1
    return dict(failures=failures,changed_he_cells=changed,identical_cells=same,rows=len(old))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False);tables={}
    for step in STEPS:
        prior=(RUNS/'20261004_three_slide_summary_in_register_v1/result' if step=='three_slide_summary'
               else Path(json.loads((OLD/step/'COMPLETE.json').read_text())['output_directory']))
        current=destination(step)
        old_names={p.name for p in prior.glob('*.tsv')};new_names={p.name for p in current.glob('*.tsv')}
        if old_names!=new_names:raise ValueError(f'table inventory changed: {step}')
        for name in sorted(old_names):
            result=compare(step,name,(prior/name).read_text(),(current/name).read_text())
            result.update(old_path=str(prior/name),new_path=str(current/name),old_sha256=sha256(prior/name),new_sha256=sha256(current/name))
            tables[f'{step}/{name}']=result
    passed=not any(t['failures'] for t in tables.values())
    result=dict(gate_b=passed,tables=tables)
    (a.out_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(gate_b=passed,tables=len(tables),failures={k:v['failures'][:5] for k,v in tables.items() if v['failures']}),indent=2))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
