#!/usr/bin/env python3
"""Bind accepted ovarian corrections on the a5502f0 manuscript-number base."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from export_spatial_paper_numbers import evaluate,display,export
from audit_live_number_macros import definitions,uncomment
from summarize_sealed_spatial_primary import sha256

OVARIAN=('heSharedInCell','heExcessInCell','heExcessDeficitPP','heSharedAUC',
         'heExcessAUC','heSharedEnrich','heExcessEnrich')
NEW=('regOvShiftSquares','regOvShiftUm')


def replace_source(value,before,after):
    if isinstance(value,list):return [replace_source(x,before,after) for x in value]
    if isinstance(value,dict):
        return {k:(after if k=='source' and v==before else replace_source(v,before,after))
                for k,v in value.items()}
    return value


def audit_changes(before,after,allowed,new_names):
    old,_=definitions(uncomment(before));new,_=definitions(uncomment(after))
    added=set(new)-set(old);removed=set(old)-set(new)
    changed={n for n in set(old)&set(new) if old[n]!=new[n]}
    if removed or added!=set(new_names) or changed-set(allowed):
        raise ValueError(f'unapproved macro changes: changed={changed}, added={added}, removed={removed}')
    return dict(changed={n:{'old':old[n]['body'],'new':new[n]['body']} for n in sorted(changed)},
                new={n:new[n]['body'] for n in sorted(added)},
                identical_existing_macros=len(old)-len(changed),removed=[],passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    old=a.record/'paper_results/spatial_v1_9_5_20260915/paper_number_inputs_20260927/manifest.json'
    manifest=json.loads(old.read_text())
    # Verify that the owning worktree's a5502f0 bindings are the accepted base.
    for prior,digest in manifest['binding_resolution']['preparation_inputs'].items():
        local=root/'publication/campaigns'/Path(prior).name
        if sha256(local)!=digest:raise ValueError('wrong manuscript binding base')
    for spec in manifest['sources'].values():
        path=a.record/spec['path']
        if sha256(path)!=spec['sha256']:raise ValueError(f'archived source drift: {path}')
        spec['path']=str(path)
    corrected=a.record/'paper_results/spatial_v1_9_5_he_corrections_20261004'
    table=corrected/'three_slide_ovarian_corrected/he_shared_excess.tsv'
    morphology=corrected/'ovarian_morphology/summary.json'
    shift=json.loads(morphology.read_text())['mask_shift']
    if shift['squares_dy_dx']!=[-3,-3]:raise ValueError('unexpected accepted ovarian correction')
    manifest['sources']['three_he_corrected']=dict(path=str(table),sha256=sha256(table),format='tsv')
    manifest['sources']['ovarian_correction']=dict(path=str(morphology),sha256=sha256(morphology))
    for name in OVARIAN:
        spec=manifest['macros'][name]
        original=copy.deepcopy(spec)
        spec.update(value=replace_source(spec['value'],'three_he','three_he_corrected'),
            result_id='hd_three_slide_ovarian_corrected_v1_9_5_20261004',
            provenance_run_id='20261004_three_slide_summary_in_register_v1',
            scope='Ovarian hard-policy H&E at 8 um; frozen sr_compat_v1 masks and support translated (-3,-3) 2 um squares; matched SR4.1.0 intron-inclusive counts',
            old_evidence={k:original[k] for k in ['result_id','provenance_run_id','value']})
    for name,value,style in [
        ('regOvShiftSquares',{'op':'product','args':[-1,{'source':'ovarian_correction','path':['mask_shift','squares_dy_dx',0]}]}, {'kind':'integer','group':False}),
        ('regOvShiftUm',{'source':'ovarian_correction','path':['mask_shift','euclidean_micrometres']},{'kind':'decimal','places':2})]:
        manifest['macros'][name]=dict(status='regenerated',new_macro=True,
            result_id='hd_ovarian_morphology_in_register_v1_9_5_20261004',
            provenance_run_id='20261004_ovarian_he_morphology_in_register_v1',
            scope='Magnitude of accepted ovarian correction; signed displacement (-3,-3) squares',value=value,display=style)
    manifest['correction_scope']='ovarian only; SPATCH support-policy decision pending after successful Gate A'
    manifest_path=a.out_dir/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    numbers=a.record/'paper/numbers.tex'
    shutil.copyfile(numbers,a.out_dir/'numbers_before.tex')
    coverage=export(manifest,a.record/'paper/main.tex',numbers,a.out_dir/'export')
    audit=audit_changes(numbers.read_text(),(a.out_dir/'export/numbers.tex').read_text(),OVARIAN,NEW)
    audit.update(allowed_macros=list(OVARIAN),new_macros=list(NEW),
                 old_numbers_sha256=sha256(numbers),new_numbers_sha256=sha256(a.out_dir/'export/numbers.tex'))
    (a.out_dir/'macro_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    summary=dict(schema='visium_hd_processing.he_number_export.v1',audit=audit,
        binding_base='a5502f0',record_base='035cb75',publication_ready=coverage['publication_ready'],
        inputs={str(old):sha256(old),str(numbers):sha256(numbers),str(table):sha256(table),str(morphology):sha256(morphology)},
        generator_sha256=sha256(Path(__file__)),exporter_sha256=sha256(Path(__file__).with_name('export_spatial_paper_numbers.py')),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=sys.argv,
        outputs={str(f.relative_to(a.out_dir)):sha256(f) for f in a.out_dir.rglob('*') if f.is_file()})
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
