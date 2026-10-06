#!/usr/bin/env python3
"""Export corrected SPATCH bindings and placement from checksum-identified evidence."""
import argparse,json,math,shutil,subprocess
from pathlib import Path
from export_spatial_paper_numbers import export
from prepare_hd_he_number_export import audit_changes,replace_source
from summarize_sealed_spatial_primary import sha256
from run_hd_spatch_rescore_step import FIT,RUNS,destination

ALLOWED=('cpSPATCHResidualNucDelta','cpSPATCHStarOnlyCellAUC','cpSPATCHWholeCellDelta','cpSROnlyAUCHi','cpSROnlyAUCLo','spatchExcessDeficitPP')
NEW=('regSpatchShiftMinUm','regSpatchShiftMaxUm','regSpatchRegionsUsed','regSpatchHeldOutPass','regSpatchHeldOutTotal','regSpatchUnsupportedBins','regSpatchUnsupportedMaxPct')


def placement(fit,support):
    dropped=set(fit['fit']['final_fit']['rejected'])
    regions=[r for r in fit['regions'] if r['usable'] and r['region'] not in dropped]
    distances=[math.hypot(r['dy_um'],r['dx_um']) for r in regions]
    masses=support['common_axis_mass']
    return dict(zip(NEW,[min(distances),max(distances),len(regions),fit['fit']['held_out_pass'],fit['fit']['held_out_total'],support['unsupported_bins'],100*max(masses[k]['missing_mass']/masses[k]['total'] for k in ('hard_star_common_bin_totals','space_ranger_bin_totals'))]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    old=a.record/'paper_results/spatial_v1_9_5_he_corrections_20261004/ovarian_number_export/manifest.json'
    if not old.exists():old=RUNS/'20261004_ovarian_he_number_export_v1/result/manifest.json'
    manifest=json.loads(old.read_text())
    for s in manifest['sources'].values():
        path=Path(s['path']);path=path if path.is_absolute() else a.record/path
        if sha256(path)!=s['sha256']:raise ValueError(f'input drift: {path}')
        s['path']=str(path)
    package=a.record/'paper_results/spatial_v1_9_5_spatch_he_corrections_20261005'
    for name,path in [('flex_he_spatch_corrected',package/'flex_morphology_summary/flex_star_sr_he_consistency.tsv'),('three_he_spatch_corrected',package/'three_slide_summary/he_shared_excess.tsv')]:
        manifest['sources'][name]=dict(path=str(path),sha256=sha256(path),format='tsv')
    for name in ALLOWED:
        spec=manifest['macros'][name]
        spec['value']=replace_source(replace_source(spec['value'],'flex_he','flex_he_spatch_corrected'),'three_he','three_he_spatch_corrected')
        step='three_slide_summary' if name=='spatchExcessDeficitPP' else 'flex_morphology_summary'
        spec.update(result_id=f'hd_{step}_he_corrected_20261005',provenance_run_id=destination(step).parent.name,scope='Corrected SPATCH affine H&E placement; unsupported bins excluded from H&E only; CRC and counts unchanged')
    support_path=RUNS/'20261004_spatch_affine_support_audit_v1/result/summary.json'
    evidence=dict(values=placement(json.loads(FIT.read_text()),json.loads(support_path.read_text())),inputs={str(p):sha256(p) for p in (FIT,support_path)})
    evidence_path=a.out_dir/'placement_values.json';evidence_path.write_text(json.dumps(evidence,indent=2)+'\n')
    manifest['sources']['spatch_placement']=dict(path=str(evidence_path),sha256=sha256(evidence_path))
    for name in NEW:
        style={'kind':'decimal','places':1} if name in NEW[:2] else ({'kind':'decimal','places':2} if name==NEW[-1] else {'kind':'integer','group':False})
        manifest['macros'][name]=dict(status='regenerated',new_macro=True,result_id='hd_spatch_placement_he_corrected_20261005',provenance_run_id='20261004_spatch_affine_placement_fit_v1',scope='Measured displacement magnitude among final usable regions; common-axis hard-policy unsupported counts',value={'source':'spatch_placement','path':['values',name]},display=style)
    manifest['correction_scope']='Accepted ovarian and SPATCH H&E corrections; approved Amendment 2 support policy'
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    before=a.record/'paper/numbers.tex';shutil.copyfile(before,a.out_dir/'numbers_before.tex')
    coverage=export(manifest,a.record/'paper/main.tex',before,a.out_dir/'export')
    audit=audit_changes(before.read_text(),(a.out_dir/'export/numbers.tex').read_text(),ALLOWED,NEW)
    audit.update(old_sha256=sha256(before),new_sha256=sha256(a.out_dir/'export/numbers.tex'))
    (a.out_dir/'macro_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (a.out_dir/'summary.json').write_text(json.dumps(dict(publication_ready=coverage['publication_ready'],audit=audit,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs={str(old):sha256(old)},outputs={str(f.relative_to(a.out_dir)):sha256(f) for f in a.out_dir.rglob('*') if f.is_file()}),indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
