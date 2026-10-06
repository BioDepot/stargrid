#!/usr/bin/env python3
"""Run one authorized SPATCH correction step, retaining exact campaign arguments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from run_spatial_analysis_manifest import render,sha256

ROOT=Path(__file__).resolve().parents[2]
OLD=Path('<storage>/analyses')
RUNS=Path('<runs>/runs')
STEPS=('spatch_morphology','spatch_he_weights','spatch_gene_bin','flex_morphology_summary',
       'flex_gene_bin_summary','three_slide_summary','flex_morphology_figure')
FIT=RUNS/'20261004_spatch_affine_placement_fit_v1/result/summary.json'
GATE=RUNS/'20261004_spatch_affine_gate_a_v1/result/summary.json'
OVARIAN=RUNS/'20261004_ovarian_he_morphology_in_register_v1/result'
REASON='PI-authorized SPATCH affine correction; 9/9 held-out regions within one square; 18 final regions; Gate A resolved at zero in 9/9 regions'


def destination(step):
    return RUNS/f'20261005_{step}_affine_supported_v1/result'


def correction_arguments(step,arguments):
    if step in ('spatch_morphology','spatch_he_weights'):
        arguments=[*arguments,'--mask-affine-json',str(FIT),'--mask-shift-reason',REASON,'--allow-unsupported-coarse-barcodes']
    if step=='spatch_morphology': arguments += ['--export-scoring-geometry']
    return arguments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',choices=STEPS,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):raise SystemExit('use CPU helper')
    if a.out_dir.exists():raise ValueError('refusing existing output')
    campaign=ROOT/'publication/campaigns/spatial_v1_9_5_20260915.json'
    if sha256(campaign)!='460239c56d601e83e20c8c746432dab8d8ebd5571c850b5267331587ff94deaf':
        raise ValueError('sealed campaign drift')
    manifest=json.loads(campaign.read_text())
    if not json.loads(GATE.read_text())['gate_a']:raise ValueError('Gate A not satisfied')
    step=next(s for s in manifest['steps'] if s['id']==a.step)
    bindings={'PRIMARY':manifest['primaries'],'FIXED':{},'PARENT':{},'STEP':{}}
    inputs={str(campaign):sha256(campaign),str(FIT):sha256(FIT),str(GATE):sha256(GATE)}
    def check(path,digest):
        if sha256(Path(path))!=digest:raise ValueError(f'input drift: {path}')
        inputs[str(path)]=digest
    for name,spec in manifest['fixed_inputs'].items():
        bindings['FIXED'][name]=spec['path'];bindings['PARENT'][name]=str(Path(spec['path']).parent)
        if any('${FIXED:'+name+'}' in s or '${PARENT:'+name+'}' in s for s in step['arguments']):
            check(spec['path'],spec['sha256'])
    for dep in step['depends_on']:
        if dep in STEPS:
            directory=destination(dep)
            complete=json.loads((directory.parent/'execution.json').read_text())
            if complete['exit_code']!=0:raise ValueError(f'dependency incomplete: {dep}')
            for path,digest in complete['outputs'].items():check(path,digest)
        elif dep=='ovarian_morphology':
            directory=OVARIAN
            summary=json.loads((directory/'summary.json').read_text())
            for name,spec in summary['outputs'].items():check(directory/name,spec['sha256'])
            inputs[str(directory/'summary.json')]=sha256(directory/'summary.json')
        else:
            complete=json.loads((OLD/dep/'COMPLETE.json').read_text())
            directory=Path(complete['output_directory'])
            for path,digest in complete['outputs'].items():check(path,digest)
            inputs[str(OLD/dep/'COMPLETE.json')]=sha256(OLD/dep/'COMPLETE.json')
        bindings['STEP'][dep]=str(directory)
    # Pin the matrix components actually read by gene-bin scoring to the accepted seal.
    if a.step=='spatch_gene_bin':
        primary=Path(manifest['primaries']['spatch_flex_2020a'])
        seal=json.loads((primary/'PRIMARY_SEAL.json').read_text())
        mex=primary/'mex.relative.sha256'
        check(mex,seal['mex_manifest_sha256'])
        for line in mex.read_text().splitlines():
            digest,relative=line.split('  ',1)
            if any('/'+policy+'/square_008um/' in '/'+relative for policy in ['hard','soft_expected']):
                check(primary/'star/SpatialFlex.out'/relative,digest)
    source=ROOT/step['generator']
    if a.step not in ('spatch_morphology','spatch_he_weights','spatch_gene_bin'):
        check(source,step['generator_sha256'])
    args=correction_arguments(a.step,render(step['arguments'],bindings,a.out_dir))
    cmd=[sys.executable,str(source),*args]
    image=step.get('container_image')
    if image:
        cmd=['docker','run','--rm','--network','none','--cpuset-cpus',os.environ['OFF_BENCH_CPUS_USED'],
             '--user',f'{os.getuid()}:{os.getgid()}',
             '--volume','<local>:<local>:ro','--volume','/storage:/storage:ro',
             '--volume',f'{a.out_dir.parent}:{a.out_dir.parent}','--workdir',str(ROOT),
             '--env',f'OFF_BENCH_CPUS_USED={os.environ["OFF_BENCH_CPUS_USED"]}',
             '--env','OMP_NUM_THREADS=8','--env','OPENBLAS_NUM_THREADS=8','--env','MKL_NUM_THREADS=8',
             image,'python',str(source),*args]
    record=dict(step=a.step,original_step=step,argv=cmd,inputs=inputs,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        generator_sha256=sha256(source),runner_sha256=sha256(Path(__file__)),
        mask_helper_sha256=sha256(ROOT/'scripts/publication/hd_mask_shift.py'),
        container_image=image,started_utc=datetime.now(timezone.utc).isoformat())
    path=a.out_dir.parent/'execution.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    rc=subprocess.run(cmd,cwd=ROOT).returncode
    record.update(exit_code=rc,finished_utc=datetime.now(timezone.utc).isoformat())
    if rc==0:
        record['outputs']={str(f):sha256(f) for f in sorted(a.out_dir.rglob('*')) if f.is_file()}
    path.write_text(json.dumps(record,indent=2)+'\n')
    return rc


if __name__=='__main__':raise SystemExit(main())
