#!/usr/bin/env python3
"""Package compact accepted H&E correction outputs without rerunning analyses."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from summarize_sealed_spatial_primary import sha256

RUNS=Path('<runs>/runs')


def package_run(run,destination,result_id,adoption="PI accepted ovarian rescoring on 2026-10-04; original successful runs reused"):
    if destination.exists():raise ValueError(f'refusing existing result: {destination}')
    if (run/'exit_code.txt').read_text().strip()!='0':raise ValueError('run did not complete')
    source=run/'result'
    summary=json.loads((source/'summary.json').read_text())
    for name,spec in summary.get('outputs',{}).items():
        digest=spec['sha256'] if isinstance(spec,dict) else spec
        if sha256(source/name)!=digest:raise ValueError(f'output drift: {name}')
    acceptance=dict(status='complete',exit_code=0,result_id=result_id,
        original_run=str(run),source_revision=(run/'source_revision.txt').read_text().strip(),
        command=(run/'command.txt').read_text().strip(),
        outputs={str(f):sha256(f) for f in source.iterdir() if f.is_file()},
        operational_records={str(run/name):sha256(run/name) for name in
            ['exit_code.txt','command.txt','source_revision.txt','console.log','environment.json','cpus.txt'] if (run/name).is_file()},
        adoption=adoption)
    destination.mkdir(parents=True)
    compact,external={},{}
    for f in sorted(source.iterdir()):
        if not f.is_file():continue
        spec=dict(source=str(f),sha256=sha256(f),bytes=f.stat().st_size)
        if f.suffix in ('.json','.tsv') and f.stat().st_size<25_000_000:
            shutil.copyfile(f,destination/f.name);compact[f.name]=spec
        else:external[str(f)]=spec
    (destination/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    manifest=dict(result_id=result_id,acceptance_source=str(run),
        acceptance_sha256=sha256(destination/'acceptance.json'),compact_files=compact,external_files=external)
    (destination/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (destination/'checksums.sha256').write_text(''.join(f'{sha256(f)}  {f.name}\n'
        for f in sorted(destination.iterdir()) if f.is_file()))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'):raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    root=a.record/'paper_results/spatial_v1_9_5_he_corrections_20261004'
    if root.exists():raise ValueError('refusing existing record package')
    specs={
        'ovarian_morphology':('20261004_ovarian_he_morphology_in_register_v1','hd_ovarian_morphology_in_register_v1_9_5_20261004'),
        'three_slide_ovarian_corrected':('20261004_three_slide_summary_in_register_v1','hd_three_slide_ovarian_corrected_v1_9_5_20261004'),
        'ovarian_gate_a':('20261004_ovarian_he_gate_a_v1','hd_ovarian_mask_placement_v1_9_5_20261004'),
        'ovarian_gate_b':('20261004_ovarian_he_gate_b_v1','hd_ovarian_mask_placement_v1_9_5_20261004'),
        'ovarian_shifted_star':('20261004_ovarian_shifted_mask_placement_star_v1','hd_ovarian_mask_placement_v1_9_5_20261004'),
        'ovarian_shifted_space_ranger':('20261004_ovarian_shifted_mask_placement_space_ranger_v1','hd_ovarian_mask_placement_v1_9_5_20261004'),
    }
    outputs={name:package_run(RUNS/run,root/name,result) for name,(run,result) in specs.items()}
    readme=('Accepted ovarian correction, packaged without repeating an analysis.\n\n'
        'SPATCH placement passed its held-out check and Gate A. Scoring stopped because corrected '
        'support leaves coarse bins without H&E squares; no corrected SPATCH scores are adopted here.\n'
        'Each directory retains acceptance, compact-file checksums and an artifact manifest. '
        'Large arrays and operational logs remain in the named external run directories.\n')
    (root/'README.md').write_text(readme)
    summary=dict(schema='visium_hd_processing.he_correction_package.v1',
        record=str(a.record),package=str(root),artifacts=outputs,
        generator_sha256=sha256(Path(__file__)),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=sys.argv,created_utc=datetime.now(timezone.utc).isoformat())
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Packaged {len(outputs)} accepted ovarian outputs at {root}')


if __name__=='__main__':main()
