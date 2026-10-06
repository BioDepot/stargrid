#!/usr/bin/env python3
"""Package accepted SPATCH corrections into a fresh record directory."""
import argparse,json,shutil
from pathlib import Path
from package_hd_he_corrections import package_run
from run_hd_spatch_rescore_step import STEPS,destination,RUNS,FIT
from summarize_sealed_spatial_primary import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    gate=RUNS/'20261005_spatch_gate_b_v1/result/summary.json'
    if not json.loads(gate.read_text())['gate_b']:raise ValueError('Gate B not passed')
    root=a.record/'paper_results/spatial_v1_9_5_spatch_he_corrections_20261005';root.mkdir(exist_ok=False)
    for step in STEPS:
        run=destination(step).parent
        package_run(run,root/step,f'hd_{step}_he_corrected_20261005',adoption='PI-authorized SPATCH affine placement and Amendment 2 support policy; Gate B passed')
    evidence=root/'placement';evidence.mkdir()
    for src,name in [(FIT,'fit_summary.json'),(RUNS/'20261004_spatch_affine_support_audit_v1/result/summary.json','support_summary.json'),(RUNS/'20261004_spatch_affine_gate_a_v1/result/summary.json','gate_a.json'),(gate,'gate_b.json')]:shutil.copyfile(src,evidence/name)
    (root/'README.md').write_text('Accepted corrected SPATCH H&E outputs under Amendment 2. Counts and non-H&E fields unchanged; Gate B passed against accepted colorectal and corrected ovarian evidence. Large arrays remain external, identified in artifact manifests. Historical packages remain unchanged.\n')
    (root/'checksums.sha256').write_text(''.join(f'{sha256(f)}  {f.relative_to(root)}\n' for f in sorted(root.rglob('*')) if f.is_file()))
    print(root)

if __name__=='__main__':main()
