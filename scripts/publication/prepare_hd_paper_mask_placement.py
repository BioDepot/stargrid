#!/usr/bin/env python3
"""Export the exact accepted campaign morphology geometry for placement checks.

No image registration or segmentation is performed. The campaign is read-only.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT/'src'))
from run_spatial_analysis_manifest import render
from score_hd_flex_registration_roi import build_roi_geometry, sha256


def runtime():
    versions={}
    for k in ['numpy','scipy','pandas','h5py','zarr']:
        try: versions[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError: versions[k]=None
    return {'python':sys.version,'packages':versions,'cpu_affinity':sorted(os.sched_getaffinity(0))}


def geometry_arrays(geometry, width, height):
    indices = geometry.indices
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= width*height):
        raise ValueError('invalid geometry support')
    if not np.isin(geometry.states, [0, 1, 2]).all():
        raise ValueError('invalid geometry states')
    support = np.zeros(width*height, bool)
    states = np.zeros(width*height, np.uint8)
    support[indices] = True
    states[indices] = geometry.states
    if not np.array_equal(np.flatnonzero(support), indices):
        raise ValueError('geometry indices are not in canonical order')
    return {k:v.reshape(height, width) for k,v in
            {'states':states, 'supported':support, 'nucleus':states==2, 'cell':states>0}.items()}


def morphology_geometry_args(argv):
    def value(name): return argv[argv.index(name)+1]
    return argparse.Namespace(native_registration_json=Path(value('--native-registration-json')),
        capture_grid_json=Path(value('--capture-grid-json')),
        cellpose_segmentation=Path(value('--cellpose-segmentation')),
        registration_moving_source_downsample=int(value('--registration-moving-source-downsample')),
        registration_moving_sampling=value('--registration-moving-sampling'),
        width=int(value('--width')), height=int(value('--height')), parent_size=8,
        roi_manifest=None, tissue_positions=None, space_ranger_alignment_json=None, barcode_mappings=None)


def accepted_step(analysis_root, name, inputs):
    path = analysis_root/name/'COMPLETE.json'
    record = json.loads(path.read_text())
    if record['status'] != 'complete' or record['exit_code'] != 0:
        raise ValueError(f'unaccepted step: {name}')
    inputs[str(path)] = sha256(path)
    for filename, digest in record['outputs'].items():
        if sha256(Path(filename)) != digest: raise ValueError(f'accepted output drift: {filename}')
        inputs[filename] = digest
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slide', choices=['crc','ovarian','spatch'], required=True)
    p.add_argument('--campaign', type=Path, default=ROOT/'publication/campaigns/spatial_v1_9_5_20260915.json')
    p.add_argument('--accepted-analysis-root', type=Path, default=Path('<storage>/analyses'))
    p.add_argument('--out-dir', type=Path, required=True)
    phases=p.add_mutually_exclusive_group()
    phases.add_argument('--export-only',action='store_true',help='Export geometry in its accepted container; counts are staged separately')
    phases.add_argument('--geometry-dir',type=Path,help='Reuse an accepted geometry export without projecting it again')
    a=p.parse_args()
    if not os.environ.get('OFF_BENCH_CPUS_USED'): raise SystemExit('use CPU helper')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    c=json.loads(a.campaign.read_text()); inputs={str(a.campaign):sha256(a.campaign)}
    if inputs[str(a.campaign)] != '460239c56d601e83e20c8c746432dab8d8ebd5571c850b5267331587ff94deaf':
        raise ValueError('campaign differs from accepted manifest')
    name=a.slide+'_morphology'; step=next(s for s in c['steps'] if s['id']==name)
    source=ROOT/step['generator']
    if sha256(source)!=step['generator_sha256']: raise ValueError('baseline scorer changed before placement export')
    inputs[str(source)]=sha256(source)
    accepted=accepted_step(a.accepted_analysis_root,name,inputs)
    bindings={'FIXED':{k:v['path'] for k,v in c['fixed_inputs'].items()},
              'PARENT':{k:str(Path(v['path']).parent) for k,v in c['fixed_inputs'].items()},
              'PRIMARY':c['primaries'],'STEP':{}}
    for dep in step['depends_on']:
        bindings['STEP'][dep]=accepted_step(a.accepted_analysis_root,dep,inputs)['output_directory']
    argv=render(step['arguments'],bindings,accepted['output_directory'])
    old_args=accepted['argv'][accepted['argv'].index('--prepared-fields'):]
    if argv!=old_args: raise ValueError('rendered campaign arguments differ from accepted scorer arguments')
    for key in [a.slide+'_cellpose_summary',a.slide+'_capture_grid',a.slide+'_native_registration']:
        spec=c['fixed_inputs'][key]; digest=sha256(Path(spec['path']))
        if digest!=spec['sha256']: raise ValueError(f'fixed input drift: {key}')
        inputs[spec['path']]=digest
    gargs=morphology_geometry_args(argv)
    old=json.loads((Path(accepted['output_directory'])/'summary.json').read_text())
    if a.geometry_dir:
        gs=a.geometry_dir/'summary.json'; previous=json.loads(gs.read_text())
        gp=a.geometry_dir/'grid_states.npz'
        if previous['campaign_step']!=name or sha256(gp)!=previous['outputs']['grid_states.npz']:
            raise ValueError('geometry export mismatch')
        inputs[str(gs)]=sha256(gs); inputs[str(gp)]=sha256(gp)
        geometry_audit=previous['geometry_audit']
        with np.load(gp) as data: arrays={k:data[k] for k in data.files}
        shutil.copyfile(gp,a.out_dir/'grid_states.npz')
    else:
        print(f'Exporting {name} through build_roi_geometry with accepted arguments',flush=True)
        geometry=build_roi_geometry(gargs)
        geometry_audit=geometry.audit
        arrays=geometry_arrays(geometry,gargs.width,gargs.height)
        np.savez_compressed(a.out_dir/'grid_states.npz',**arrays)
    if geometry_audit != old['geometry_audit']['base']:
        raise ValueError('export geometry audit differs from the accepted scorer')
    if a.export_only:
        for path in [Path(__file__),HERE/'score_hd_flex_registration_roi.py']:
            inputs[str(path)]=sha256(path)
        exported={'schema':'visium_hd_processing.paper_geometry_export.v1','status':'complete',
            'campaign_step':name,'accepted_arguments':argv,'geometry_audit':geometry_audit,
            'geometry_audit_matches_accepted':True,'inputs':inputs,'runtime':runtime(),
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'command':sys.argv,'cpus':os.environ['OFF_BENCH_CPUS_USED'],
            'outputs':{'grid_states.npz':sha256(a.out_dir/'grid_states.npz')}}
        (a.out_dir/'summary.json').write_text(json.dumps(exported,indent=2)+'\n')
        print('Exact accepted geometry exported; count staging will reuse this export',flush=True)
        return
    from prepare_spatch_qc_reference import mex_totals
    from measure_hd_qc_reference import published_nuclei
    from measure_hd_nucleus_dip_and_edge_profiles import map_displacement

    primary_name={'crc':'crc_flex_2020a','spatch':'spatch_flex_2020a','ovarian':'ovarian_gex_2024a_compat'}[a.slide]
    primary=Path(c['primaries'][primary_name]); seal_path=primary/'PRIMARY_SEAL.json'
    seal=json.loads(seal_path.read_text()); manifest=primary/'mex.relative.sha256'
    if seal['status']!='sealed' or sha256(manifest)!=seal['mex_manifest_sha256']:
        raise ValueError('unsealed or changed primary manifest')
    inputs[str(seal_path)]=sha256(seal_path); inputs[str(manifest)]=sha256(manifest)
    spatial=primary/('starSpatialGex.out' if a.slide=='ovarian' else 'star/SpatialFlex.out')
    mex=spatial/'hard/square_002um'
    expected={str((spatial/rel).resolve()):digest for digest,rel in
              (line.split('  ',1) for line in manifest.read_text().splitlines())}
    for filename in ['matrix.mtx','barcodes.tsv','features.tsv']:
        path=mex/filename; digest=sha256(path)
        if digest!=expected[str(path.resolve())]: raise ValueError(f'primary component drift: {path}')
        inputs[str(path)]=digest
    b=Path('<runs>/')
    if a.slide=='spatch':
        print('Staging full SPATCH hard totals from sealed MEX',flush=True)
        total,matrix=mex_totals(mex)
        np.savez_compressed(a.out_dir/'square_totals.npz',total=total)
        counts=a.out_dir/'square_totals.npz'
        tissue=b/'runs/20260727_spatch_visium_hd_ffpe_100k_sr_version_audit_v2/sr410_2020a_visium_v2_100k/outs/binned_outputs/square_002um/spatial/tissue_positions.parquet'
    else:
        stage=b/('runs/20260927_ovarian_cell_mapping_inputs_v1' if a.slide=='ovarian' else 'runs/20260930_crc_calibration_inputs_v1')
        stage_summary=stage/'summary.json'; staged=json.loads(stage_summary.read_text())
        staged=staged['inputs'] if a.slide=='ovarian' else staged
        if staged['star_matrix_sha256'] != inputs[str(mex/'matrix.mtx')]:
            raise ValueError('retained total-count input differs from campaign primary')
        # The accepted survey independently pins the retained NPZ; reuse its totals.
        survey=b/f'runs/20261004_qc_survey_{a.slide}_star_v1/summary.json'
        survey_record=json.loads(survey.read_text()); counts=stage/'square_totals.npz'
        if sha256(counts)!=survey_record['inputs'][str(counts)]: raise ValueError('retained totals changed')
        inputs[str(stage_summary)]=sha256(stage_summary); inputs[str(survey)]=sha256(survey)
        tissue=Path(c['fixed_inputs'][a.slide+'_vendor2']['path']).parent/'spatial/tissue_positions.parquet'
        matrix={'reused_totals':str(counts),'campaign_primary_matrix_sha256':inputs[str(mex/'matrix.mtx')]}
    inputs[str(tissue)]=sha256(tissue); inputs[str(counts)]=sha256(counts)
    displacement=None
    if a.slide!='spatch':
        mapping=Path(c['fixed_inputs']['crc_vendor_barcode_mappings']['path']) if a.slide=='crc' else b/'10x/visium_hd_3prime_human_ovarian_ff_min_depth/source/downloads/Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth_barcode_mappings.parquet'
        inputs[str(mapping)]=sha256(mapping)
        if a.slide=='crc' and inputs[str(mapping)]!=c['fixed_inputs']['crc_vendor_barcode_mappings']['sha256']:
            raise ValueError('published mapping changed')
        if a.slide=='ovarian' and inputs[str(mapping)]!=staged['tenx_barcode_mappings_sha256']:
            raise ValueError('published ovarian mapping changed')
        displacement=map_displacement(arrays['nucleus'],published_nuclei(mapping,gargs.width))
    metadata={'slide':seal['slide_area'],'slide_id':seal['slide_area'],
        'chemistry':'3prime' if a.slide=='ovarian' else 'probe-based',
        'preservation':'fresh frozen' if a.slide=='ovarian' else 'FFPE',
        'tissue':'colorectal cancer' if a.slide=='crc' else 'ovarian cancer','species':'human',
        'counts_complete':True,'reference_eligible':False,'tissue_positions':str(tissue),
        'tissue_mask_origin':'local Space Ranger' if a.slide!='crc' else 'deposit',
        'nucleus_mask_origin':'accepted paper Cellpose geometry',
        'primary_nucleus_map':'paper_cellpose',
        'extra_nuclei':{'paper_cellpose':{'path':str(a.out_dir/'grid_states.npz'),'key':'nucleus'}},
        'supporting_records':[str(a.out_dir/'summary.json')]}
    configs={'star':{**metadata,'count_source':'star','pipeline_version':'STAR Suite 1.9.5',
                     'counts':{'kind':'npz','path':str(counts)}}}
    if a.slide!='spatch':
        spec=c['fixed_inputs'][a.slide+'_vendor2']; digest=sha256(Path(spec['path']))
        if digest!=spec['sha256']: raise ValueError('Space Ranger raw 2um matrix differs from campaign')
        inputs[spec['path']]=digest
        configs['space_ranger']={**metadata,'count_source':'space_ranger','pipeline_version':'Space Ranger 4.1.0'+(' introns' if a.slide=='ovarian' else ''),
            'counts':{'kind':'h5','path':spec['path']}}
    for label,config in configs.items():
        (a.out_dir/f'{label}_qc_config.json').write_text(json.dumps(config,indent=2)+'\n')
    for path in [Path(__file__),HERE/'score_hd_flex_registration_roi.py',HERE.parent/'prepare_spatch_qc_reference.py',HERE.parent/'measure_hd_qc_reference.py',HERE.parent/'measure_hd_nucleus_dip_and_edge_profiles.py']:
        inputs[str(path)]=sha256(path)
    summary={'schema':'visium_hd_processing.paper_mask_placement_inputs.v1','status':'complete',
        'slide':seal['slide_area'],'campaign_step':name,'accepted_arguments':argv,
        'geometry_audit_matches_accepted':True,'geometry_audit':geometry_audit,
        'counts':matrix,'displacement_of_published_map':displacement,
        'sign_convention':'nucleus at (r,c) scores counts at (r+dy,c+dx); mask correction moves states to (r+dy,c+dx)',
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runtime':runtime(),
        'command':sys.argv,'cpus':os.environ['OFF_BENCH_CPUS_USED'],'inputs':inputs,
        'outputs':{p.name:sha256(p) for p in a.out_dir.iterdir() if p.is_file()}}
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'slide':summary['slide'],'geometry_audit_matches_accepted':True,'displacement':displacement}),flush=True)


if __name__=='__main__': main()
