import importlib.util
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "score_hd_flex_star_sr_morphology.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "score_hd_flex_star_sr_morphology", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_spatial_bin_components_reconcile_both_methods():
    star = np.array([5.0, 1.0, 0.0])
    sr = np.array([3.0, 2.0, 4.0])
    shared, star_only, sr_only = MODULE.spatial_bin_components(star, sr)
    assert shared.tolist() == [3.0, 1.0, 0.0]
    assert star_only.tolist() == [2.0, 0.0, 0.0]
    assert sr_only.tolist() == [0.0, 1.0, 4.0]
    np.testing.assert_allclose(shared + star_only, star)
    np.testing.assert_allclose(shared + sr_only, sr)


def test_uniform_expansion_preserves_mass():
    coarse = np.array([8.0, 6.0])
    child_axis = np.array([0, 0, 0, 0, 1, 1])
    child_counts = np.array([4, 2])
    expanded = MODULE.expand_uniformly(coarse, child_axis, child_counts)
    assert expanded.tolist() == [2.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    assert expanded.sum() == coarse.sum()


def test_axis_geometry_selects_children_and_preserves_axis_order():
    base = MODULE.RoiGeometry(
        indices=np.arange(16, dtype=np.int64),
        states=np.zeros(16, dtype=np.uint8),
        parent_ids=np.zeros(16, dtype=np.int64),
        complete_parent_mask=np.array([False]),
        audit={},
    )
    geometry, child_axis, child_counts = MODULE.build_axis_geometry(
        base,
        np.array([[0, 1], [0, 0]], dtype=np.int32),
        width=4,
        height=4,
        scale_um=4,
    )
    assert len(geometry.indices) == 8
    assert child_counts.tolist() == [4, 4]
    assert child_axis.tolist() == [1, 1, 0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize('affine',[False,True])
def test_cli_scores_and_exports_the_same_shifted_geometry(tmp_path, monkeypatch,affine):
    from hd_mask_shift import shift_geometry, geometry_grids
    rng=np.random.default_rng(34)
    indices=np.arange(32*32,dtype=np.int64)
    parents=(indices//32//8)*4+indices%32//8
    base=MODULE.RoiGeometry(indices,rng.integers(0,3,len(indices),dtype=np.uint8),
                            parents,np.bincount(parents)==64,{})
    monkeypatch.setattr(MODULE,'build_roi_geometry',lambda _:base)
    prepared=tmp_path/'prepared';prepared.mkdir()
    segmentation=tmp_path/'cellpose';segmentation.mkdir()
    (segmentation/'summary.json').write_text('{}')
    for name in ['grid.json','registration.json']: (tmp_path/name).write_text('{}')
    coordinates=np.array([(r,c) for r in range(8) for c in range(8)])
    sr=rng.integers(1,100,len(coordinates)).astype(float)
    star=sr+rng.integers(-1,7,len(coordinates))
    arrays=prepared/'coarse_fields.npz'
    np.savez(arrays,space_ranger_coordinates=coordinates,space_ranger_bin_totals=sr,
             hard_star_common_bin_totals=star,hard_star_full_coordinates=coordinates,
             hard_star_full_bin_totals=star)
    (prepared/'summary.json').write_text(json.dumps({
        'schema':'visium_hd_processing.flex_star_sr_prepared_fields.v1','dataset':'SYNTH',
        'comparison_scale_um':8,'methods':{'hard':{'inputs':[]}},'space_ranger':{},
        'outputs':{'coarse_fields.npz':{'sha256':MODULE.sha256(arrays)}}}))
    out=tmp_path/'scored'
    options=['--mask-shift-squares','-1','1']
    if affine:
        fit=tmp_path/'affine.json'
        fit.write_text(json.dumps({'schema':'visium_hd_processing.mask_affine_fit.v1',
            'fit':{'accepted':True,'final_fit':{'coefficients':[[-1,1],[0,0],[0,0]]}}}))
        options=['--mask-affine-json',str(fit)]
    monkeypatch.setattr(sys,'argv',['score','--prepared-fields',str(prepared),
        '--cellpose-segmentation',str(segmentation),'--capture-grid-json',str(tmp_path/'grid.json'),
        '--native-registration-json',str(tmp_path/'registration.json'),'--width','32','--height','32',
        '--slide','SYNTH','--area','A1','--dataset','SYNTH',*options,
        '--mask-shift-reason','synthetic translation','--export-scoring-geometry','--out-dir',str(out)])
    assert MODULE.main()==0
    moved=shift_geometry(base,-1,1,width=32,height=32)
    expected=geometry_grids(moved,width=32,height=32)
    with np.load(out/'scoring_geometry.npz') as exported:
        for key in expected: np.testing.assert_array_equal(exported[key],expected[key])
    geometry,axis,counts=MODULE.build_axis_geometry(moved,coordinates,width=32,height=32,scale_um=8)
    expected_field=MODULE.score_field('space_ranger_common_axis','space_ranger_whole','space_ranger',
                                    MODULE.expand_uniformly(sr,axis,counts),geometry)
    with (out/'morphology_fields.tsv').open() as h: first=next(csv.DictReader(h,delimiter='\t'))
    assert float(first['in_cell_mass_fraction'])==expected_field['in_cell_mass_fraction']
    summary=json.loads((out/'summary.json').read_text())
    if affine:
        assert summary['mask_affine']['coefficients']==[[-1.,1.],[0.,0.],[0.,0.]]
    else:
        assert summary['mask_shift']['squares_dy_dx']==[-1,1]
    assert summary['mask_shift']['reason']=='synthetic translation'
    assert summary['outputs']['scoring_geometry.npz']['sha256']==MODULE.sha256(out/'scoring_geometry.npz')
