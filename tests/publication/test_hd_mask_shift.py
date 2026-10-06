import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'scripts/publication'))
from hd_mask_shift import shift_geometry,geometry_grids,affine_geometry,correct_geometry
from score_hd_flex_registration_roi import RoiGeometry,score_field
from score_hd_flex_star_sr_morphology import build_axis_geometry,expand_uniformly
from measure_hd_qc_reference import feature,shift_surface


def base(states,width):
    indices=np.arange(states.size)
    parents=(indices//width//8)*int(np.ceil(width/8))+(indices%width//8)
    return RoiGeometry(indices,states.ravel(),parents,np.bincount(parents)==64,{'original':True})


def test_zero_shift_keeps_scored_fields_exactly_identical():
    rng=np.random.default_rng(83)
    original=base(rng.integers(0,3,(32,32),dtype=np.uint8),32)
    shifted=shift_geometry(original,0,0,width=32,height=32)
    assert shifted is original
    coords=np.array([(r,c) for r in range(8) for c in range(8)])
    coarse=rng.integers(0,100,len(coords)).astype(float)
    before,axis,n=build_axis_geometry(original,coords,width=32,height=32,scale_um=8)
    after,shifted_axis,shifted_n=build_axis_geometry(shifted,coords,width=32,height=32,scale_um=8)
    old=score_field('fixture','shared','hard',expand_uniformly(coarse,axis,n),before)
    new=score_field('fixture','shared','hard',expand_uniformly(coarse,shifted_axis,shifted_n),after)
    assert old==new


@pytest.mark.parametrize('shift,indices,states', [((-1,1),[1,7],[0,1]),((1,-1),[6,10],[1,2])])
def test_edges_do_not_wrap_and_extracellular_support_moves(shift,indices,states):
    original=RoiGeometry(np.array([0,3,4,7,10]),np.array([2,1,0,2,1],np.uint8),
                         np.zeros(5,int),np.array([False]),{})
    moved=shift_geometry(original,*shift,width=4,height=3)
    assert moved.indices.tolist()==indices and moved.states.tolist()==states
    grids=geometry_grids(moved,width=4,height=3)
    assert np.flatnonzero(grids['supported']).tolist()==indices
    assert np.all(grids['states'][~grids['supported']]==0)
    assert moved.audit['shifted_out_supported_squares']==3


def test_shift_sign_undoes_planted_displacement_then_feature_is_zero():
    rng=np.random.default_rng(14);truth=rng.random((150,150))>.8
    truth[:20]=False;truth[-20:]=False;truth[:,:20]=False;truth[:,-20:]=False
    original=base(truth.astype(np.uint8)*2,150)
    counts=20-8*truth.astype(float);tissue=np.ones_like(truth)
    displaced=shift_geometry(original,3,3,width=150,height=150)
    mask=geometry_grids(displaced,width=150,height=150)['nucleus']
    f=feature(shift_surface(mask,counts,tissue))
    assert (f['dy'],f['dx'])==(-3,-3)
    corrected=shift_geometry(displaced,f['dy'],f['dx'],width=150,height=150)
    f=feature(shift_surface(geometry_grids(corrected,width=150,height=150)['nucleus'],counts,tissue))
    assert (f['dy'],f['dx'],f['sign'])==(0,0,'dip')


def test_zero_affine_is_identity_and_translation_moves_support_without_wrap():
    original=RoiGeometry(np.array([0,3,4,7,10]),np.array([2,1,0,2,1],np.uint8),
                         np.zeros(5,int),np.array([False]),{})
    assert affine_geometry(original,np.zeros((3,2)),width=4,height=3) is original
    affine=affine_geometry(original,[[-1,1],[0,0],[0,0]],width=4,height=3)
    shifted=shift_geometry(original,-1,1,width=4,height=3)
    for key,value in geometry_grids(shifted,width=4,height=3).items():
        np.testing.assert_array_equal(geometry_grids(affine,width=4,height=3)[key],value)


def test_planted_affine_is_measured_predicted_and_undone():
    from fit_hd_mask_affine import measure_regions,fit_regions
    rng=np.random.default_rng(926)
    n=rng.random((400,400))>.8
    n[:20]=False;n[-20:]=False;n[:,:20]=False;n[:,-20:]=False
    original=base(n.astype(np.uint8)*2,400)
    coeff=np.array([[-2.5,1.5],[.0125,0],[0,-.0125]])
    truth=affine_geometry(original,coeff,width=400,height=400)
    counts=20-12*geometry_grids(truth,width=400,height=400)['nucleus'].astype(float)
    tissue=np.ones_like(n)
    rows,_=measure_regions(n,counts,tissue,minimum=100)
    fit=fit_regions(rows)
    assert fit['accepted'] and fit['held_out_pass']==12
    recovered=np.array(fit['final_fit']['coefficients'])
    points=np.array([[1,r,c] for r,c in [(40,40),(40,360),(360,40),(360,360),(200,200)]])
    # Integer regional features quantize a continuous affine field; recover
    # its displacement to less than half a square, not exact coefficients.
    np.testing.assert_allclose(points@recovered,points@coeff,atol=.25)
    corrected=affine_geometry(original,recovered,width=400,height=400)
    f=feature(shift_surface(geometry_grids(corrected,width=400,height=400)['nucleus'],counts,tissue))
    assert (f['dy'],f['dx'],f['sign'])==(0,0,'dip')
    # Independent scalar inverse oracle includes unsupported outside edges.
    grid=geometry_grids(corrected,width=400,height=400)
    matrix=np.eye(2)+recovered[1:].T
    for r,c in [(0,0),(0,399),(399,0),(399,399),(57,142),(219,72)]:
        y,x=np.rint(np.linalg.solve(matrix,np.array([r,c])-recovered[0])).astype(int)
        valid=0<=y<400 and 0<=x<400
        assert grid['supported'][r,c]==valid
        assert grid['states'][r,c]==(n[y,x]*2 if valid else 0)


def test_affine_rejects_invalid_transform_and_combined_options():
    original=base(np.zeros((8,8),np.uint8),8)
    with pytest.raises(ValueError,match='singular'):
        affine_geometry(original,[[0,0],[-1,0],[0,0]],width=8,height=8)
    with pytest.raises(ValueError,match='mutually exclusive'):
        correct_geometry(original,width=8,height=8,shift=(1,0),affine='unused.json')
