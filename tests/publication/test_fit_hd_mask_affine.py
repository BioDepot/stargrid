from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from fit_hd_mask_affine import measure_regions,fit_regions


def affine_rows():
    return [dict(region=f'r{r}c{c}',row=r,col=c,centroid_y=100*r+50,
                 centroid_x=100*c+50,dy=-2+r,dx=1-c,usable=True)
            for r in range(5) for c in range(5)]


def test_alternate_region_prediction_recovers_affine_and_rejects_training_outlier():
    rows=affine_rows();rows[0]['dy']+=9
    result=fit_regions(rows)
    assert result['accepted'] and result['held_out_pass']==12
    assert result['training_fit']['rejected']==['r0c0']
    assert result['final_fit']['rejected']==['r0c0']
    np.testing.assert_allclose(result['final_fit']['coefficients'],
                               [[-2.5,1.5],[.01,0],[0,-.01]],atol=1e-12)


def test_held_out_rows_are_never_excluded_and_failure_prevents_final_fit():
    rows=affine_rows()
    for row in rows:
        if (row['row']+row['col'])%2: row['dy']+=3
    result=fit_regions(rows)
    assert not result['accepted'] and result['held_out_pass']==0
    assert result['held_out_total']==12 and result['final_fit'] is None


def test_fewer_than_twelve_final_regions_stops():
    rows=affine_rows()[:11]
    result=fit_regions(rows)
    assert not result['accepted'] and result['stop_reason']=='fewer than 12 regions remain'


def test_synthetic_regional_features_and_fit_recover_planted_displacement():
    rng=np.random.default_rng(123)
    n=rng.random((400,400))>.85
    n[:18]=False;n[-18:]=False;n[:,:18]=False;n[:,-18:]=False
    counts=np.full(n.shape,20.)
    for r in range(5):
        for c in range(5):
            yy,xx=np.nonzero(n[r*80:(r+1)*80,c*80:(c+1)*80])
            counts[yy+r*80-2+r,xx+c*80+1-c]=8
    rows,_=measure_regions(n,counts,np.ones_like(n),minimum=100)
    assert all(row['usable'] for row in rows)
    assert [(row['dy'],row['dx']) for row in rows]==[(-2+r,1-c) for r in range(5) for c in range(5)]
    fit=fit_regions(rows)
    assert fit['accepted'] and fit['held_out_pass']==12
