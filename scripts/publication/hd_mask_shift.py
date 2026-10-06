"""Whole-square translations of supported H&E states, without wrapping."""
import math
import operator
import json
from pathlib import Path

import numpy as np

from score_hd_flex_registration_roi import RoiGeometry


def shift_geometry(base, dy, dx, *, width, height, parent_size=8):
    dy,dx=operator.index(dy),operator.index(dx)
    if dy==0 and dx==0:
        return base  # Preserve all original arrays, ordering and audit values.
    rows=base.indices//width+dy
    cols=base.indices%width+dx
    kept=(rows>=0)&(rows<height)&(cols>=0)&(cols<width)
    indices=(rows[kept]*width+cols[kept]).astype(np.int64)
    states=base.states[kept]
    parent_width=math.ceil(width/parent_size)
    parent_height=math.ceil(height/parent_size)
    parents=(rows[kept]//parent_size)*parent_width+cols[kept]//parent_size
    complete=np.bincount(parents,minlength=parent_width*parent_height)==parent_size**2
    audit={**base.audit,'before_mask_shift':base.audit,
        'mask_shift_squares_dy_dx':[dy,dx],
        'shifted_out_supported_squares':int((~kept).sum()),
        'cellpose_supported_roi_2um_bins':len(indices),
        'cell_mask_bins':int((states>0).sum()),'nucleus_mask_bins':int((states==2).sum()),
        'extracellular_mask_bins':int((states==0).sum()),
        'complete_16um_parents':int(complete.sum())}
    return RoiGeometry(indices,states,parents,complete,audit)


def geometry_grids(base, *, width, height):
    states=np.zeros(width*height,np.uint8);support=np.zeros(width*height,bool)
    states[base.indices]=base.states;support[base.indices]=True
    return {k:v.reshape(height,width) for k,v in
        {'states':states,'supported':support,'nucleus':states==2,'cell':states>0}.items()}


def export_geometry(path: Path, base, *, width, height):
    if path.exists(): raise ValueError(f'refusing existing geometry export: {path}')
    np.savez_compressed(path,**geometry_grids(base,width=width,height=height))


def load_affine(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='visium_hd_processing.mask_affine_fit.v1' or not data['fit']['accepted']:
        raise ValueError('affine correction requires an accepted held-out fit')
    return data['fit']['final_fit']['coefficients']


def affine_geometry(base, coefficients, *, width, height, parent_size=8):
    """Pull nearest states/support through inverse of q=p+d(p), no wrapping.

    Displacement is [1,source_row,source_col] @ coefficients. np.rint uses
    nearest square, with exact halfway cases rounded to the even index.
    """
    coefficients=np.asarray(coefficients,dtype=float)
    if coefficients.shape!=(3,2) or not np.isfinite(coefficients).all():
        raise ValueError('affine coefficients must be finite 3x2')
    if not np.any(coefficients): return base
    forward=np.eye(2)+coefficients[1:].T
    if np.linalg.det(forward)<=1e-8:
        raise ValueError('affine correction is singular or reverses orientation')
    inverse=np.linalg.inv(forward)
    source=geometry_grids(base,width=width,height=height)
    # Rows in chunks avoid several full float64 grids in memory.
    indices=[]; states=[]; outside=0
    for start in range(0,height,128):
        yy,xx=np.mgrid[start:min(start+128,height),0:width]
        sy=inverse[0,0]*(yy-coefficients[0,0])+inverse[0,1]*(xx-coefficients[0,1])
        sx=inverse[1,0]*(yy-coefficients[0,0])+inverse[1,1]*(xx-coefficients[0,1])
        sy=np.rint(sy).astype(np.int64);sx=np.rint(sx).astype(np.int64)
        valid=(sy>=0)&(sy<height)&(sx>=0)&(sx<width)
        outside+=int((~valid).sum())
        sy=np.clip(sy,0,height-1);sx=np.clip(sx,0,width-1)
        valid &= source['supported'][sy,sx]
        indices.append((yy[valid]*width+xx[valid]).astype(np.int64))
        states.append(source['states'][sy[valid],sx[valid]])
    indices=np.concatenate(indices);states=np.concatenate(states)
    parent_width=math.ceil(width/parent_size);parent_height=math.ceil(height/parent_size)
    parents=(indices//width//parent_size)*parent_width+indices%width//parent_size
    complete=np.bincount(parents,minlength=parent_width*parent_height)==parent_size**2
    audit={**base.audit,'before_mask_affine':base.audit,
        'mask_affine_coefficients':coefficients.tolist(),
        'affine_model':'[dy,dx] = [1,source_row,source_col] @ coefficients; corrected q=p+d(p)',
        'resampling':'inverse transform, nearest square (ties to even), states and support together',
        'destination_squares_without_source_in_grid':outside,
        'cellpose_supported_roi_2um_bins':len(indices),
        'cell_mask_bins':int((states>0).sum()),'nucleus_mask_bins':int((states==2).sum()),
        'extracellular_mask_bins':int((states==0).sum()),'complete_16um_parents':int(complete.sum())}
    return RoiGeometry(indices,states,parents,complete,audit)


def correct_geometry(base, *, width, height, shift=(0,0), affine=None):
    if affine is not None:
        if any(shift): raise ValueError('whole-square and affine corrections are mutually exclusive')
        return affine_geometry(base,load_affine(affine),width=width,height=height)
    return shift_geometry(base,*shift,width=width,height=height)
