import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from prepare_hd_paper_mask_placement import geometry_arrays, morphology_geometry_args
from score_hd_flex_registration_roi import RoiGeometry


def test_export_distinguishes_unsupported_from_supported_extracellular():
    g=RoiGeometry(np.array([0,2,7]),np.array([0,1,2],np.uint8),np.zeros(3,int),np.array([False]),{})
    a=geometry_arrays(g,4,2)
    assert a['supported'].ravel().tolist()==[True,False,True,False,False,False,False,True]
    assert a['cell'].sum()==2 and a['nucleus'].sum()==1
    np.testing.assert_array_equal(a['states'].ravel()[g.indices],g.states)
    with pytest.raises(ValueError,match='invalid geometry support'):
        geometry_arrays(RoiGeometry(np.array([0,0]),np.array([1,2]),np.zeros(2),np.array([False]),{}),4,2)


def test_campaign_geometry_arguments_preserve_slide_specific_sampling():
    argv=['--native-registration-json','registration.json','--capture-grid-json','grid.json',
          '--cellpose-segmentation','cellpose','--registration-moving-source-downsample','8',
          '--registration-moving-sampling','decimate','--width','3350','--height','3350']
    a=morphology_geometry_args(argv)
    assert a.registration_moving_source_downsample==8 and a.parent_size==8
    assert a.registration_moving_sampling=='decimate' and a.barcode_mappings is None
    assert a.roi_manifest is None and a.tissue_positions is None
