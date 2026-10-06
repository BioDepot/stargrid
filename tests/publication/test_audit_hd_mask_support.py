import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from audit_hd_mask_support import unsupported


def test_missing_support_preserves_axis_order_and_partial_bins():
    support=np.zeros((9,9),bool);support[0,0]=True;support[8,8]=True
    assert unsupported(support,np.array([[2,2],[1,1],[0,0]]),8).tolist()==[False,True,False]
