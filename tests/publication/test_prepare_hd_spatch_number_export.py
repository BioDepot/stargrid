import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from prepare_hd_spatch_number_export import placement


def test_placement_uses_retained_regions_and_pipeline_maximum():
    fit={'regions':[{'usable':True,'region':'a','dy_um':6,'dx_um':8}, {'usable':True,'region':'b','dy_um':0,'dx_um':100}], 'fit':{'final_fit':{'rejected':['b']},'held_out_pass':3,'held_out_total':4}}
    support={'unsupported_bins':2,'common_axis_mass':{'hard_star_common_bin_totals':{'missing_mass':3,'total':100},'space_ranger_bin_totals':{'missing_mass':4,'total':100}}}
    got=placement(fit,support)
    assert got['regSpatchShiftMinUm']==got['regSpatchShiftMaxUm']==10
    assert got['regSpatchRegionsUsed']==1
    assert got['regSpatchUnsupportedMaxPct']==4
