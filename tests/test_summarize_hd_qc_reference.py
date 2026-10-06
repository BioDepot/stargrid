from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_hd_qc_reference import grouped, range_flags, capture_key


def test_reference_groups_do_not_pool_sources_or_auxiliary_masks():
    base={'chemistry':'probe-based','preservation':'FFPE','reference_eligible':True,
          'q3_contrast_at_zero':.1,'q4_feature_contrast':.1,'q4_dy_um':0,'q4_dx_um':0,
          'feature_half_recovery_radius_um':2,'mean_count_in_tissue':10,
          'tissue_share_of_squares':.5,'nucleus_share_of_tissue_squares':.2}
    rows=pd.DataFrame([{**base,'slide_id':'H1-A/A1','count_source':'deposit'},
                       {**base,'slide_id':'H1-A/A1','count_source':'star'},
                       {**base,'slide_id':'H1-B/A1','count_source':'star','reference_eligible':False}])
    result=grouped(rows)
    assert set(result.count_source)=={'star','deposit'}
    assert result.slides_in_group.eq(1).all()
    assert 'mean' not in result and 'standard_deviation' not in result
    with pytest.raises(ValueError,match='multiple observations'):
        grouped(pd.concat([rows,rows.iloc[:1]],ignore_index=True))


def test_leave_one_out_flags_ties_missing_values_and_auxiliary_exclusion():
    rows=pd.DataFrame([{'slide_id':f'H1-{i}/A1','count_source':'deposit',
        'chemistry':'probe-based','preservation':'FFPE','reference_eligible':True,
        'q1_off_to_in_mean_ratio':i, 'q2_0_2um':1, 'q3_contrast_at_zero':i}
        for i in range(6)])
    rows.loc[0,'q3_contrast_at_zero']=float('nan')
    flags=range_flags(rows)
    q1=flags[flags.measure.eq('q1_off_to_in_mean_ratio')]
    assert list(q1.flag)==['outside','inside','inside','inside','inside','outside']
    assert q1.other_capture_areas.eq(5).all()
    assert q1.iloc[0].other_minimum==1 and q1.iloc[-1].other_maximum==4
    assert flags[flags.measure.eq('q2_0_2um')].flag.eq('inside').all()
    assert flags[flags.measure.eq('q3_contrast_at_zero')].flag.eq('no reference').all()
    rows.loc[5,'reference_eligible']=False
    assert range_flags(rows).flag.eq('no reference').all()


def test_capture_identity_merges_deposits_and_keeps_areas_distinct():
    assert capture_key(' h1-vm2jxxk / a1 ')==capture_key('H1-VM2JXXK/A1')
    assert capture_key('H1-VM2JXXK/D1')!=capture_key('H1-VM2JXXK/A1')
    with pytest.raises(ValueError,match='capture identity'):
        capture_key('GEO P2CRC')


def test_mean_and_sd_only_with_ten_available_values():
    base={'chemistry':'probe-based','preservation':'FFPE','count_source':'deposit',
        'reference_eligible':True,'q3_contrast_at_zero':.1,'q4_feature_contrast':.1,
        'q4_dy_um':0,'q4_dx_um':0,'feature_half_recovery_radius_um':2,
        'mean_count_in_tissue':10,'tissue_share_of_squares':.5,'nucleus_share_of_tissue_squares':.2}
    rows=pd.DataFrame([{**base,'slide_id':f'H1-{i}/A1'} for i in range(10)])
    assert 'mean' not in grouped(rows.iloc[:9])
    result=grouped(rows)
    assert result['mean'].notna().all() and result.standard_deviation.abs().lt(1e-14).all()
