from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from audit_hd_ovarian_he_rescore import check_placement,compare_tables


def test_placement_requires_expected_sign_position_and_regional_agreement():
    m={'resolved':True,'feature':{'dy':-3,'dx':-3,'sign':'dip'},
       'regions':{'a':{'eligible':True,'agrees':True}},'eligible_regions':1,'agreeing_regions':1}
    s={'slide':'synthetic','metadata':{'count_source':'star'},'maps':{'paper_cellpose':m}}
    assert check_placement(s,(-3,-3),True)['resolved']
    with pytest.raises(ValueError,match='placement differs'): check_placement(s,(0,0))
    m['regions']['a']['agrees']=False
    with pytest.raises(ValueError,match='regions'): check_placement(s,(-3,-3),True)


def test_only_ovarian_he_rows_may_change(tmp_path):
    a=tmp_path/'old';b=tmp_path/'new';a.mkdir();b.mkdir()
    header='dataset\tmethod\tcomponent\tvalue\n'
    common='CRC\thard\tshared\t0.5\n'
    (a/'he_shared_excess.tsv').write_text(header+common+'OVARIAN_GEX\thard\tshared\t0.7\n')
    (b/'he_shared_excess.tsv').write_text(header+common+'OVARIAN_GEX\thard\tshared\t0.8\n')
    for d in [a,b]: (d/'count_gain.tsv').write_text('dataset\tcount\nOVARIAN_GEX\t100\n')
    audit=compare_tables(a,b)
    assert audit['count_gain.tsv']['byte_identical']
    assert len(audit['he_shared_excess.tsv']['changed_ovarian_rows'])==1
    (b/'count_gain.tsv').write_text('dataset\tcount\nOVARIAN_GEX\t101\n')
    with pytest.raises(ValueError,match='non-H&E'): compare_tables(a,b)
    (b/'count_gain.tsv').write_bytes((a/'count_gain.tsv').read_bytes())
    (b/'he_shared_excess.tsv').write_text(header+common.replace('0.5','0.6')+'OVARIAN_GEX\thard\tshared\t0.8\n')
    with pytest.raises(ValueError,match='non-ovarian'): compare_tables(a,b)
