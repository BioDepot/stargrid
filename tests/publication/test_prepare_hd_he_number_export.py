from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from prepare_hd_he_number_export import audit_changes,replace_source


def test_only_allowed_macro_bodies_and_explicit_new_names_change():
    before=r'\newcommand{\ov}{1}\newcommand{\crc}{2}'
    after=r'\newcommand{\ov}{3}\newcommand{\crc}{2}\newcommand{\shift}{4}'
    result=audit_changes(before,after,['ov'],['shift'])
    assert result['identical_existing_macros']==1 and result['passed']
    with pytest.raises(ValueError,match='unapproved'):
        audit_changes(before,after.replace(r'{\crc}{2}',r'{\crc}{5}'),['ov'],['shift'])
    value={'op':'product','args':[100,{'source':'old','path':[0,'fraction']}]}
    assert replace_source(value,'old','new')['args'][1]['source']=='new'
    assert value['args'][1]['source']=='old'
