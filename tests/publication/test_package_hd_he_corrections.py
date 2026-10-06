from pathlib import Path
import sys
import json
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from package_hd_he_corrections import package_run,sha256


def test_compact_copy_records_external_arrays_and_refuses_drift(tmp_path):
    run=tmp_path/'run';source=run/'result';source.mkdir(parents=True)
    for name in ['exit_code.txt','command.txt','source_revision.txt','console.log','environment.json','cpus.txt']:
        (run/name).write_text('0' if name=='exit_code.txt' else 'synthetic')
    (source/'table.tsv').write_text('x\n1\n');(source/'large.npz').write_bytes(b'array')
    (source/'summary.json').write_text(json.dumps({'outputs':{'table.tsv':sha256(source/'table.tsv')}}))
    result=package_run(run,tmp_path/'record','fixture')
    assert (tmp_path/'record/table.tsv').read_bytes()==(source/'table.tsv').read_bytes()
    assert not (tmp_path/'record/large.npz').exists()
    assert str(source/'large.npz') in result['external_files']
    with pytest.raises(ValueError,match='existing'):package_run(run,tmp_path/'record','fixture')
    (source/'table.tsv').write_text('x\n2\n')
    with pytest.raises(ValueError,match='drift'):package_run(run,tmp_path/'changed','fixture')
