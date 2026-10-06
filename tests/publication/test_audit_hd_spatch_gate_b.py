import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from audit_hd_spatch_gate_b import compare


def test_non_he_and_crc_changes_fail():
    old='dataset\tstar_mass\tstar_cell_mass_fraction\nSPATCH\t10\t.5\nCRC\t20\t.4\n'
    assert not compare('flex_gene_bin_summary','two_slide_method_summary.tsv',old,old.replace('.5','.6'))['failures']
    assert compare('flex_gene_bin_summary','two_slide_method_summary.tsv',old,old.replace('10','11'))['failures']
    assert compare('flex_gene_bin_summary','two_slide_method_summary.tsv',old,old.replace('.4','.3'))['failures']
