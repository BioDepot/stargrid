from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/publication'))
from run_hd_spatch_rescore_step import correction_arguments


def test_only_mask_consumers_receive_identical_correction():
    original=['--out-dir','synthetic']
    morphology=correction_arguments('spatch_morphology',original)
    weights=correction_arguments('spatch_he_weights',original)
    assert morphology[:-1]==weights and morphology[-1]=='--export-scoring-geometry'
    assert weights[:2]==original
    assert correction_arguments('spatch_gene_bin',original)==original


def test_unsupported_counts_remain_in_totals_but_not_he_fractions():
    import numpy as np
    from scipy import sparse
    from prepare_hd_flex_he_bin_weights import compartment_weights
    from score_hd_flex_star_sr_gene_bin_residuals import analyse_method
    weights = compartment_weights(np.array([2, 0]), np.array([0, 1]), np.array([1, 1, 0]), True)
    weights['supported'] = np.array([1., 1., 0.])
    summary, rows = analyse_method(dataset='synthetic', method='hard', gene_ids=['g', 'h'], gene_names=['G', 'H'],
        star=sparse.csr_matrix([[8., 2., 90.], [0., 0., 0.]]), space_ranger=sparse.csr_matrix([[4., 1., 45.], [0., 0., 0.]]), weights=weights)
    assert summary['star_mass_on_space_ranger_axis'] == 100
    assert summary['space_ranger_mass'] == 50
    assert summary['star_whole_cell_mass_fraction'] == .8
    assert summary['space_ranger_whole_cell_mass_fraction'] == .8
    assert rows[0]['star_positive_residual_cell_mass_fraction'] == .8
