import copy
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).parents[2] / 'scripts/publication/compare_spatial_codex_generations.py'
spec = importlib.util.spec_from_file_location('compare_codex_generations', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    summary = {'parameters': {'bin_sizes_um': [100, 200]}, 'coordinate_serialization': {'fixed': True},
               'inputs': {k: {'sha256': k} for k in ('codex_h5ad', 'probe_csv', 'space_ranger_h5ad', 'space_ranger_filtered_probe_set')}}
    tables = {'method_summary.tsv': [], 'marker_metrics.tsv': []}
    for scale in ('100', '200'):
        for method in ('space_ranger_2020a',) + tuple('old_' + p for p in module.POLICIES):
            tables['method_summary.tsv'].append(dict(method=method, bin_size_um=scale, markers_scored='1',
                                                     **{k: '0.5' for k in module.METRICS}))
            tables['marker_metrics.tsv'].append(dict(method=method, bin_size_um=scale, protein='CD8',
                                                     **{k: 'fixed' for k in module.COHORT}))
    new = copy.deepcopy(tables)
    for rows in new.values():
        for row in rows:
            row['method'] = row['method'].replace('old_', 'new_')
    return summary, tables, new


def test_preserves_all_scales_policies_and_signed_changes():
    summary, old, new = fixture()
    new['method_summary.tsv'][1]['macro_roc_auc'] = '0.4'
    rows = module.compare(summary, summary, old, new, 'old_', 'new_', '2020a')
    assert len(rows) == 2 * 4 * 4
    changed = [r for r in rows if r['absolute_change']]
    assert len(changed) == 1
    assert changed[0]['absolute_change'] == pytest.approx(-0.1)


@pytest.mark.parametrize('change', ['vendor', 'cohort', 'marker', 'transform'])
def test_rejects_changed_comparison_scope(change):
    summary, old, new = fixture()
    new_summary = copy.deepcopy(summary)
    if change == 'vendor':
        new['method_summary.tsv'][0]['macro_roc_auc'] = '0.7'
    elif change == 'cohort':
        new['marker_metrics.tsv'][1]['evaluated_bins'] = 'other'
    elif change == 'marker':
        new['marker_metrics.tsv'][1]['protein'] = 'CD3'
    else:
        new_summary['coordinate_serialization'] = {'fixed': False}
    with pytest.raises(ValueError):
        module.compare(summary, new_summary, old, new, 'old_', 'new_', '2020a')
