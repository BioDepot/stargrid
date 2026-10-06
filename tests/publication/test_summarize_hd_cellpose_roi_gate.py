import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/publication/summarize_hd_cellpose_roi_gate.py"
SPEC = importlib.util.spec_from_file_location("summarize_hd_cellpose_roi_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_gate_requires_both_whole_field_and_additional_mass_support():
    def row(mass, value):
        return {
            "raw_molecule_mass": str(mass),
            "in_cell_roc_auc": str(value),
            "in_nucleus_roc_auc": str(value),
            "cell_fraction_pearson_16um": str(value),
            "nucleus_fraction_pearson_16um": str(value),
        }

    rows = {"space_ranger": row(100, 0.6)}
    for policy in MODULE.POLICIES:
        rows[policy] = row(110, 0.7)
        rows[f"{policy}_star_only"] = row(20, 0.65)
        rows[f"{policy}_space_ranger_only"] = row(10, 0.55)
    segmentation = {
        "counts": {"retained_nucleus_labels": 3, "nucleus_outside_cell_pixels": 0},
        "input_contract": {"expression_or_vendor_segmentation_loaded": False},
    }
    scoring = {"roi_geometry_audit": {
        "cellpose_supported_roi_2um_bins": 999,
        "native_roi_2um_bins": 1000,
    }}
    checks, passed = MODULE.evaluate(rows, segmentation, scoring)
    assert passed
    assert len(checks) == 18

    rows["postcollapse_soft_star_only"]["in_cell_roc_auc"] = "0.5"
    checks, passed = MODULE.evaluate(rows, segmentation, scoring)
    assert not passed
    assert any(not row["pass"] for row in checks)
