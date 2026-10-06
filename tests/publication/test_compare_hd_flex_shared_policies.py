from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "compare_hd_flex_shared_policies.py"
SPEC = importlib.util.spec_from_file_location("shared_policies", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def assignment(gene: str, umi: str, row: int, col: int) -> dict[str, object]:
    return {"gene": gene, "umi": umi, "coordinate": (row, col)}


def test_policy_evaluation_separates_strict_and_ambiguous_increment() -> None:
    vendor = {
        "strict": assignment("g", "A", 0, 0),
        "ambiguous": assignment("g", "B", 0, 1),
    }
    products = {
        "strict": {"strict": assignment("g", "A", 0, 0)},
        "postcollapse_hard": {
            "strict": assignment("g", "A", 0, 0),
            "ambiguous": assignment("g", "X", 0, 2),
        },
        "gated_hard": {"strict": assignment("g", "A", 0, 0)},
    }
    cliques = {
        "c1": {"members": ("strict",), "candidates": [((0, 0), 1.0)]},
        "c2": {"members": ("ambiguous",), "candidates": [((0, 1), 0.75), ((0, 2), 0.25)]},
    }
    rows, identity = MODULE.evaluate_policies(
        vendor, products, cliques, {"strict": "c1", "ambiguous": "c2"},
    )
    assert identity == {
        "shared_reads": 2, "gene_concordant_reads": 2,
        "corrected_umi_concordant_reads": 1,
        "strict_shared_reads": 1, "ambiguous_increment_reads": 1,
    }
    soft_ambiguous_2um = next(
        row for row in rows
        if row["subset"] == "ambiguous_increment"
        and row["policy"] == "soft_expected" and row["scale_um"] == 2
    )
    assert soft_ambiguous_2um["compatible_support"] == 0.75
    hard_ambiguous_2um = next(
        row for row in rows
        if row["subset"] == "ambiguous_increment"
        and row["policy"] == "hard" and row["scale_um"] == 2
    )
    assert hard_ambiguous_2um["conditional_fraction"] == 0.0
    reversed_rows, reversed_identity = MODULE.evaluate_policies(
        dict(reversed(list(vendor.items()))),
        {name: dict(reversed(list(values.items()))) for name, values in products.items()},
        dict(reversed(list(cliques.items()))),
        {"ambiguous": "c2", "strict": "c1"},
    )
    assert reversed_identity == identity
    assert reversed_rows == rows
