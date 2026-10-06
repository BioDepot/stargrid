import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "plot_flex_he_morphology_consistency.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_flex_he_morphology_consistency", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_rows_requires_complete_dataset_policy_grid(tmp_path):
    path = tmp_path / "table.tsv"
    columns = sorted(MODULE.REQUIRED_COLUMNS)
    values = {column: "0" for column in columns}
    values["dataset"] = "CRC"
    values["policy"] = "hard"
    path.write_text(
        "\t".join(columns) + "\n"
        + "\t".join(values[column] for column in columns) + "\n",
        encoding="utf-8",
    )
    try:
        MODULE.load_rows(path)
    except ValueError as exc:
        assert "unexpected dataset/policy rows" in str(exc)
    else:
        raise AssertionError("incomplete comparison grid was accepted")


def test_row_index_keys_dataset_and_policy():
    rows = [
        {"dataset": "CRC", "policy": "hard"},
        {"dataset": "SPATCH", "policy": "soft_expected"},
    ]
    assert MODULE.row_index(rows) == {
        ("CRC", "hard"): rows[0],
        ("SPATCH", "soft_expected"): rows[1],
    }
