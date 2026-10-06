import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "plot_flex_star_sr_he_consistency.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_flex_star_sr_he_consistency", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_rows_rejects_incomplete_dataset_method_grid(tmp_path):
    path = tmp_path / "table.tsv"
    columns = sorted(MODULE.REQUIRED_COLUMNS)
    row = {column: "0" for column in columns}
    row["dataset"] = "CRC"
    row["method"] = "hard"
    path.write_text(
        "\t".join(columns) + "\n"
        + "\t".join(row[column] for column in columns) + "\n",
        encoding="utf-8",
    )
    try:
        MODULE.load_rows(path)
    except ValueError as exc:
        assert "unexpected dataset/method rows" in str(exc)
    else:
        raise AssertionError("incomplete figure table was accepted")


def test_row_index_uses_dataset_and_method():
    rows = [
        {"dataset": "CRC", "method": "hard"},
        {"dataset": "SPATCH", "method": "soft_expected"},
    ]
    assert MODULE.row_index(rows) == {
        ("CRC", "hard"): rows[0],
        ("SPATCH", "soft_expected"): rows[1],
    }
