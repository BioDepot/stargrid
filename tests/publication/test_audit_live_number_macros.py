from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/publication"))
from audit_live_number_macros import audit, definitions


def test_nested_definitions_and_transitive_aliases(tmp_path):
    main = tmp_path / "main.tex"
    numbers = tmp_path / "numbers.tex"
    main.write_text(r"\input{numbers}\input{results}")
    numbers.write_text(r"\newcommand{\countA}{1\,{234}}\let\countB=\countA" + "\n" + r"\newcommand{\old}{999}")
    (tmp_path / "results.tex").write_text("% \\old is commented\n" + r"\countB{}")
    (tmp_path / "abstract_extended.tex").write_text(r"\old{}")
    result = audit(main, numbers)
    assert result["macros"]["countA"]["live"]
    assert result["macros"]["countB"]["live"]
    assert not result["macros"]["old"]["live"]
    assert result["live_count"] == 2
    assert len(result["included_files"]) == 3


def test_duplicate_definitions_fail():
    with pytest.raises(ValueError, match="duplicate"):
        definitions(r"\newcommand{\x}{1}\newcommand{\x}{2}")
