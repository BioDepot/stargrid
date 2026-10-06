import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "publication"
    / "prepare_hd_flex_star_sr_fields.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "prepare_hd_flex_star_sr_fields", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_method_requires_label_and_path():
    assert MODULE.parse_method("hard=/tmp/hard") == ("hard", Path("/tmp/hard"))
    try:
        MODULE.parse_method("/tmp/hard")
    except Exception as exc:
        assert "LABEL=MEX_DIR" in str(exc)
    else:
        raise AssertionError("unlabeled method was accepted")
