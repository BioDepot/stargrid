from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publication" / "export_hd_space_ranger_read_ledger.py"
SPEC = importlib.util.spec_from_file_location("vendor_ledger", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Record:
    def __init__(self, tags: dict[str, object], flag: int = 0):
        self.query_name = "read"
        self.flag = flag
        self.tags = tags

    def has_tag(self, name: str) -> bool:
        return name in self.tags

    def get_tag(self, name: str) -> object:
        return self.tags[name]


def test_record_eligibility_requires_target_gene_coordinate_umi_and_xf() -> None:
    tags = {"sb": "s_002um_1_2-1", "GX": "g1", "UB": "AAAA", "xf": 1, "pr": "p1"}
    row = MODULE.record_to_row(Record(tags, 1024), {"g1"})
    assert row["eligible"] == 1
    assert row["unit_2um"] == "s_002um_1_2-1"
    assert row["duplicate"] == 1
    assert MODULE.record_to_row(Record(tags | {"GX": "off_target"}), {"g1"})["eligible"] == 0
    assert MODULE.record_to_row(Record(tags | {"xf": 0}), {"g1"})["eligible"] == 0
    assert MODULE.record_to_row(Record(tags | {"UB": ""}), {"g1"})["eligible"] == 0
