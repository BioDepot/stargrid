from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


HD_DECODED_ASSIGNMENT_SCHEMA = "star_spatial.hd.decoded_assignment.v1"

HD_DECODED_ASSIGNMENT_HEADER = (
    "assignment_group_id",
    "candidate_index",
    "candidate_count",
    "feature_id",
    "feature_name",
    "umi",
    "row2",
    "col2",
    "unit_2um_id",
    "unit_8um_id",
    "unit_16um_id",
    "sequence_status",
    "sequence_score",
    "sequence_rank",
    "score0",
    "score1",
    "score2",
    "score3",
    "posterior_weight",
    "resolver",
    "resolver_version",
    "resolver_margin",
    "source",
    "read_id",
    "raw_bc_sequence",
    "bc1_edit",
    "bc2_edit",
    "anchor_score",
    "offset_sum",
    "full_start",
    "bc1_obs_len",
    "bc2_obs_len",
    "same_parent_8um",
    "same_parent_16um",
    "cell_id",
    "region_id",
)

SEQUENCE_STATUSES = {
    "unique",
    "top_tie",
    "near_tie",
    "rejected",
    "no_candidate",
}

RESOLVERS = {
    "sequence",
    "same_parent",
    "expression_prior",
    "bin_support",
    "unresolved",
    "rejected",
}


class HdAssignmentLedgerError(ValueError):
    """Raised when an HD decoded-assignment ledger row is malformed."""


@dataclass(frozen=True)
class HdUnitIds:
    unit_2um_id: str
    unit_8um_id: str
    unit_16um_id: str


@dataclass(frozen=True)
class HdDecodedAssignmentRecord:
    assignment_group_id: str
    candidate_index: int
    candidate_count: int
    feature_id: str = ""
    feature_name: str = ""
    umi: str = ""
    row2: int | None = None
    col2: int | None = None
    unit_2um_id: str = ""
    unit_8um_id: str = ""
    unit_16um_id: str = ""
    sequence_status: str = "unique"
    sequence_score: float | None = None
    sequence_rank: int | None = None
    score0: float | None = None
    score1: float | None = None
    score2: float | None = None
    score3: float | None = None
    posterior_weight: float = 1.0
    resolver: str = "sequence"
    resolver_version: str = "v1"
    resolver_margin: float | None = None
    source: str = ""
    read_id: str = ""
    raw_bc_sequence: str = ""
    bc1_edit: int | None = None
    bc2_edit: int | None = None
    anchor_score: int | None = None
    offset_sum: int | None = None
    full_start: int | None = None
    bc1_obs_len: int | None = None
    bc2_obs_len: int | None = None
    same_parent_8um: bool | None = None
    same_parent_16um: bool | None = None
    cell_id: str = ""
    region_id: str = ""

    def as_tsv_row(self) -> list[str]:
        return [
            self.assignment_group_id,
            str(self.candidate_index),
            str(self.candidate_count),
            self.feature_id,
            self.feature_name,
            self.umi,
            _format_optional_int(self.row2),
            _format_optional_int(self.col2),
            self.unit_2um_id,
            self.unit_8um_id,
            self.unit_16um_id,
            self.sequence_status,
            _format_optional_float(self.sequence_score),
            _format_optional_int(self.sequence_rank),
            _format_optional_float(self.score0),
            _format_optional_float(self.score1),
            _format_optional_float(self.score2),
            _format_optional_float(self.score3),
            _format_optional_float(self.posterior_weight),
            self.resolver,
            self.resolver_version,
            _format_optional_float(self.resolver_margin),
            self.source,
            self.read_id,
            self.raw_bc_sequence,
            _format_optional_int(self.bc1_edit),
            _format_optional_int(self.bc2_edit),
            _format_optional_int(self.anchor_score),
            _format_optional_int(self.offset_sum),
            _format_optional_int(self.full_start),
            _format_optional_int(self.bc1_obs_len),
            _format_optional_int(self.bc2_obs_len),
            _format_optional_bool(self.same_parent_8um),
            _format_optional_bool(self.same_parent_16um),
            self.cell_id,
            self.region_id,
        ]


def bin_unit_id(row2: int, col2: int, size_um: int = 2) -> str:
    if size_um <= 0 or size_um % 2:
        raise HdAssignmentLedgerError(f"HD bin size must be a positive even um value: {size_um}")
    if row2 < 0 or col2 < 0:
        raise HdAssignmentLedgerError(f"HD bin coordinates must be non-negative: {row2}, {col2}")
    factor = size_um // 2
    return f"s_{size_um:03d}um_{row2 // factor}_{col2 // factor}"


def hd_unit_ids(row2: int, col2: int) -> HdUnitIds:
    return HdUnitIds(
        unit_2um_id=bin_unit_id(row2, col2, 2),
        unit_8um_id=bin_unit_id(row2, col2, 8),
        unit_16um_id=bin_unit_id(row2, col2, 16),
    )


def make_hd_decoded_assignment_record(
    assignment_group_id: str,
    candidate_index: int,
    candidate_count: int,
    *,
    row2: int | None,
    col2: int | None,
    sequence_status: str = "unique",
    resolver: str = "sequence",
    resolver_version: str = "v1",
    posterior_weight: float = 1.0,
    **fields: Any,
) -> HdDecodedAssignmentRecord:
    units = HdUnitIds("", "", "")
    if row2 is not None and col2 is not None:
        units = hd_unit_ids(row2, col2)
    record = HdDecodedAssignmentRecord(
        assignment_group_id=assignment_group_id,
        candidate_index=candidate_index,
        candidate_count=candidate_count,
        row2=row2,
        col2=col2,
        unit_2um_id=fields.pop("unit_2um_id", units.unit_2um_id),
        unit_8um_id=fields.pop("unit_8um_id", units.unit_8um_id),
        unit_16um_id=fields.pop("unit_16um_id", units.unit_16um_id),
        sequence_status=sequence_status,
        resolver=resolver,
        resolver_version=resolver_version,
        posterior_weight=posterior_weight,
        **fields,
    )
    _validate_record(record, Path("record"))
    return record


def annotate_same_parent_flags(
    records: Iterable[HdDecodedAssignmentRecord],
) -> list[HdDecodedAssignmentRecord]:
    rows = list(records)
    by_group: dict[str, list[HdDecodedAssignmentRecord]] = defaultdict(list)
    for row in rows:
        by_group[row.assignment_group_id].append(row)
    flags: dict[str, tuple[bool, bool]] = {}
    for group_id, group in by_group.items():
        unit8 = {row.unit_8um_id for row in group if row.unit_8um_id}
        unit16 = {row.unit_16um_id for row in group if row.unit_16um_id}
        flags[group_id] = (len(unit8) == 1, len(unit16) == 1)
    return [
        replace(row, same_parent_8um=flags[row.assignment_group_id][0],
                same_parent_16um=flags[row.assignment_group_id][1])
        for row in rows
    ]


def write_hd_decoded_assignment_records(
    path: str | Path,
    records: Iterable[HdDecodedAssignmentRecord],
    overwrite: bool = False,
) -> dict[str, Any]:
    out_path = Path(path)
    rows = list(records)
    summary = validate_hd_decoded_assignment_records(rows)
    _prepare_output_file(out_path, overwrite)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HD_DECODED_ASSIGNMENT_HEADER)
        for row in rows:
            writer.writerow(row.as_tsv_row())
    return summary


def read_hd_decoded_assignment_records(path: str | Path) -> list[HdDecodedAssignmentRecord]:
    in_path = Path(path)
    with in_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(reader.fieldnames, HD_DECODED_ASSIGNMENT_HEADER, in_path)
        records = [
            _parse_record(row, row_no, in_path) for row_no, row in enumerate(reader, start=2)
        ]
    validate_hd_decoded_assignment_records(records)
    return records


def validate_hd_decoded_assignment_records(
    records: Iterable[HdDecodedAssignmentRecord],
) -> dict[str, Any]:
    rows = list(records)
    by_group: dict[str, list[HdDecodedAssignmentRecord]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    resolver_counts: Counter[str] = Counter()
    for row_no, row in enumerate(rows, start=2):
        _validate_record(row, Path("records"), row_no)
        by_group[row.assignment_group_id].append(row)
        status_counts[row.sequence_status] += 1
        resolver_counts[row.resolver] += 1

    ambiguous_groups = 0
    for group_id, group in by_group.items():
        expected_count = group[0].candidate_count
        indices = sorted(row.candidate_index for row in group)
        if any(row.candidate_count != expected_count for row in group):
            raise HdAssignmentLedgerError(
                f"group {group_id!r} has inconsistent candidate_count values"
            )
        if len(group) != expected_count:
            raise HdAssignmentLedgerError(
                f"group {group_id!r} has {len(group)} row(s), expected {expected_count}"
            )
        if indices != list(range(expected_count)):
            raise HdAssignmentLedgerError(
                f"group {group_id!r} candidate indices are not 0..{expected_count - 1}"
            )
        if expected_count > 1:
            ambiguous_groups += 1

    return {
        "schema": HD_DECODED_ASSIGNMENT_SCHEMA,
        "records": len(rows),
        "assignment_groups": len(by_group),
        "ambiguous_groups": ambiguous_groups,
        "sequence_status": dict(sorted(status_counts.items())),
        "resolver": dict(sorted(resolver_counts.items())),
    }


def _validate_record(
    record: HdDecodedAssignmentRecord,
    path: Path,
    row_no: int | None = None,
) -> None:
    label = f"{path}:{row_no}" if row_no is not None else str(path)
    if not record.assignment_group_id:
        raise HdAssignmentLedgerError(f"{label}: assignment_group_id is required")
    if record.candidate_count < 1:
        raise HdAssignmentLedgerError(f"{label}: candidate_count must be at least 1")
    if record.candidate_index < 0 or record.candidate_index >= record.candidate_count:
        raise HdAssignmentLedgerError(f"{label}: candidate_index is out of range")
    if record.sequence_status not in SEQUENCE_STATUSES:
        raise HdAssignmentLedgerError(
            f"{label}: unsupported sequence_status {record.sequence_status!r}"
        )
    if record.resolver not in RESOLVERS:
        raise HdAssignmentLedgerError(f"{label}: unsupported resolver {record.resolver!r}")
    if record.sequence_status == "unique" and record.candidate_count != 1:
        raise HdAssignmentLedgerError(f"{label}: unique records must have candidate_count=1")
    if record.sequence_status not in {"rejected", "no_candidate"}:
        if record.row2 is None or record.col2 is None:
            raise HdAssignmentLedgerError(
                f"{label}: candidate records require row2 and col2 coordinates"
            )
    if record.row2 is not None and record.row2 < 0:
        raise HdAssignmentLedgerError(f"{label}: row2 must be non-negative")
    if record.col2 is not None and record.col2 < 0:
        raise HdAssignmentLedgerError(f"{label}: col2 must be non-negative")
    if not (0.0 <= record.posterior_weight <= 1.0):
        raise HdAssignmentLedgerError(f"{label}: posterior_weight must be in [0, 1]")


def _parse_record(row: dict[str, str], row_no: int, path: Path) -> HdDecodedAssignmentRecord:
    try:
        return HdDecodedAssignmentRecord(
            assignment_group_id=row["assignment_group_id"],
            candidate_index=_parse_int(row["candidate_index"]),
            candidate_count=_parse_int(row["candidate_count"]),
            feature_id=row["feature_id"],
            feature_name=row["feature_name"],
            umi=row["umi"],
            row2=_parse_optional_int(row["row2"]),
            col2=_parse_optional_int(row["col2"]),
            unit_2um_id=row["unit_2um_id"],
            unit_8um_id=row["unit_8um_id"],
            unit_16um_id=row["unit_16um_id"],
            sequence_status=row["sequence_status"],
            sequence_score=_parse_optional_float(row["sequence_score"]),
            sequence_rank=_parse_optional_int(row["sequence_rank"]),
            score0=_parse_optional_float(row["score0"]),
            score1=_parse_optional_float(row["score1"]),
            score2=_parse_optional_float(row["score2"]),
            score3=_parse_optional_float(row["score3"]),
            posterior_weight=_parse_float(row["posterior_weight"]),
            resolver=row["resolver"],
            resolver_version=row["resolver_version"],
            resolver_margin=_parse_optional_float(row["resolver_margin"]),
            source=row["source"],
            read_id=row["read_id"],
            raw_bc_sequence=row["raw_bc_sequence"],
            bc1_edit=_parse_optional_int(row["bc1_edit"]),
            bc2_edit=_parse_optional_int(row["bc2_edit"]),
            anchor_score=_parse_optional_int(row["anchor_score"]),
            offset_sum=_parse_optional_int(row["offset_sum"]),
            full_start=_parse_optional_int(row["full_start"]),
            bc1_obs_len=_parse_optional_int(row["bc1_obs_len"]),
            bc2_obs_len=_parse_optional_int(row["bc2_obs_len"]),
            same_parent_8um=_parse_optional_bool(row["same_parent_8um"]),
            same_parent_16um=_parse_optional_bool(row["same_parent_16um"]),
            cell_id=row["cell_id"],
            region_id=row["region_id"],
        )
    except (KeyError, ValueError) as exc:
        raise HdAssignmentLedgerError(f"{path}:{row_no}: malformed record") from exc


def _prepare_output_file(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise HdAssignmentLedgerError(f"output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_columns(
    fieldnames: list[str] | None,
    required: tuple[str, ...],
    path: Path,
) -> None:
    if fieldnames is None:
        raise HdAssignmentLedgerError(f"empty decoded assignment TSV: {path}")
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise HdAssignmentLedgerError(
            f"decoded assignment TSV missing column(s) {', '.join(missing)}: {path}"
        )


def _parse_int(value: str) -> int:
    if value == "":
        raise ValueError("empty integer")
    return int(value)


def _parse_float(value: str) -> float:
    if value == "":
        raise ValueError("empty float")
    return float(value)


def _parse_optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _parse_optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _parse_optional_bool(value: str) -> bool | None:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid bool field: {value!r}")


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "True" if value else "False"
