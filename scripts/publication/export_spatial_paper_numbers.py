#!/usr/bin/env python3
"""Export paper macros and an old/new audit from checksum-pinned result fields."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from audit_live_number_macros import audit
from summarize_sealed_spatial_primary import sha256


def field(value, path):
    for key in path:
        value = value[key]
    return value


def evaluate(expression, sources):
    if isinstance(expression, (int, float, str)):
        return expression
    if "source" in expression:
        return field(sources[expression["source"]], expression["path"])
    operation = expression["op"]
    if operation == "length" and len(expression["args"]) == 1:
        value = evaluate(expression["args"][0], sources)
        if not isinstance(value, (list, dict)):
            raise ValueError("length requires a result list or mapping")
        return len(value)
    values = [float(evaluate(item, sources)) for item in expression["args"]]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite or empty numeric expression")
    if operation == "sum":
        return sum(values)
    if operation == "difference" and len(values) == 2:
        return values[0] - values[1]
    if operation == "ratio" and len(values) == 2:
        return values[0] / values[1]
    if operation == "product":
        return math.prod(values)
    if operation in ("min", "max"):
        return (min if operation == "min" else max)(values)
    raise ValueError(f"unsupported expression: {operation}")


def display(value, style):
    kind = style["kind"]
    if kind == "text":
        return style.get("prefix", "") + str(value) + style.get("suffix", "")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cannot publish a nonfinite number")
    if kind == "integer":
        if number != round(number):
            raise ValueError("integer formatting would discard fractional mass")
        return format(int(number), ",d" if style.get("group", True) else "d")
    if kind == "decimal":
        return format(number, f".{style['places']}f")
    if kind == "scientific":
        mantissa, exponent = format(number, f".{style['places']}e").split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    if kind == "duration":
        seconds = round(number)
        if seconds < 0:
            raise ValueError("negative duration")
        return f"{seconds // 60}:{seconds % 60:02d}"
    raise ValueError(f"unsupported formatting: {kind}")


def displayed_number(body):
    """Recover only a displayed number, never imply unrounded historical data."""
    text = body.replace(r"\,", " ").replace(",", "").strip()
    scientific = re.fullmatch(r"([+-]?[\d.]+)\\times10\^\{([+-]?\d+)\}", text)
    if scientific:
        return float(scientific[1]) * 10 ** int(scientific[2])
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return float(text)
    duration = text.replace(r"\,", " ")
    if re.fullmatch(r"(?:\d+\s*(?:h|min|s)\s*)+", duration):
        return sum(float(n) * {"h": 3600, "min": 60, "s": 1}[unit]
                   for n, unit in re.findall(r"(\d+)\s*(h|min|s)", duration))
    if re.fullmatch(r"\d+:\d{2}(?::\d{2})?", duration):
        value = 0
        for part in duration.split(":"):
            value = value * 60 + int(part)
        return float(value)
    return None


def export(manifest, main, numbers, output):
    if output.exists():
        raise ValueError(f"refusing existing export: {output}")
    inventory = audit(main, numbers)
    sources = {}
    for name, item in manifest["sources"].items():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source hash drift: {path}")
        if item.get("format", "json") == "json":
            sources[name] = json.loads(path.read_text())
        elif item["format"] == "tsv":
            with path.open() as handle:
                sources[name] = list(csv.DictReader(handle, delimiter="\t"))
        else:
            raise ValueError("only JSON and TSV result inputs are supported")
    specs = manifest["macros"]
    unknown = set(specs) - inventory["macros"].keys()
    unapproved = {name for name in unknown if not specs[name].get('new_macro')}
    if unapproved:
        raise ValueError(f"unknown macros: {sorted(unapproved)}")
    for name in sorted(unknown):
        if not re.fullmatch(r'[A-Za-z]+',name) or specs[name].get('status')=='pending_decision':
            raise ValueError(f'invalid new evidence macro: {name}')
        inventory['macros'][name] = {'body':'','dependencies':[],'kind':'command','live':False,
                                     'new_definition':True}
    rows, resolved, active = {}, {}, set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in active:
            raise ValueError(f"cyclic alias: {name}")
        active.add(name)
        definition = inventory["macros"][name]
        spec = specs.get(name)
        row = {"old_display": definition["body"], "live": definition["live"]}
        if not definition["live"] and spec is None:
            row.update(status="historical_unused", new_display=definition["body"])
        elif spec is None:
            raise ValueError(f"unmapped live macro: {name}")
        else:
            status = spec["status"]
            if status not in ("regenerated", "verified_carry_forward", "pending_decision"):
                raise ValueError(f"invalid status: {name}")
            row.update(spec)
            if definition["kind"] == "alias":
                target = definition["dependencies"][0]
                target_row = resolve(target)
                if status != target_row["status"]:
                    raise ValueError(f"alias status differs from target: {name}")
                row.update(raw_value=target_row.get("raw_value"),
                           new_display=definition["body"], alias_of=target,
                           resolved_display=target_row.get("resolved_display", target_row["new_display"]))
                for key in ("result_id", "provenance_run_id", "scope", "verification",
                            "old_numeric_display", "absolute_change_from_old_display",
                            "relative_change_percent_from_old_display", "change_basis",
                            "old_raw_value", "old_result_id", "old_provenance_run_id",
                            "old_evidence_status", "absolute_change", "relative_change_percent"):
                    if key in target_row:
                        row[key] = target_row[key]
            elif status == "pending_decision":
                if not spec.get("reason"):
                    raise ValueError(f"pending macro needs reason: {name}")
                row["new_display"] = definition["body"]
            else:
                if not all(spec.get(key) for key in ("result_id", "provenance_run_id", "scope")):
                    raise ValueError(f"missing result provenance or scope: {name}")
                if status == "verified_carry_forward" and not spec.get("verification"):
                    raise ValueError(f"carry-forward needs verification: {name}")
                expression = spec["value"]
                # Values must derive from recorded evidence; numeric constants are
                # allowed only as operands in calculations (e.g. percentages).
                if not isinstance(expression, dict) or "source" not in json.dumps(expression):
                    raise ValueError(f"macro has no evidence source: {name}")
                value = evaluate(expression, sources)
                row.update(raw_value=value, new_display=display(value, spec["display"]))
                old_evidence = spec.get("old_evidence")
                if old_evidence is not None:
                    if not all(old_evidence.get(key) for key in ("result_id", "provenance_run_id")):
                        raise ValueError(f"missing historical result provenance: {name}")
                    old_expression = old_evidence.get("value")
                    if not isinstance(old_expression, dict) or "source" not in json.dumps(old_expression):
                        raise ValueError(f"historical raw value has no evidence source: {name}")
                    old_value = evaluate(old_expression, sources)
                    row.update(old_raw_value=old_value,
                               old_result_id=old_evidence["result_id"],
                               old_provenance_run_id=old_evidence["provenance_run_id"],
                               old_evidence_status="checksum-pinned historical raw evidence")
                    if spec["display"]["kind"] != "text":
                        old_numeric = float(old_value)
                        if not math.isfinite(old_numeric):
                            raise ValueError(f"nonfinite historical raw value: {name}")
                        delta = float(value) - old_numeric
                        row.update(absolute_change=delta,
                                   relative_change_percent=(100 * delta / abs(old_numeric) if old_numeric else None))
                else:
                    row.update(old_raw_value=None,
                               old_evidence_status="historical raw evidence not bound; displayed value only")
                old_number = displayed_number(definition["body"])
                if old_number is not None and spec["display"]["kind"] != "text":
                    delta = float(value) - old_number
                    row.update(old_numeric_display=old_number,
                               absolute_change_from_old_display=delta,
                               relative_change_percent_from_old_display=(
                                   100 * delta / abs(old_number) if old_number else None),
                               change_basis="new unrounded evidence versus old displayed value")
        active.remove(name)
        rows[name] = row
        resolved[name] = row
        return row

    for name in inventory["macros"]:
        resolve(name)
    pending = [name for name, row in rows.items() if row["live"] and row["status"] == "pending_decision"]
    lines = ["% Generated by export_spatial_paper_numbers.py; see coverage.json.",
             "% Every active value names its evidence result, run and scope below.",
             "% Historical unused definitions retain their original values."]
    # TeX's \let copies the target's current definition, so forward aliases
    # must follow their targets even when the inventory is alphabetic.
    ordered, emitted, emitting = [], set(), set()
    def order(name):
        if name in emitted:
            return
        if name in emitting:
            raise ValueError(f"cyclic alias definition: {name}")
        emitting.add(name)
        definition = inventory["macros"][name]
        if definition["kind"] == "alias":
            target = definition["dependencies"][0]
            if target in inventory["macros"]:
                order(target)
        emitting.remove(name)
        emitted.add(name)
        ordered.append(name)
    for name in inventory["macros"]:
        order(name)
    for name in ordered:
        definition = inventory["macros"][name]
        row = rows[name]
        lines.append(f"% {name}: {row['status']}")
        for key in ("result_id", "provenance_run_id", "scope", "reason"):
            if row.get(key):
                lines.append(f"% {key}: {' '.join(str(row[key]).splitlines())}")
        if definition["kind"] == "alias":
            lines.append(f"\\let\\{name}={row['new_display']}")
        else:
            lines.append(f"\\newcommand{{\\{name}}}{{{row['new_display']}}}")
    output.mkdir(parents=True)
    (output / "numbers.tex").write_text("\n".join(lines) + "\n")
    result = {"schema": "visium_hd_processing.paper_number_export.v1",
              "release": manifest["release"], "publication_ready": not pending,
              "pending_live_macros": pending, "sources": manifest["sources"],
              "input_inventory": inventory, "macros": rows,
              "numbers_sha256": sha256(output / "numbers.tex")}
    (output / "coverage.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output / "changes.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["macro", "live", "status", "old", "new",
                         "old_raw_value", "new_raw_value", "absolute_change", "relative_change_percent",
                         "old_result_id", "old_provenance_run_id", "old_evidence_status",
                         "absolute_change_from_old_display", "relative_change_percent_from_old_display",
                         "change_basis", "result_id", "provenance_run_id", "scope", "carry_forward_reason"])
        for name, row in rows.items():
            writer.writerow([name, row["live"], row["status"], row["old_display"],
                             row.get("resolved_display", row["new_display"]),
                             *[row.get(key) if row.get(key) is not None else "not_applicable" for key in (
                                 "old_raw_value", "raw_value", "absolute_change", "relative_change_percent",
                                 "old_result_id", "old_provenance_run_id", "old_evidence_status",
                                 "absolute_change_from_old_display", "relative_change_percent_from_old_display",
                                 "change_basis", "result_id", "provenance_run_id", "scope")],
                             row.get("verification", row.get("reason", ""))])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--numbers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(json.loads(args.manifest.read_text()), args.main, args.numbers, args.out)


if __name__ == "__main__":
    main()
