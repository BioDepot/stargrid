#!/usr/bin/env python3
"""Inventory numerical macros reachable from the manuscript's actual inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CALL = re.compile(r"\\([A-Za-z@]+)")
INCLUDE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
DEFINE = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))")
ALIAS = re.compile(r"\\let\s*\\([A-Za-z@]+)\s*=?\s*\\([A-Za-z@]+)")


def uncomment(text):
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


def balanced(text, start, opening="{", closing="}"):
    if text[start] != opening:
        raise ValueError("expected balanced group")
    depth, index = 1, start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
        index += 1
    raise ValueError("unclosed TeX group")


def definitions(text):
    found, spans = {}, []
    for match in DEFINE.finditer(text):
        name = match[1] or match[2]
        index = match.end()
        while index < len(text) and text[index].isspace():
            index += 1
        while index < len(text) and text[index] == "[":
            _, index = balanced(text, index, "[", "]")
            while index < len(text) and text[index].isspace():
                index += 1
        body, end = balanced(text, index)
        if name in found:
            raise ValueError(f"duplicate definition: {name}")
        found[name] = {"body": body, "dependencies": sorted(set(CALL.findall(body))), "kind": "command"}
        spans.append((match.start(), end))
    for match in ALIAS.finditer(text):
        if match[1] in found:
            raise ValueError(f"duplicate definition: {match[1]}")
        found[match[1]] = {"body": "\\" + match[2], "dependencies": [match[2]], "kind": "alias"}
        spans.append(match.span())
    chars = list(text)
    for begin, end in spans:
        chars[begin:end] = " " * (end - begin)
    return found, "".join(chars)


def audit(main, numbers):
    root, visited, active = main.parent.resolve(), {}, set()
    all_definitions, used = {}, set()
    def visit(path):
        path = path.resolve()
        if path in active:
            raise ValueError(f"recursive TeX input: {path}")
        if path in visited:
            return
        raw = path.read_text()
        visited[path] = hashlib.sha256(raw.encode()).hexdigest()
        active.add(path)
        defs, body = definitions(uncomment(raw))
        for name, definition in defs.items():
            if name in all_definitions:
                raise ValueError(f"duplicate manuscript macro: {name}")
            all_definitions[name] = dict(definition, source=str(path))
        used.update(CALL.findall(body))
        for match in INCLUDE.finditer(body):
            child = root / match[1]
            if not child.suffix:
                child = child.with_suffix(".tex")
            visit(child)
        active.remove(path)
    visit(main)
    pending = list(used)
    while pending:
        name = pending.pop()
        for dependency in all_definitions.get(name, {}).get("dependencies", []):
            if dependency not in used:
                used.add(dependency)
                pending.append(dependency)
    number_defs, _ = definitions(uncomment(numbers.read_text()))
    return {
        "schema": "visium_hd_processing.live_number_macros.v1",
        "main": str(main.resolve()), "included_files": {str(k): v for k, v in visited.items()},
        "macros": {name: dict(value, live=name in used) for name, value in sorted(number_defs.items())},
        "live_count": sum(name in used for name in number_defs),
        "unused_count": sum(name not in used for name in number_defs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--numbers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = audit(args.main, args.numbers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
