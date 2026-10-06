"""Read the manuscript's number macros so figures show the same values as the text."""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
FILES = ("numbers.tex", "design_numbers.tex", "audit_numbers.tex")


def load():
    macros = {}
    for name in FILES:
        text = (PAPER / name).read_text()
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text):
            macros[m.group(1)] = m.group(2)
        for m in re.finditer(r"\\let\\(\w+)=\\(\w+)", text):
            macros[m.group(1)] = macros.get(m.group(2), "")
    return macros
