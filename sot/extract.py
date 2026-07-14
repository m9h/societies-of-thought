"""Answer-extraction primitives, with no heavy dependencies.

Split out of data.py because grade.py needs extract_boxed, data.py imports `datasets`,
and that transitively made the GRADER unusable anywhere the ML stack isn't installed --
including the provenance verifier, which must run in a bare environment precisely so it
can be trusted to run at all. A check that cannot run is a check you cannot trust.
"""

from __future__ import annotations


def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST \boxed{...}, brace-matched."""
    key = "\\boxed"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    while i < len(text) and text[i] != "{":
        if not text[i].isspace():
            return None
        i += 1
    if i >= len(text):
        return None
    depth = 0
    out: list[str] = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return None
