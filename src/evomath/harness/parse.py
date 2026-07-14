"""GSM8K answer parsing (CLAUDE.md §解析规范 — do not deviate).

gold: text after '####', strip commas/'$'/whitespace.
pred: LAST numeric match `-?[\\d,]*\\.?\\d+` in the generation, commas removed,
      compared as float with tolerance 1e-4. No match -> wrong, never crash.
"""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")
TOL = 1e-4


def parse_gold(answer_text: str) -> float:
    """Extract gold value from a GSM8K answer field ('... #### 42')."""
    if "####" in answer_text:
        tail = answer_text.rsplit("####", 1)[1]
    else:
        tail = answer_text
    tail = tail.replace(",", "").replace("$", "").strip()
    m = _NUM_RE.search(tail)
    if m is None:
        raise ValueError(f"gold answer has no numeric value: {answer_text!r}")
    return float(m.group())


def parse_pred(generated_text: str) -> float | None:
    """Last numeric token string in the model output; None if absent."""
    matches = _NUM_RE.findall(generated_text)
    if not matches:
        return None
    s = matches[-1].replace(",", "")
    # a bare '.' or '-' cannot reach here (regex requires trailing digit)
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(generated_text: str, gold_text: str) -> bool:
    pred = parse_pred(generated_text)
    if pred is None:
        return False
    return abs(pred - parse_gold(gold_text)) <= TOL
