"""Fixed eval-set loading (M0).

Evolution code must only ever touch the dev set (`eval_indices.json`).
`load_heldout` exists solely for scripts/final_eval.py and refuses to run if
imported from anywhere else (design constraint 2).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evomath import REPO_ROOT

_CALC = re.compile(r"<<[^>]*>>")


def _gsm8k_split(split: str):
    from datasets import load_dataset
    return load_dataset("gsm8k", "main", split=split)


def _load(indices_path: str | Path) -> list[dict]:
    p = Path(indices_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    indices = json.loads(p.read_text())
    ds = _gsm8k_split("test")
    return _docs_from_dataset(ds, indices)


def _docs_from_dataset(ds, indices=None) -> list[dict]:
    docs = []
    if indices is None:
        indices = range(len(ds))
    for i in indices:
        d = ds[int(i)]
        solution = _CALC.sub("", d["answer"])
        reasoning, gold = solution.rsplit("####", 1)
        gold = gold.strip().replace(",", "").replace("$", "")
        docs.append({
            "index": int(i),
            "question": d["question"].strip(),
            "gold_text": d["answer"],                      # raw, for parse_gold
            "solution": f"{reasoning.strip()} The answer is {gold}.",  # loglik target
        })
    return docs


def load_dev(cfg: dict) -> list[dict]:
    return _load(cfg["eval_indices"])


def load_train(cfg: dict) -> list[dict]:
    return _docs_from_dataset(_gsm8k_split("train"))


def load_test(cfg: dict) -> list[dict]:
    return _docs_from_dataset(_gsm8k_split("test"))


def load_heldout(cfg: dict) -> list[dict]:
    import inspect
    caller = inspect.stack()[1].filename
    if Path(caller).name != "final_eval.py":
        raise RuntimeError(
            "heldout set may only be read by scripts/final_eval.py "
            f"(attempted from {caller})"
        )
    return _load(cfg["heldout_indices"])
