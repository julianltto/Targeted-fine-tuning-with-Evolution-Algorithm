"""Generate data/eval_indices.json (500 dev) + data/heldout_indices.json (819).

Both from GSM8K *test* (1319): avoids MetaMathQA<-GSM8K-train leakage into dev.
Refuses to overwrite existing files — these indices are frozen forever
(CLAUDE.md: 一经生成永不改动).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEV_N = 500
SEED = 42


def main():
    dev_p = REPO / "data/eval_indices.json"
    held_p = REPO / "data/heldout_indices.json"
    if dev_p.exists() or held_p.exists():
        raise SystemExit(f"refusing to overwrite frozen indices: {dev_p} / {held_p}")

    import os
    os.environ.setdefault("HF_HOME", "/home/iailab71/wul1/hf_cache")
    from datasets import load_dataset
    n_test = len(load_dataset("gsm8k", "main", split="test"))
    assert n_test == 1319, f"unexpected GSM8K test size {n_test}"

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_test)
    dev = sorted(int(i) for i in perm[:DEV_N])
    held = sorted(int(i) for i in perm[DEV_N:])
    assert not set(dev) & set(held)

    dev_p.write_text(json.dumps(dev))
    held_p.write_text(json.dumps(held))
    print(f"wrote {dev_p} ({len(dev)}) and {held_p} ({len(held)}), seed={SEED}")


if __name__ == "__main__":
    main()
