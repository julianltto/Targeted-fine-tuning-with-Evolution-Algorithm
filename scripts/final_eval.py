"""Final held-out evaluation — the ONLY place allowed to read heldout_indices.json.

Stub for now (implemented at reporting time). Refuses to be imported so no
evolution code can route through it (design constraint 2).
"""

import sys
from pathlib import Path

if __name__ != "__main__":
    raise ImportError("final_eval.py must not be imported — run it directly")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

raise SystemExit("final_eval: not implemented yet — heldout stays untouched until final reporting")
