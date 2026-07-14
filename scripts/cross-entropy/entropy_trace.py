"""Per-token entropy trace through a CoT: teacher-force, print each generated token with
the entropy of the model's next-token distribution at that step. Marks numeric tokens.
Shows whether entropy SPIKES at the erroneous reasoning decision or stays flat (confident
error). Compares a WRONG CoT to a CORRECT CoT on the same question.
Run: PYTHONPATH=. .venv/bin/python scripts/evoroute/entropy_trace.py
"""
import sys, re
sys.path.insert(0, "src")
import numpy as np
import torch
import torch.nn.functional as F
from evomath import load_config, REPO_ROOT
from evomath.harness import model as hmodel
from evomath.evoroute import dataio
from evomath.evoroute.arith import load_texts
from scripts.eval_track4_archive import build_8shot_prompt, load_8shot_samples, load_lora_from_delta

cfg = load_config("configs/full_lowmem.yaml")
members, docs, preds, gold, base_preds = dataio.load_population(cfg)
_, _, texts = load_texts(cfg)
model, tok = hmodel.load_base(cfg)
torch.set_grad_enabled(False)
dev = model.device
shots = load_8shot_samples()
DIGIT = re.compile(r"\d")

# (adapter_seed, pos, label). idx354: wrong 125 (skips 80) vs correct 45 ; idx148: wrong 14 vs correct 30
CASES = [(0, 354, "WRONG 125 (skips 80)"), (1, 354, "CORRECT 45"),
         (0, 148, "WRONG 14 (subtract change)"), (19, 148, "CORRECT 30")]


def trace(seed, pos):
    ind = load_lora_from_delta(REPO_ROOT / f"results/evomath/track4_sgd_ens/s0/sgd_ens_{seed}.pt")
    h = hmodel.patch_lora(model, ind)
    try:
        gen = texts[seed][pos].split("The answer is")[0] + "The answer is" + \
              (texts[seed][pos].split("The answer is")[1].split(".")[0] + "." if "The answer is" in texts[seed][pos] else "")
        prompt = build_8shot_prompt(docs[pos]["question"], shots)
        penc = tok(prompt, add_special_tokens=True)["input_ids"]
        full = tok(prompt + gen, add_special_tokens=True)["input_ids"]
        ids = torch.tensor([full]).to(dev)
        logits = model(input_ids=ids).logits[0].float()
        lp = F.log_softmax(logits, -1)
        ent = -(lp.exp() * lp).sum(-1)                     # entropy per position
        out = []
        for t in range(len(penc) - 1, len(full) - 1):      # positions predicting gen tokens
            tokstr = tok.decode([full[t + 1]])
            out.append((tokstr, float(ent[t])))
        return out
    finally:
        hmodel.unpatch_lora(h)


for seed, pos, label in CASES:
    print("=" * 90)
    print(f"idx {docs[pos]['index']} | adapter sgd_{seed} | {label} | GOLD={round(float(gold[pos]),4):g}")
    tr = trace(seed, pos)
    # print numeric tokens with entropy (the decision points), plus flag high-entropy tokens
    ents = [e for _, e in tr]
    hi = np.percentile(ents, 90) if ents else 0
    line = ""
    for tokstr, e in tr:
        mark = "**" if DIGIT.search(tokstr) else ""       # numeric token
        flag = "!" if e > hi else ""
        line += f"{tokstr}{mark}{flag}[{e:.2f}] "
    print(line.strip()[:1600])
    nums = [(t, e) for t, e in tr if DIGIT.search(t)]
    print(f"\n  numeric-token entropy: mean={np.mean([e for _,e in nums]):.3f}  "
          f"max={max((e for _,e in nums), default=0):.3f}  "
          f"(all-token mean={np.mean(ents):.3f})")
    print()
