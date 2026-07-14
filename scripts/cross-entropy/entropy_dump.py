"""Dump per-token entropy JSON for a correct vs wrong CoT (idx 354) for visualization."""
import sys, re, json
sys.path.insert(0, "src")
import torch, torch.nn.functional as F
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
shots = load_8shot_samples()


def trace(seed, pos):
    ind = load_lora_from_delta(REPO_ROOT / f"results/evomath/track4_sgd_ens/s0/sgd_ens_{seed}.pt")
    h = hmodel.patch_lora(model, ind)
    try:
        raw = texts[seed][pos]
        gen = raw.split("The answer is")[0] + ("The answer is" + raw.split("The answer is")[1].split(".")[0] + "." if "The answer is" in raw else "")
        prompt = build_8shot_prompt(docs[pos]["question"], shots)
        penc = tok(prompt, add_special_tokens=True)["input_ids"]
        full = tok(prompt + gen, add_special_tokens=True)["input_ids"]
        logits = model(input_ids=torch.tensor([full]).to(model.device)).logits[0].float()
        lp = F.log_softmax(logits, -1)
        ent = -(lp.exp() * lp).sum(-1)
        return [[tok.decode([full[t + 1]]), round(float(ent[t]), 3)] for t in range(len(penc) - 1, len(full) - 1)]
    finally:
        hmodel.unpatch_lora(h)


out = {"question": docs[354]["question"], "gold": 45,
       "wrong": {"answer": 125, "tokens": trace(0, 354)},
       "correct": {"answer": 45, "tokens": trace(1, 354)}}
print("JSON_START")
print(json.dumps(out))
print("JSON_END")
