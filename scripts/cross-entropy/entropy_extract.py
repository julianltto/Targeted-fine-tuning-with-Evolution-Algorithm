"""Logit-distribution / entropy at the ANSWER token, split by correct vs wrong.
Teacher-force (prompt + cached generation); at the position predicting the first token
of the final numeric answer, compute full-vocab entropy H, the chosen-token logprob, and
the top1-top2 probability margin. Tests the 'confident error' hypothesis: is the model as
peaked (low entropy) on wrong answers as on right ones?
Run: PYTHONPATH=. .venv/bin/python scripts/evoroute/entropy_extract.py [n_adapters] [n_q]
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

NUM = re.compile(r"-?[\d,]*\.?\d+")
NA = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BATCH = 4
cfg = load_config("configs/full_lowmem.yaml")
members, docs, preds, gold, base_preds = dataio.load_population(cfg)
_, _, texts = load_texts(cfg)
sp = dataio.load_split()
pos = sp["ea_pos"][:NQ]
shots = load_8shot_samples()
prompts = {p: build_8shot_prompt(docs[p]["question"], shots) for p in pos}

model, tok = hmodel.load_base(cfg)
torch.set_grad_enabled(False)
dev = model.device
tok.padding_side = "right"

rows = []   # (entropy, logprob, margin, correct)
for si in range(NA):
    ind = load_lora_from_delta(REPO_ROOT / f"results/evomath/track4_sgd_ens/s0/sgd_ens_{si}.pt")
    h = hmodel.patch_lora(model, ind)
    try:
        for s in range(0, len(pos), BATCH):
            bp = pos[s:s + BATCH]
            fulls, t0s, tokids, corrects = [], [], [], []
            for p in bp:
                gen = texts[si][p]
                m = list(NUM.finditer(gen))
                if not m:
                    continue
                pr = prompts[p]
                full = pr + gen
                enc = tok(full, return_offsets_mapping=True, add_special_tokens=True)
                offs = enc["offset_mapping"]
                astart = len(pr) + m[-1].start()
                atoks = [t for t in range(len(offs)) if offs[t][1] > offs[t][0] and offs[t][0] >= astart and t >= 1]
                if not atoks:
                    continue
                t0 = min(atoks)
                fulls.append(enc["input_ids"]); t0s.append(t0)
                tokids.append(enc["input_ids"][t0])
                gv = round(float(gold[p]), 4)
                corrects.append(not np.isnan(preds[si, p]) and abs(preds[si, p] - gv) <= 1e-4)
            if not fulls:
                continue
            maxlen = max(len(f) for f in fulls)
            ids = torch.full((len(fulls), maxlen), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(fulls), maxlen), dtype=torch.long)
            for b, f in enumerate(fulls):
                ids[b, :len(f)] = torch.tensor(f); attn[b, :len(f)] = 1
            logits = model(input_ids=ids.to(dev), attention_mask=attn.to(dev)).logits
            for b in range(len(fulls)):
                row = logits[b, t0s[b] - 1, :].float()
                lp = F.log_softmax(row, -1)
                p_ = lp.exp()
                ent = float(-(p_ * lp).sum())
                top2 = torch.topk(p_, 2).values
                rows.append((ent, float(lp[tokids[b]]), float(top2[0] - top2[1]), corrects[b]))
    finally:
        hmodel.unpatch_lora(h)
    if si % 10 == 9:
        print(f"  {si+1}/{NA} adapters", flush=True)

R = np.array([(e, l, m) for e, l, m, c in rows])
C = np.array([c for *_, c in rows])
np.savez("results/evomath/evoroute/entropy.npz", feats=R, correct=C)
print(f"\n[answer-token distribution] n={len(rows)}  correct={C.sum()} ({100*C.mean():.1f}%)")
print(f"{'metric':10}{'CORRECT':>18}{'WRONG':>18}{'gap':>8}")
for j, name in enumerate(["entropy", "logprob", "margin"]):
    c, w = R[C, j], R[~C, j]
    print(f"{name:10}{c.mean():8.3f}±{c.std():.3f}  {w.mean():8.3f}±{w.std():.3f}  {c.mean()-w.mean():+7.3f}")
    # percentiles
    print(f"           correct p10/50/90: {np.percentile(c,10):.2f}/{np.percentile(c,50):.2f}/{np.percentile(c,90):.2f}"
          f"   wrong: {np.percentile(w,10):.2f}/{np.percentile(w,50):.2f}/{np.percentile(w,90):.2f}")
from sklearn.metrics import roc_auc_score
for j, name in enumerate(["entropy", "logprob", "margin"]):
    auc = roc_auc_score(C, R[:, j] if name != "entropy" else -R[:, j])   # higher logprob/margin, lower entropy => correct
    print(f"AUROC({name} predicts correct) = {auc:.3f}")
