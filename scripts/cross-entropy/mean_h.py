"""Cross-entropy (token-entropy) analysis of LoRA-ensemble CoTs, and mean-H weighted voting.

Single entry point. Replaces mean_h_probe / mean_h_vote / mean_h_confirm.

Data: D_select (330) tunes the one hyperparameter (temperature T); D_heldout (989 = the
former D_ea + D_final) is evaluated once. Nothing is fitted, so there is no train split.

  probe   correct-vs-wrong mean-H distributions + AUROC over every cached CoT
  vote    tune T on D_select, then majority vs mean-H-weighted voting on D_heldout

H[i, q] = mean per-token entropy of adapter i's cached CoT on question q, teacher-forced.
Cached to mean_h_all.npz [50, 1319]; after the first build both commands are GPU-free.

Run: PYTHONPATH=. .venv/bin/python scripts/evoroute/mean_h.py {probe,vote}
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
from evomath import load_config, REPO_ROOT
from evomath.evoroute import dataio

NA, TOL = 50, 1e-4
BATCH = 8
CACHE = "results/evomath/evoroute/mean_h_all.npz"
LEGACY = ["results/evomath/evoroute/mean_h_matrix.npz",      # select + final
          "results/evomath/evoroute/mean_h_matrix_ea.npz"]   # ea
T_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

cfg = load_config("configs/full_lowmem.yaml")
members, docs, preds, gold, base_preds = dataio.load_population(cfg)
sp = dataio.load_split()
N = preds.shape[1]
SELECT = list(sp["select_pos"])
HELDOUT = list(sp["heldout_pos"])
GV = np.array([round(float(gold[p]), 4) for p in range(N)])
CORRECT = np.array([[not np.isnan(preds[i, p]) and abs(round(float(preds[i, p]), 4) - GV[p]) <= TOL
                     for p in range(N)] for i in range(NA)])


def _gen_text(raw):
    if "The answer is" in raw:
        return raw.split("The answer is")[0] + "The answer is" + raw.split("The answer is")[1].split(".")[0] + "."
    return raw


def _compute(H, todo):
    """Fill H[:, q] for q in todo by teacher-forcing each adapter's cached CoT."""
    import torch, torch.nn.functional as F
    from evomath.harness import model as hmodel
    from evomath.evoroute.arith import load_texts
    from scripts.eval_track4_archive import build_8shot_prompt, load_8shot_samples, load_lora_from_delta
    _, _, texts = load_texts(cfg)
    model, tok = hmodel.load_base(cfg)
    torch.set_grad_enabled(False)
    tok.padding_side = "right"
    shots = load_8shot_samples()
    prompts = {p: build_8shot_prompt(docs[p]["question"], shots) for p in todo}
    plen = {p: len(tok(prompts[p], add_special_tokens=True)["input_ids"]) for p in todo}
    for si in range(NA):
        ind = load_lora_from_delta(REPO_ROOT / ("results/evomath/track4_sgd_ens/s0/sgd_ens_%d.pt" % si))
        h = hmodel.patch_lora(model, ind)
        try:
            items = []
            for p in todo:
                full = tok(prompts[p] + _gen_text(texts[si][p]), add_special_tokens=True)["input_ids"]
                if len(full) - plen[p] >= 3:
                    items.append((p, plen[p], full))
            items.sort(key=lambda x: len(x[2]))
            for s in range(0, len(items), BATCH):
                bt = items[s:s + BATCH]
                ml = max(len(f) for _, _, f in bt)
                ids = torch.full((len(bt), ml), tok.pad_token_id, dtype=torch.long)
                am = torch.zeros((len(bt), ml), dtype=torch.long)
                for b, (_, _, f) in enumerate(bt):
                    ids[b, :len(f)] = torch.tensor(f)
                    am[b, :len(f)] = 1
                # hidden states only: the full [B, L, 128k] logits OOM (8-shot prompts ~1200 tok)
                hs = model.model(input_ids=ids.to(model.device),
                                 attention_mask=am.to(model.device)).last_hidden_state
                for b, (p, pl, f) in enumerate(bt):
                    rows = model.lm_head(hs[b, pl - 1:len(f) - 1, :]).float()
                    lp = F.log_softmax(rows, -1)
                    H[si, p] = float((-(lp.exp() * lp).sum(-1)).mean())
                    del rows, lp
                del hs
        finally:
            hmodel.unpatch_lora(h)
        print("  adapter %d/%d" % (si + 1, NA), flush=True)
    return H


def get_H(need):
    """H[50, 1319] mean CoT entropy. Builds/extends the cache; migrates legacy matrices."""
    H = np.load(CACHE)["H"] if os.path.exists(CACHE) else np.full((NA, N), np.nan, np.float32)
    if not os.path.exists(CACHE):
        for lg in LEGACY:                       # reuse already-computed columns
            if os.path.exists(lg):
                d = np.load(lg)
                for j, p in enumerate(list(d["pos"])):
                    H[:, int(p)] = d["H"][:, j]
    todo = [p for p in need if np.isnan(H[:, p]).all()]
    if todo:
        print("computing H for %d questions x %d adapters (GPU)..." % (len(todo), NA), flush=True)
        H = _compute(H, todo)
    np.savez(CACHE, H=H)
    return H


def fill(H):
    return np.where(np.isnan(H), float(np.nanmedian(H)), H)


# ---------------------------------------------------------------- probe
def probe():
    H = fill(get_H(list(range(N))))
    c, w = H[CORRECT], H[~CORRECT]
    print("\n=== mean CoT entropy H: correct vs wrong (all %d cached CoTs) ===" % H.size)
    for name, x in [("CORRECT", c), ("WRONG", w)]:
        print("  %-8s n=%-6d mean=%.3f  std=%.3f  p10/p50/p90 = %.3f / %.3f / %.3f"
              % (name, len(x), x.mean(), x.std(),
                 np.percentile(x, 10), np.percentile(x, 50), np.percentile(x, 90)))
    print("  gap (wrong - correct) = %+.3f" % (w.mean() - c.mean()))
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(c)), np.ones(len(w))]
    print("  AUROC(mean H predicts WRONG) = %.3f" % roc_auc_score(y, np.r_[c, w]))


# ---------------------------------------------------------------- vote
def _vote(p, w):
    t = {}
    for s in range(NA):
        v = preds[s, p]
        if np.isnan(v):
            continue
        k = round(float(v), 4)
        t[k] = t.get(k, 0.0) + float(w[s])
    return max(t.items(), key=lambda kv: (kv[1], -kv[0]))[0] if t else None


def _hits(POS, wfn, H):
    return np.array([1 if (lambda r: r is not None and abs(r - GV[p]) <= TOL)(_vote(p, wfn(H[:, p]))) else 0
                     for p in POS])


def vote():
    H = fill(get_H(SELECT + HELDOUT))
    ones = lambda h: np.ones(NA)

    best = max(((T, _hits(SELECT, (lambda t: lambda h: np.exp(-h / t))(T), H).mean()) for T in T_GRID),
               key=lambda kv: kv[1])
    T = best[0]
    print("\n=== mean-H weighted voting ===")
    print("  tuned on D_select (%d q): T* = %.2f  (select acc %.2f)" % (len(SELECT), T, 100 * best[1]))
    print("  evaluated once on D_heldout (%d q = former D_ea + D_final)\n" % len(HELDOUT))

    schemes = [("majority (w=1)", ones),
               ("exp(-H/T*=%.2f)" % T, lambda h: np.exp(-h / T)),
               ("inv 1/(H+0.05)", lambda h: 1.0 / (h + 0.05)),
               ("linear (Hmax-H)", lambda h: H.max() - h),
               ("drop top-decile H", lambda h: (h <= np.quantile(h, 0.9)).astype(float))]
    maj = _hits(HELDOUT, ones, H)
    print("%-24s%10s" % ("scheme", "HELDOUT"))
    for name, fn in schemes:
        print("%-24s%10.2f" % (name, 100 * _hits(HELDOUT, fn, H).mean()))
    orc = np.mean([CORRECT[:, p].any() for p in HELDOUT])
    print("%-24s%10.2f" % ("ORACLE", 100 * orc))

    wt = _hits(HELDOUT, lambda h: np.exp(-h / T), H)
    b = int(((maj == 0) & (wt == 1)).sum())
    c = int(((maj == 1) & (wt == 0)).sum())
    from scipy.stats import binomtest
    n = len(HELDOUT)
    rng = np.random.default_rng(0)
    dif = wt - maj
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(10000)])
    print("\n  exp(-H/T*) vs majority:  %+.2fpp" % (100 * (wt.mean() - maj.mean())))
    print("  McNemar: rescued=%d  broken=%d   exact p = %.4g" % (b, c, binomtest(b, b + c, 0.5).pvalue))
    print("  paired bootstrap 95%% CI: [%+.2f, %+.2f] pp   P(>0) = %.4f"
          % (100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), (bs > 0).mean()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "vote"
    {"probe": probe, "vote": vote}[cmd]()
