"""M0 acceptance (CLAUDE.md):

1. base 4-shot greedy accuracy on dev-500 in [25, 42]%
2. same seed twice -> per-question outputs identical
3. Spearman rho between loglik proxy and accuracy over base + 8 random LoRA
   individuals; report to W&B (expect > 0.5; if not, STOP and report — do not
   swap the proxy)
4. fitness cache: second evaluation of the same individual < 1s

--smoke-test: 10 questions, 2 individuals, no W&B, < 3 min.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from evomath import load_config  # noqa: E402
from evomath.common.seeding import seed_everything  # noqa: E402
from evomath.harness import data as hdata  # noqa: E402
from evomath.harness import model as hmodel  # noqa: E402
from evomath.harness.cache import FitnessCache, file_hash, individual_key  # noqa: E402
from evomath.harness.fitness import fitness_acc, fitness_loglik  # noqa: E402


def spearman(a: list[float], b: list[float]) -> float:
    import numpy as np
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--lora-sigma", type=float, default=0.02)
    p.add_argument("--offline", action="store_true", help="wandb offline mode")
    args = p.parse_args()

    cfg = load_config(args.config)
    gen = seed_everything(args.seed)
    docs = hdata.load_dev(cfg)
    n_ind = 2 if args.smoke_test else 8
    if args.smoke_test:
        docs = docs[:10]
    bs = cfg["gen_batch_size"]
    print(f"[m0] {len(docs)} dev questions, {n_ind} random LoRA individuals, bs={bs}")

    model, tok = hmodel.load_base(cfg)
    t0 = time.time()

    # -- 1. base accuracy in range ------------------------------------------
    acc_base, outs1, _ = fitness_acc(model, tok, docs, batch_size=bs,
                                     max_new_tokens=cfg["max_new_tokens"],
                                     return_outputs=True)
    print(f"[m0] base acc = {acc_base:.4f}  ({time.time()-t0:.0f}s)")
    in_range = 0.25 <= acc_base <= 0.42

    # -- 2. determinism: same seed twice, identical outputs ------------------
    seed_everything(args.seed)
    _, outs2, _ = fitness_acc(model, tok, docs, batch_size=bs,
                              max_new_tokens=cfg["max_new_tokens"], return_outputs=True)
    deterministic = outs1 == outs2
    print(f"[m0] determinism: {'PASS' if deterministic else 'FAIL'} "
          f"({sum(a != b for a, b in zip(outs1, outs2))} mismatches)")

    # -- 3. proxy correlation over base + N random LoRA individuals ----------
    ll_base = fitness_loglik(model, tok, docs, batch_size=cfg["loglik_batch_size"])
    accs, lls = [acc_base], [ll_base]
    # graded perturbation strengths -> a spread of degradations, so the rank
    # correlation has signal (identical sigmas can collapse to equal accs)
    sigmas = [args.lora_sigma * (k + 1) / n_ind for k in range(n_ind)]
    for k in range(n_ind):
        ind = hmodel.random_lora_individual(model, sigma=sigmas[k], generator=gen)
        handles = hmodel.patch_lora(model, ind)
        try:
            a = fitness_acc(model, tok, docs, batch_size=bs,
                            max_new_tokens=cfg["max_new_tokens"])
            l = fitness_loglik(model, tok, docs, batch_size=cfg["loglik_batch_size"])
        finally:
            hmodel.unpatch_lora(handles)
        accs.append(a)
        lls.append(l)
        print(f"[m0] ind {k+1}/{n_ind} (sigma={sigmas[k]:.4f}): acc={a:.4f} "
              f"loglik={l:.4f} ({time.time()-t0:.0f}s)")
    rho = spearman(lls, accs)
    print(f"[m0] Spearman rho(loglik, acc) over {len(accs)} points = {rho:.3f}")

    # -- 4. cache hit < 1s ----------------------------------------------------
    cache = FitnessCache(cfg["cache_path"])
    idx_hash = file_hash(cfg["eval_indices"])
    ind = hmodel.random_lora_individual(model, sigma=args.lora_sigma, generator=gen)
    key = individual_key(ind.named_tensors(), "loglik", idx_hash)
    handles = hmodel.patch_lora(model, ind)
    try:
        cache.put(key, fitness_loglik(model, tok, docs, batch_size=cfg["loglik_batch_size"]))
    finally:
        hmodel.unpatch_lora(handles)
    t1 = time.time()
    hit = cache.get(individual_key(ind.named_tensors(), "loglik", idx_hash))
    cache_seconds = time.time() - t1
    cache_ok = hit is not None and cache_seconds < 1.0
    print(f"[m0] cache second lookup: {cache_seconds*1000:.0f} ms -> {'PASS' if cache_ok else 'FAIL'}")

    summary = {
        "base_acc": acc_base, "acc_in_25_42": in_range,
        "deterministic": deterministic, "spearman_rho": rho,
        "rho_gt_0.5": rho > 0.5, "cache_lookup_s": cache_seconds,
        "n_individuals": n_ind, "n_questions": len(docs),
    }
    if not args.smoke_test:
        from evomath.common.wb import init_run
        run = init_run(cfg, "m0-acceptance", args.seed,
                       {"lora_sigma": args.lora_sigma}, offline=args.offline)
        run.log(summary)
        run.summary.update(summary)
        run.finish()

    print("[m0] summary:", summary)
    if args.smoke_test:
        print("SMOKE OK")
    else:
        verdict = in_range and deterministic and rho > 0.5 and cache_ok
        print("M0 ACCEPTANCE:", "PASS" if verdict else "FAIL — see items above")


if __name__ == "__main__":
    main()
