"""Arithmetic-verification feature (design figure Section 6.2), zero-GPU.

Parse equations "a op b = c" from each cached CoT, check each step, and score a
generation by (valid steps / total steps). A candidate answer's f_arith is the mean
over its supporters. Unlike vote-derived features this is CONTENT-based, so it may rank
a correct minority answer #1 on hard cases where vote features cannot.
"""
from __future__ import annotations

import re

import numpy as np

from evomath.harness.parse import parse_gold
from scripts.optimize_track4_ensemble import load_bundle
from .dataio import TEST_DIRS, TOL, candidates_for, _seed_of

_EQ = re.compile(r"(-?[\d,]*\.?\d+)\s*([+\-*x×/÷])\s*(-?[\d,]*\.?\d+)\s*=\s*(-?[\d,]*\.?\d+)")


def _num(s):
    return float(s.replace(",", ""))


def arith_score(text: str):
    """valid_steps / total_steps for a generation; np.nan if no equation found."""
    steps = _EQ.findall(text)
    if not steps:
        return np.nan
    ok = 0
    for a, op, b, c in steps:
        try:
            x, y, z = _num(a), _num(b), _num(c)
        except ValueError:
            continue
        if op in "*x×":
            r = x * y
        elif op in "/÷":
            r = x / y if y != 0 else None
        elif op == "+":
            r = x + y
        else:
            r = x - y
        if r is not None and abs(r - z) <= 1e-4:
            ok += 1
    return ok / len(steps)


def load_texts(cfg):
    """texts[M][N] aligned to load_population member/position order."""
    docs = None
    base_txt = None
    member_txt = {}
    for root in TEST_DIRS:
        outs, _, _, dcs, _ = load_bundle(cfg, [root])
        if docs is None:
            docs = dcs
        for k, v in outs.items():
            if k == "base":
                base_txt = v if base_txt is None else base_txt
            else:
                member_txt[k] = v
    members = sorted(member_txt, key=_seed_of)
    texts = [member_txt[k] for k in members]
    return members, docs, texts


def arith_matrix(cfg):
    """[M,N] arithmetic-validity score (nan where no equation)."""
    members, docs, texts = load_texts(cfg)
    M, N = len(texts), len(docs)
    A = np.full((M, N), np.nan)
    for i in range(M):
        for q in range(N):
            A[i, q] = arith_score(texts[i][q])
    return members, docs, A


def candidate_arith(preds_col, arith_col):
    """Per candidate value -> mean arith score over its supporters (nan-safe)."""
    cands = candidates_for(preds_col)
    out = {}
    for v, mask in cands.items():
        vals = arith_col[mask]
        vals = vals[~np.isnan(vals)]
        out[v] = float(vals.mean()) if len(vals) else 0.5
    return out
