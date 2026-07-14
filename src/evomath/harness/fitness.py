"""Two-tier fitness (M0). Model weights frozen, no_grad throughout.

- fitness_loglik: mean per-token log-likelihood of the gold CoT solution
  (teacher-forced, one forward per question, batched). Cheap, continuous,
  low-noise -> inner loop of every EA.
- fitness_acc: greedy generation + answer parsing. Expensive -> elites only.
  Raises if parse-failure rate > 15% (almost always a template/parser bug,
  CLAUDE.md constraint 8).
"""

from __future__ import annotations

import torch

from evomath.harness.generate import build_prompt, generate_batch
from evomath.harness.parse import is_correct, parse_pred

PARSE_FAIL_LIMIT = 0.15


@torch.no_grad()
def fitness_acc(model, tokenizer, docs: list[dict],
                batch_size: int = 32, max_new_tokens: int = 512,
                return_outputs: bool = False):
    outs = generate_batch(model, tokenizer, [d["question"] for d in docs],
                          batch_size=batch_size, max_new_tokens=max_new_tokens)
    n_fail = sum(1 for o in outs if parse_pred(o) is None)
    if n_fail / len(outs) > PARSE_FAIL_LIMIT:
        raise RuntimeError(
            f"answer parse failure rate {n_fail}/{len(outs)} exceeds "
            f"{PARSE_FAIL_LIMIT:.0%} — check prompt template / parser before "
            "trusting any fitness value"
        )
    correct = [is_correct(o, d["gold_text"]) for o, d in zip(outs, docs)]
    acc = sum(correct) / len(correct)
    if return_outputs:
        return acc, outs, correct
    return acc


def _gather_token_loglik(logits: torch.Tensor, targets: torch.Tensor,
                         seq_chunk: int = 128) -> torch.Tensor:
    """Per-token loglik of `targets` under `logits` (already shifted), fp32
    log-softmax computed in sequence chunks — never materializes the full-vocab
    fp32 tensor (which is ~8GB for bs16 x 1k tokens x 128k vocab)."""
    parts = []
    for s in range(0, logits.shape[1], seq_chunk):
        lp = torch.log_softmax(logits[:, s:s + seq_chunk].float(), dim=-1)
        parts.append(lp.gather(-1, targets[:, s:s + seq_chunk].unsqueeze(-1)).squeeze(-1))
        del lp
    return torch.cat(parts, dim=1)


@torch.no_grad()
def fitness_loglik(model, tokenizer, docs: list[dict], batch_size: int = 16) -> float:
    """Mean over questions of (mean per-token loglik of the gold solution)."""
    tokenizer.padding_side = "right"
    per_doc: list[float] = []
    for start in range(0, len(docs), batch_size):
        chunk = docs[start:start + batch_size]
        prompts = [build_prompt(d["question"]) for d in chunk]
        fulls = [p + " " + d["solution"] for p, d in zip(prompts, chunk)]
        enc = tokenizer(fulls, return_tensors="pt", padding=True).to(model.device)
        prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]

        logits = model(**enc).logits
        token_ll = _gather_token_loglik(logits[:, :-1], enc["input_ids"][:, 1:])

        attn = enc["attention_mask"][:, 1:]
        for i, plen in enumerate(prompt_lens):
            # answer tokens = positions >= plen-1 in the shifted frame, unpadded
            mask = attn[i].clone()
            mask[: plen - 1] = 0
            n = int(mask.sum())
            per_doc.append(float((token_ll[i] * mask).sum() / max(n, 1)))
    return sum(per_doc) / len(per_doc)
