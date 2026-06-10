"""Batched greedy decoding with the fixed 4-shot prompt (M0).

PROMPT CONSTANTS ARE FROZEN (CLAUDE.md): the 4 shots are the first four of the
standard gsm8k_cot 8-shot set; never edit them. All tracks share this template.
"""

from __future__ import annotations

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

FOUR_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove "
        "today. After they are done, there will be 21 trees. How many trees did the "
        "grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were "
        "planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
        "are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
        "do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
        "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
)

PROMPT_PREFIX = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FOUR_SHOTS)
STOP_STRING = "\nQ:"


def build_prompt(question: str) -> str:
    return f"{PROMPT_PREFIX}Q: {question}\nA:"


class _StopOnQ(StoppingCriteria):
    """Stop when every sequence in the batch has emitted the stop string."""

    def __init__(self, tokenizer, prompt_lens: list[int]):
        self.tok = tokenizer
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids, scores, **kw) -> bool:
        for i in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[i, self.prompt_lens[i]:], skip_special_tokens=True)
            if STOP_STRING not in text:
                return False
        return True


def truncate_at_stop(text: str) -> str:
    return text.split(STOP_STRING, 1)[0] if STOP_STRING in text else text


@torch.no_grad()
def generate_batch(model, tokenizer, questions: list[str],
                   batch_size: int = 32, max_new_tokens: int = 512) -> list[str]:
    """Greedy completions (truncated at the next 'Q:'), order-preserving."""
    tokenizer.padding_side = "left"
    outs: list[str] = []
    for start in range(0, len(questions), batch_size):
        chunk = [build_prompt(q) for q in questions[start:start + batch_size]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        stopper = _StopOnQ(tokenizer, [n_prompt] * len(chunk))
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        for row in gen:
            text = tokenizer.decode(row[n_prompt:], skip_special_tokens=True)
            outs.append(truncate_at_stop(text))
    return outs
