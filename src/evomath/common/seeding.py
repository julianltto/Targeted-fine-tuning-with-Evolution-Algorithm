from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed all RNGs; returns a dedicated torch.Generator for EA sampling so
    library-internal RNG consumption can't shift the evolution stream."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g
