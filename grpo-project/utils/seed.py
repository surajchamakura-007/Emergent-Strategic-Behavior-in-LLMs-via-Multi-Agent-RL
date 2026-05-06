"""Deterministic seeding for reproducibility.

Authority: Implementation Map §2.1.

Sets every RNG that affects training: Python `random`, NumPy, PyTorch CPU/CUDA,
HuggingFace `transformers.set_seed`, and `PYTHONHASHSEED`. vLLM consumes the
same seed via `LLM(seed=...)` — call `seed_all` BEFORE constructing vLLM.

Determinism caveats. CUDA matmul kernels are non-deterministic by default;
we do not force `torch.use_deterministic_algorithms(True)` because vLLM's
attention kernels do not support it. Stage 1 reproducibility is therefore
"same-seed runs are statistically indistinguishable", not bit-identical.
"""

from __future__ import annotations

import os
import random


def seed_all(seed: int) -> int:
    """Seed every RNG that influences training. Returns `seed` for logging."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be non-negative int, got {seed!r}")

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # HuggingFace transformers
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass

    return seed


__all__ = ["seed_all"]
