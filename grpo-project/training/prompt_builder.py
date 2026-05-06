"""Prompt assembly for the IPD agent.

Authority: Implementation Map §2.3 + PRD v6.1 §6.1.

Constraints.
    - Prompt MUST mention p=0.95 continuation (PRD §6.1).
    - Prompt MUST NOT mention the 60-round cap (Map §2.3, PRD §6.1).
    - History block respects the 400-token budget (Map §4) via
      `serialize_history`.
    - Format spec in the prompt must match `training.reward.FORMAT_RE` exactly
      so that reward and prompt agree on what "valid" means.

Token budget (Map §4.1).
    system + payoff matrix      ≈  150 tokens
    history (truncated)         ≤  400 tokens
    round header                ≈   10 tokens
    generation budget           ≈  400 tokens
    cushion                     ≈   64 tokens
    total                       ≤ 1024 (max_seq_length)
"""

from __future__ import annotations

from typing import Any

from configs.config import PayoffMatrix
from envs.history import (
    Round,
    TruncationResult,
    serialize_history,
    write_truncation_event,
)


# ---------------------------------------------------------------------------
# Static text — the system block is identical across all rounds of all episodes
# ---------------------------------------------------------------------------

# Note on T value: the payoff matrix is shown EXPLICITLY each round. The agent
# does not need to memorize numbers; it just reads them.
_SYSTEM_TEMPLATE = """You are playing the Iterated Prisoner's Dilemma.

Payoff matrix (your payoff, opponent's payoff):
  Both cooperate (C,C): {R:g}, {R:g}
  Both defect    (D,D): {P:g}, {P:g}
  You D, they C  (D,C): {T:g}, {S:g}
  You C, they D  (C,D): {S:g}, {T:g}

After each round, the game continues with probability {p:g}; otherwise it ends.
The exact number of rounds is random.
"""

# Format spec — must match training.reward.FORMAT_RE
_FORMAT_SPEC = """Your response MUST follow this exact format:
<reasoning>
[your chain-of-thought reasoning here]
</reasoning>
<action>[C or D]</action>"""


def _build_system_block(payoffs: PayoffMatrix, p: float) -> str:
    return _SYSTEM_TEMPLATE.format(
        R=payoffs.R, P=payoffs.P, T=payoffs.T, S=payoffs.S, p=p,
    )


def build_prompt(
    history: list[Round],
    round_idx: int,
    payoffs: PayoffMatrix,
    p: float,
    tokenizer: Any | None = None,
    *,
    history_token_budget: int = 400,
    keep_last_k: int = 25,
    step: int = 0,
    episode_id: str = "",
    wandb_run=None,
) -> tuple[str, TruncationResult]:
    """Assemble a single-round prompt.

    Args:
        history: Completed rounds in chronological order (may be empty).
        round_idx: 1-indexed round number for the *next* (not-yet-played) round.
        payoffs: T/R/P/S matrix.
        p: Continuation probability (printed in the system block).
        tokenizer: Optional HF tokenizer for accurate budget enforcement.
        history_token_budget / keep_last_k: passed through to serialize_history.
        step / episode_id / wandb_run: optional, for truncation event logging.

    Returns:
        (prompt_string, truncation_result). The truncation_result is the same
        object that was logged (caller can use it for additional bookkeeping).
    """
    system_block = _build_system_block(payoffs, p)

    truncation = serialize_history(
        history,
        tokenizer=tokenizer,
        budget=history_token_budget,
        keep_last_k=keep_last_k,
    )
    history_block = truncation.serialized

    round_header = f"It is now Round {round_idx}. What is your action?"

    prompt = (
        f"{system_block}\n"
        f"Round history:\n{history_block}\n\n"
        f"{round_header}\n\n"
        f"{_FORMAT_SPEC}\n"
    )

    # Side-effect: log truncation events
    if truncation.summary_used:
        write_truncation_event(truncation, step=step, episode_id=episode_id, wandb_run=wandb_run)

    return prompt, truncation


__all__ = ["build_prompt"]
