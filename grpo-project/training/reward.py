"""Multiplicative-format-gate × payoff reward.

Authority: PRD v6.1 §4.1 (locked) + Implementation Map §2.3.

What this replaces. The pre-Stage-1 reward was
`R = R_payoff + α·format_bonus + reasoning_length_bonus` — a model could
maximize the format-bonus term by emitting well-formed but empty reasoning,
producing a tractable reward-hack target under GRPO's uniform per-token
credit assignment.

What this is. A binary gate on parse success, multiplied into the payoff.

    format_gate(o) = 1  if o matches /<reasoning>.*?</reasoning>\\s*<action>(C|D)</action>/s
                  = 0  otherwise
    R_i = format_gate(o_i) · R_payoff(a_i, opp_action)

Risk (PRD §4.1, R1). Early in training, format-violation rate may exceed 30%
for the first ~20 steps. The `FormatWarmupCallback` watches for this and
halts with an instruction to run a brief SFT format-only warmup. The reward
itself does not adapt — sparsity is the user's signal, not a hidden recovery
mechanism.

What this does NOT do. It does not detect filler reasoning. The text inside
`<reasoning>` can be empty whitespace and still pass. That is acknowledged
out-of-scope in PRD §4.1; RFEval (post-training) is the measurement response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, NamedTuple

from configs.config import PayoffMatrix


# ---------------------------------------------------------------------------
# Regex — single-source-of-truth so RFEval and reward agree on what "valid" means
# ---------------------------------------------------------------------------

# DOTALL: '.' matches newlines. The pattern requires the action tag *immediately*
# after </reasoning> (modulo whitespace) so trailing junk doesn't pass.
FORMAT_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<action>([CD])</action>",
    re.DOTALL,
)


class RewardDebugInfo(NamedTuple):
    format_ok: bool
    action: str | None
    reasoning_len_chars: int


@dataclass(frozen=True)
class RewardOutput:
    """Return type of `compute_reward`. Logged as part of step diagnostics."""
    reward: float
    debug: RewardDebugInfo


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def parse_completion(completion: str) -> tuple[bool, str | None, int]:
    """Return (format_ok, action, reasoning_len_chars).

    `action` is `None` iff `format_ok` is False.
    """
    if not isinstance(completion, str):
        return False, None, 0
    m = FORMAT_RE.search(completion)
    if m is None:
        return False, None, 0
    reasoning, action = m.group(1), m.group(2)
    return True, action, len(reasoning)


def compute_reward(
    completion: str,
    opp_action: Literal["C", "D"],
    payoffs: PayoffMatrix,
) -> RewardOutput:
    """Compute the gated payoff for a single completion.

    Args:
        completion: The model's full output string.
        opp_action: The opponent's action this round, in {"C", "D"}.
        payoffs: T/R/P/S matrix for the current run.

    Returns:
        RewardOutput with `reward` ∈ {0} ∪ {S, P, R, T} (per the matrix).
    """
    if opp_action not in ("C", "D"):
        raise ValueError(f"opp_action must be 'C' or 'D', got {opp_action!r}")

    format_ok, action, reasoning_len = parse_completion(completion)
    debug = RewardDebugInfo(
        format_ok=format_ok,
        action=action,
        reasoning_len_chars=reasoning_len,
    )

    if not format_ok:
        return RewardOutput(reward=0.0, debug=debug)

    return RewardOutput(
        reward=float(payoffs.lookup(action, opp_action)),
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Vectorized helper for TRL's per-batch reward signature
# ---------------------------------------------------------------------------

def batch_compute_rewards(
    completions: list[str],
    opp_actions: list[str],
    payoffs: PayoffMatrix,
) -> tuple[list[float], list[RewardDebugInfo]]:
    """Apply `compute_reward` element-wise. Lengths must match."""
    if len(completions) != len(opp_actions):
        raise ValueError(
            f"len(completions)={len(completions)} != "
            f"len(opp_actions)={len(opp_actions)}"
        )
    rewards: list[float] = []
    debugs: list[RewardDebugInfo] = []
    for c, o in zip(completions, opp_actions):
        out = compute_reward(c, o, payoffs)
        rewards.append(out.reward)
        debugs.append(out.debug)
    return rewards, debugs


__all__ = [
    "FORMAT_RE",
    "parse_completion",
    "compute_reward",
    "batch_compute_rewards",
    "RewardOutput",
    "RewardDebugInfo",
]
