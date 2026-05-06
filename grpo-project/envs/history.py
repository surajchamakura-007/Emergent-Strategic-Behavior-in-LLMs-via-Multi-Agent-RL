"""Round serialization + sliding-window history truncation.

Authority: Implementation Map §2.2 + §4 (full policy spec).

Problem. With max_seq_length=1024 and episode cap=60, a long episode's
history can blow the prompt budget (~12-15 tok/round × 60 = 720-900 tok,
leaving no room for the system prompt + generation budget).

Policy (Map §4.2). Two tiers, evaluated in order:
    Tier 1 (rounds 1-27): emit verbatim.
    Tier 2 (rounds 28+): summary line + last K verbatim rounds.

The summary line preserves "opponent's last action" so a TfT-style strategy
can still execute after truncation (Map §4.3). If summary + K rounds doesn't
fit, K is decremented by 5 until it does (or K=0, pure-summary fallback).

Token budget (Map §4.1). 400 tokens for history; the rest of the 1024-token
sequence is system prompt (~150) + round header (~10) + generation (~400) +
64-token cushion against tokenizer drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Round + serialization output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Round:
    """A single completed PD round, from the perspective of the agent."""
    my_action: Literal["C", "D"]
    opp_action: Literal["C", "D"]
    my_payoff: float

    def __post_init__(self) -> None:
        if self.my_action not in ("C", "D"):
            raise ValueError(f"my_action {self.my_action!r} not in {{C,D}}")
        if self.opp_action not in ("C", "D"):
            raise ValueError(f"opp_action {self.opp_action!r} not in {{C,D}}")


@dataclass(frozen=True)
class TruncationResult:
    """Output of `serialize_history`. Used for prompt assembly and W&B logging."""
    serialized: str
    raw_round_count: int
    kept_verbatim: int
    summary_used: bool
    estimated_tokens: int


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _format_round(idx: int, r: Round) -> str:
    """Single-line round encoding. ~12-15 tokens per line."""
    return f"  Round {idx}: You={r.my_action}, Opp={r.opp_action} (you got {r.my_payoff:g})"


def _format_summary(rounds: list[Round]) -> str:
    """Compact summary of `rounds` preserving TfT-relevant signal.

    Per Map §4.2 — must include opp's *last* action so a TfT-style strategy
    survives truncation.
    """
    n = len(rounds)
    coop_self = sum(1 for r in rounds if r.my_action == "C")
    coop_opp = sum(1 for r in rounds if r.opp_action == "C")
    total = sum(r.my_payoff for r in rounds)
    last_opp = rounds[-1].opp_action
    return (
        f"  [Earlier {n} rounds — you played C {coop_self}/{n}, "
        f"opponent C {coop_opp}/{n}, your total payoff {total:.1f}, "
        f"opponent's last action: {last_opp}]"
    )


def _estimate_tokens(text: str, tokenizer) -> int:
    """Cheap token-count estimate. Uses tokenizer if provided, else heuristic."""
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Heuristic fallback: ~4 chars/token in English text.
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serialize_history(
    history: list[Round],
    tokenizer=None,
    budget: int = 400,
    keep_last_k: int = 25,
) -> TruncationResult:
    """Serialize `history` for prompt insertion under a token budget.

    Algorithm (Map §4.2).
        1. If `history` is empty → return "  (no rounds yet)".
        2. Try verbatim. If it fits within `budget`, return it.
        3. Else: try summary + last K rounds (K = keep_last_k).
        4. If still over budget, decrement K by 5 and retry; floor K at 0
           (pure-summary fallback).

    Args:
        history: List of completed rounds, in chronological order.
        tokenizer: HuggingFace tokenizer (or None for heuristic estimation).
        budget: Token cap for the serialized output.
        keep_last_k: Initial verbatim window size in summary mode.

    Returns:
        TruncationResult — never raises.
    """
    n = len(history)
    if n == 0:
        text = "  (no rounds yet)"
        return TruncationResult(
            serialized=text,
            raw_round_count=0,
            kept_verbatim=0,
            summary_used=False,
            estimated_tokens=_estimate_tokens(text, tokenizer),
        )

    # --- Tier 1: verbatim ---
    verbatim_lines = [_format_round(i + 1, r) for i, r in enumerate(history)]
    verbatim = "\n".join(verbatim_lines)
    verbatim_tokens = _estimate_tokens(verbatim, tokenizer)
    if verbatim_tokens <= budget:
        return TruncationResult(
            serialized=verbatim,
            raw_round_count=n,
            kept_verbatim=n,
            summary_used=False,
            estimated_tokens=verbatim_tokens,
        )

    # --- Tier 2: summary + last K, with K decay ---
    K = max(0, min(keep_last_k, n))  # can't keep more than we have
    while True:
        old_rounds = history[: n - K] if K < n else []
        new_rounds = history[n - K :] if K > 0 else []

        if old_rounds:
            summary_line = _format_summary(old_rounds)
        else:
            # K == n means we're keeping everything verbatim, but we already
            # know that didn't fit in Tier 1 — should not happen; defensive.
            summary_line = ""

        new_lines = [
            _format_round(n - K + i + 1, r) for i, r in enumerate(new_rounds)
        ]
        parts = [p for p in ([summary_line] if summary_line else []) + new_lines if p]
        text = "\n".join(parts) if parts else "  (no rounds yet)"
        tokens = _estimate_tokens(text, tokenizer)

        if tokens <= budget:
            return TruncationResult(
                serialized=text,
                raw_round_count=n,
                kept_verbatim=K,
                summary_used=bool(old_rounds),
                estimated_tokens=tokens,
            )

        if K == 0:
            # Pure-summary fallback over all rounds — always fits in practice
            # (the summary line is ~30 tokens). Defensive return.
            text = _format_summary(history)
            return TruncationResult(
                serialized=text,
                raw_round_count=n,
                kept_verbatim=0,
                summary_used=True,
                estimated_tokens=_estimate_tokens(text, tokenizer),
            )

        K = max(0, K - 5)


def write_truncation_event(
    result: TruncationResult,
    step: int,
    episode_id: str,
    wandb_run=None,
) -> None:
    """Append a metadata-only truncation event to W&B.

    Per Map §2.2 — only metadata, never the serialized content (would blow up
    log size). Called from `prompt_builder.build_prompt` whenever
    `result.summary_used` is True.
    """
    if wandb_run is None or not result.summary_used:
        return
    try:
        wandb_run.log(
            {
                "truncation/step": step,
                "truncation/episode_id_hash": hash(episode_id) % (1 << 31),
                "truncation/raw_round_count": result.raw_round_count,
                "truncation/kept_verbatim": result.kept_verbatim,
                "truncation/estimated_tokens": result.estimated_tokens,
            },
            step=step,
        )
    except Exception:
        # Logging must never crash the rollout
        pass


__all__ = [
    "Round",
    "TruncationResult",
    "serialize_history",
    "write_truncation_event",
]
