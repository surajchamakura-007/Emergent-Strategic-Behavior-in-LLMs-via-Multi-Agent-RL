"""Iterated Prisoner's Dilemma environment with stochastic horizon.

Authority: PRD v6.1 §6.1 (D5: p=0.95, D8 v6: cap=60) + Implementation Map §2.2.

Episode dynamics. After each completed round:
    done = True if    Bernoulli(0.05) fires    (geometric continuation)
                  OR  round_count == 60        (hard cap)

Probability mass beyond cap: P(T ≥ 60) = 0.95^59 ≈ 0.048 → ~5% truncation.
Effective E[T] under cap ≈ 19.1. Acceptance criterion (PRD §6.1) requires
1000-episode histogram to match (mean 19.1±0.5, std 14.5±0.5, trunc 4.8±0.5%).

Information available to the agent. The agent knows p=0.95 (the prompt says
so). It does NOT know the cap — the prompt template never mentions 60.

Public surface.
    PrisonersDilemmaEnv(payoffs, p, cap, rng)
        .reset() -> initial state
        .step(my_action, opp_action) -> (info, done)
        .history -> list[Round] (read-only view)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from configs.config import PayoffMatrix
from envs.history import Round


@dataclass
class StepInfo:
    """Per-step diagnostic info returned by `step()`."""
    round: int                    # 1-indexed
    my_payoff: float
    opp_payoff: float
    truncated_at_cap: bool
    geometric_done: bool


class PrisonersDilemmaEnv:
    """Stochastic-horizon IPD environment (PRD v6.1 §6.1)."""

    def __init__(
        self,
        payoffs: PayoffMatrix,
        p: float = 0.95,
        cap: int = 60,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0,1), got {p}")
        if cap < 2:
            raise ValueError(f"cap must be >= 2, got {cap}")
        self.payoffs = payoffs
        self.p = p
        self.cap = cap
        self.rng = rng if rng is not None else np.random.default_rng()
        self._history: list[Round] = []
        self._round = 0
        self._done = False

    # ---------- properties ----------

    @property
    def history(self) -> list[Round]:
        """Read-only view of completed rounds (most recent last)."""
        return list(self._history)

    @property
    def round(self) -> int:
        """1-indexed current round (0 before first step)."""
        return self._round

    @property
    def done(self) -> bool:
        return self._done

    # ---------- core API ----------

    def reset(self) -> None:
        """Begin a fresh episode."""
        self._history = []
        self._round = 0
        self._done = False

    def step(
        self,
        my_action: Literal["C", "D"],
        opp_action: Literal["C", "D"],
    ) -> StepInfo:
        """Apply both players' actions and advance one round.

        Args:
            my_action / opp_action: from the parsed completions of the two
                rollouts. Must be valid; the caller (rollout loop) is
                responsible for handling format-failure (reward=0, action=None)
                BEFORE calling step().

        Returns:
            StepInfo with payoffs and termination flags.
        """
        if self._done:
            raise RuntimeError("step() called after episode is done")
        if my_action not in ("C", "D") or opp_action not in ("C", "D"):
            raise ValueError(
                f"Bad actions in step(): my={my_action!r}, opp={opp_action!r}"
            )

        self._round += 1
        my_payoff = self.payoffs.lookup(my_action, opp_action)
        opp_payoff = self.payoffs.lookup(opp_action, my_action)

        self._history.append(
            Round(my_action=my_action, opp_action=opp_action, my_payoff=my_payoff)
        )

        # Termination: cap takes precedence in the truncated_at_cap flag.
        truncated_at_cap = self._round >= self.cap
        # Bernoulli is sampled on EVERY step (including the capped one).
        # We sample first to keep the RNG path identical regardless of cap.
        geometric_done = bool(self.rng.random() >= self.p)

        if truncated_at_cap or geometric_done:
            self._done = True

        return StepInfo(
            round=self._round,
            my_payoff=my_payoff,
            opp_payoff=opp_payoff,
            truncated_at_cap=truncated_at_cap,
            geometric_done=geometric_done,
        )


# ---------------------------------------------------------------------------
# Diagnostic helper used by preflight 07_episode_dist_smoke.py
# ---------------------------------------------------------------------------

def simulate_episode_lengths(
    n_episodes: int,
    p: float = 0.95,
    cap: int = 60,
    seed: int = 0,
) -> dict:
    """Simulate `n_episodes` of horizon-only dynamics (no agent actions).

    Returns a dict with mean / std / truncation_rate / per-episode lengths.
    Used by preflight to verify the env conforms to PRD §6.1 acceptance.
    """
    rng = np.random.default_rng(seed)
    lengths = np.empty(n_episodes, dtype=np.int64)
    n_truncated = 0
    for i in range(n_episodes):
        t = 0
        while True:
            t += 1
            if t >= cap:
                n_truncated += 1
                break
            if rng.random() >= p:
                break
        lengths[i] = t
    return {
        "n_episodes": n_episodes,
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "truncation_rate": n_truncated / n_episodes,
        "lengths": lengths.tolist(),
    }


__all__ = [
    "PrisonersDilemmaEnv",
    "StepInfo",
    "simulate_episode_lengths",
]
