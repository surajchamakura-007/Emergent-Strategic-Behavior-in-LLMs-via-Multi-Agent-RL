"""Stag Hunt environment — Tier C transfer eval (PRD v6.1 §8.2).

Stag Hunt has the same action set ({C, D}, where C means "hunt stag" and D
means "hunt hare"), the same stochastic horizon (p=0.95, cap=60), and the
same `step()` signature as `PrisonersDilemmaEnv`. The only thing that
changes is the payoff matrix.

Standard Stag Hunt payoffs (2-player, normal-form):
    Both stag (CC):     (R, R) where R is high (mutual gain).
    Both hare (DD):     (P, P) where P > 0 (safe; no temptation to deviate).
    Stag vs hare (CD):  (S, T_) where S=0 (you went hunting alone) and T_>=P.
    Hare vs stag (DC):  (T_, S).

Crucially, T_ < R: there is no temptation to defect from mutual cooperation.
The dilemma is coordination, not incentive-compatibility.

We use the canonical numbers from PRD §8.2: R=4, P=2, S=0, T_=2.
(These are *not* the IPD payoffs — same names, different cell.)

The env interface is intentionally identical to PrisonersDilemmaEnv so eval
code can swap envs by config.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from envs.history import Round


@dataclass(frozen=True)
class StagHuntPayoffs:
    R: float = 4.0
    P: float = 2.0
    S: float = 0.0
    T_: float = 2.0   # underscore to avoid colliding with `T` in IPD code

    def lookup(self, my: str, opp: str) -> float:
        if my == "C" and opp == "C":
            return self.R
        if my == "D" and opp == "D":
            return self.P
        if my == "C" and opp == "D":
            return self.S
        if my == "D" and opp == "C":
            return self.T_
        raise ValueError(f"actions must be C/D; got {my=!r}, {opp=!r}")


@dataclass(frozen=True)
class StepInfo:
    my_payoff: float
    opp_payoff: float
    done: bool
    round_idx: int


class StagHuntEnv:
    """Stag Hunt with stochastic horizon (geometric p=0.95, capped at 60)."""

    def __init__(
        self,
        payoffs: Optional[StagHuntPayoffs] = None,
        *,
        p: float = 0.95,
        cap: int = 60,
        seed: int = 0,
    ) -> None:
        self.payoffs = payoffs or StagHuntPayoffs()
        self.p = float(p)
        self.cap = int(cap)
        self.rng = random.Random(seed)
        self._round = 0
        self._done = False
        self.history: list[Round] = []

    def reset(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._round = 0
        self._done = False
        self.history = []

    def step(self, my_action: str, opp_action: str) -> StepInfo:
        if self._done:
            raise RuntimeError("step called after env signaled done")
        my_pay = self.payoffs.lookup(my_action, opp_action)
        opp_pay = self.payoffs.lookup(opp_action, my_action)
        self.history.append(Round(
            my_action=my_action, opp_action=opp_action, my_payoff=my_pay,
        ))
        self._round += 1
        # Continue with prob p; stop with prob 1-p; hard cap at `cap`.
        cont = self.rng.random() < self.p
        if not cont or self._round >= self.cap:
            self._done = True
        return StepInfo(
            my_payoff=my_pay, opp_payoff=opp_pay,
            done=self._done, round_idx=self._round,
        )


__all__ = ["StagHuntEnv", "StagHuntPayoffs", "StepInfo"]
