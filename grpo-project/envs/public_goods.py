"""Public Goods Game (PGG, 2-player) — Tier C transfer eval (PRD v6.1 §8.2).

Two-player binary PGG: each agent decides to Contribute (C) or Free-ride (D)
with an endowment of 1 token. Contributions are pooled, multiplied by a
factor m (1 < m < N=2 ⇒ social dilemma), and split equally. Payoffs:

    Each agent's payoff = (1 - contribution) + m * total_contribution / N
    where contribution = 1 if action == C else 0.

For m = 1.5, N = 2:
    CC: 0 + 1.5 * 2 / 2 = 1.5      (R)
    CD: 0 + 1.5 * 1 / 2 = 0.75     (S, contributor)
    DC: 1 + 1.5 * 1 / 2 = 1.75     (T_, free-rider)
    DD: 1 + 1.5 * 0 / 2 = 1.0      (P)

Same stochastic-horizon machinery as IPD: p=0.95, cap=60. The transfer
test checks whether IPD-trained policies generalize to a different payoff
cell with the same C/D action interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from envs.history import Round


@dataclass(frozen=True)
class PublicGoodsPayoffs:
    """N=2 PGG with multiplier m. Default m=1.5."""
    multiplier: float = 1.5
    endowment: float = 1.0

    def lookup(self, my: str, opp: str) -> float:
        n = 2.0
        my_contrib = 1.0 if my == "C" else 0.0
        opp_contrib = 1.0 if opp == "C" else 0.0
        return (
            (self.endowment - my_contrib)
            + self.multiplier * (my_contrib + opp_contrib) / n
        )


@dataclass(frozen=True)
class StepInfo:
    my_payoff: float
    opp_payoff: float
    done: bool
    round_idx: int


class PublicGoodsEnv:
    def __init__(
        self,
        payoffs: Optional[PublicGoodsPayoffs] = None,
        *,
        p: float = 0.95,
        cap: int = 60,
        seed: int = 0,
    ) -> None:
        self.payoffs = payoffs or PublicGoodsPayoffs()
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
        cont = self.rng.random() < self.p
        if not cont or self._round >= self.cap:
            self._done = True
        return StepInfo(
            my_payoff=my_pay, opp_payoff=opp_pay,
            done=self._done, round_idx=self._round,
        )


__all__ = ["PublicGoodsEnv", "PublicGoodsPayoffs", "StepInfo"]
