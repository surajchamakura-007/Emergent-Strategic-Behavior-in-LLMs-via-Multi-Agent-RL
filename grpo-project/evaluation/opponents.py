"""Fixed-strategy opponents for IPD evaluation.

Authority: PRD v6.1 §8.2 + STAGE1_EXECUTION_PLAN_v3.1 §6.1 (Tier A).

Each opponent implements a single contract:

    .name: str                       — used in W&B and JSON output keys
    .reset(seed: int) -> None         — start of episode, may seed an RNG
    .act(history: list[Round]) -> str — return 'C' or 'D' given history

Because LLM rollouts are sequential per-round (the agent emits one
`<action>` per turn), the opponent is queried synchronously each round.

We deliberately keep these dependency-free — no torch, no vllm — so they
can be reused in unit tests and in the trace eval.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Sequence

from envs.history import Round


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Opponent(ABC):
    """Stateless or seeded fixed-strategy opponent."""

    name: str = "base"

    def reset(self, seed: int = 0) -> None:
        """Override in subclasses that hold per-episode RNG state."""

    @abstractmethod
    def act(self, history: Sequence[Round]) -> str:
        """Return 'C' or 'D' given the *agent's* observed history.

        Note: `history[i].opp_action` is the action the OPPONENT played in
        round i. So a TfT opponent that mirrors the agent looks at the
        agent's PREVIOUS action, which is `history[-1].my_action`.
        """


# ---------------------------------------------------------------------------
# Fixed strategies — Tier A baselines
# ---------------------------------------------------------------------------

class AlwaysCooperate(Opponent):
    name = "AlwaysCooperate"

    def act(self, history: Sequence[Round]) -> str:
        return "C"


class AlwaysDefect(Opponent):
    name = "AlwaysDefect"

    def act(self, history: Sequence[Round]) -> str:
        return "D"


class TitForTat(Opponent):
    """Round 1: cooperate. Subsequent: mirror the agent's last action."""
    name = "TfT"

    def act(self, history: Sequence[Round]) -> str:
        if not history:
            return "C"
        return history[-1].my_action


class Random50(Opponent):
    """Bernoulli(0.5) coin flip per round, with a per-episode seed."""
    name = "Random50"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def act(self, history: Sequence[Round]) -> str:
        return "C" if self._rng.random() < 0.5 else "D"


class GenerousTitForTat(Opponent):
    """Cooperate, mirror, but with a small forgiveness rate (used in transfer)."""
    name = "GenerousTfT"

    def __init__(self, forgiveness: float = 0.10, seed: int = 0) -> None:
        self.forgiveness = forgiveness
        self._rng = random.Random(seed)

    def reset(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, history: Sequence[Round]) -> str:
        if not history:
            return "C"
        prev_agent = history[-1].my_action
        if prev_agent == "C":
            return "C"
        return "C" if self._rng.random() < self.forgiveness else "D"


class GrimTrigger(Opponent):
    """Cooperate until the agent defects once; then defect forever."""
    name = "GrimTrigger"

    def __init__(self) -> None:
        self._tripped = False

    def reset(self, seed: int = 0) -> None:
        self._tripped = False

    def act(self, history: Sequence[Round]) -> str:
        if self._tripped:
            return "D"
        if any(r.my_action == "D" for r in history):
            self._tripped = True
            return "D"
        return "C"


# ---------------------------------------------------------------------------
# GPT-4o-mini opponent (Tier A external baseline)
# ---------------------------------------------------------------------------

class GPT4oMiniOpponent(Opponent):
    """GPT-4o-mini playing IPD via the OpenAI chat completions API.

    The agent prompt is the same as the one our trained model sees, but
    with a system role describing the task in plain English. The opponent
    returns 'C' or 'D' parsed from a `<action>X</action>` block.

    This is *not* a faithfulness comparison — it's a payoff comparison.
    The trained model's reasoning is what we measure with REMUL/RFEval.
    """
    name = "GPT-4o-mini"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; required for GPT-4o-mini opponent."
            )

    def act(self, history: Sequence[Round]) -> str:
        # Lazy import so the OpenAI dep doesn't bleed into other contexts.
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        prompt = self._render_history(history)
        msg = [
            {"role": "system", "content": (
                "You are an agent in an iterated game where, each round, "
                "you choose to Cooperate (C) or Defect (D). The game has a "
                "stochastic horizon (continuation probability 0.95). "
                "Reply ONLY in the format:\n"
                "<reasoning>...</reasoning>\n"
                "<action>C</action>  or  <action>D</action>"
            )},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(
            model=self.model,
            messages=msg,
            temperature=self.temperature,
            max_tokens=200,
        )
        text = resp.choices[0].message.content or ""
        return self._parse(text)

    @staticmethod
    def _render_history(history: Sequence[Round]) -> str:
        if not history:
            return "Round 1. No history yet. Make your move."
        lines = []
        for i, r in enumerate(history[-25:], start=max(1, len(history) - 24)):
            lines.append(f"Round {i}: you={r.opp_action}, opp={r.my_action}")
        lines.append(f"Round {len(history) + 1}. Make your move.")
        return "\n".join(lines)

    @staticmethod
    def _parse(text: str) -> str:
        # Tolerant parse: scan from the end for the first <action>X</action>.
        # If GPT-4o-mini fails the format, default to 'D' (worst-case for
        # the trained model — keeps the comparison conservative).
        import re
        m = re.search(r"<action>([CD])</action>", text)
        return m.group(1) if m else "D"


__all__ = [
    "Opponent",
    "AlwaysCooperate",
    "AlwaysDefect",
    "TitForTat",
    "GenerousTitForTat",
    "GrimTrigger",
    "Random50",
    "GPT4oMiniOpponent",
]
