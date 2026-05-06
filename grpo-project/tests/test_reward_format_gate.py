"""Reward function unit tests.

Authority: PRD v6.1 §4.1 acceptance criteria + Implementation Map §2.3.

Asserts:
  - Malformed completion → reward = 0
  - Well-formed C/D × opp C/D → expected payoff (no additive bonus)
  - Reward distribution is bimodal {0} ∪ {S, P, R, T}
"""

from __future__ import annotations

import pytest

from configs.config import PayoffMatrix
from training.reward import compute_reward, parse_completion


PAYOFFS = PayoffMatrix(T=5.0, R=3.0, P=1.0, S=0.0)


# ---------------------------------------------------------------------------
# parse_completion
# ---------------------------------------------------------------------------

def test_parse_well_formed_C():
    ok, action, n = parse_completion(
        "<reasoning>I will cooperate.</reasoning>\n<action>C</action>"
    )
    assert ok and action == "C" and n > 0


def test_parse_well_formed_D():
    ok, action, n = parse_completion(
        "<reasoning>defect</reasoning><action>D</action>"
    )
    assert ok and action == "D"


def test_parse_missing_reasoning_tag():
    ok, action, _ = parse_completion("<action>C</action>")
    assert not ok and action is None


def test_parse_missing_action_tag():
    ok, action, _ = parse_completion("<reasoning>think</reasoning>")
    assert not ok and action is None


def test_parse_invalid_action_letter():
    ok, action, _ = parse_completion(
        "<reasoning>x</reasoning><action>X</action>"
    )
    assert not ok and action is None


def test_parse_lowercase_rejected():
    ok, _, _ = parse_completion(
        "<reasoning>x</reasoning><action>c</action>"
    )
    assert not ok  # We require strict C or D


def test_parse_extra_whitespace_ok():
    ok, action, _ = parse_completion(
        "<reasoning>x</reasoning>     \n  <action>D</action>"
    )
    assert ok and action == "D"


# ---------------------------------------------------------------------------
# compute_reward — gate × payoff
# ---------------------------------------------------------------------------

def test_malformed_returns_zero():
    out = compute_reward("no tags", "C", PAYOFFS)
    assert out.reward == 0.0
    assert out.debug.format_ok is False


def test_well_formed_CC_gets_R():
    out = compute_reward(
        "<reasoning>x</reasoning><action>C</action>", "C", PAYOFFS
    )
    assert out.reward == PAYOFFS.R == 3.0


def test_well_formed_DC_gets_T():
    out = compute_reward(
        "<reasoning>x</reasoning><action>D</action>", "C", PAYOFFS
    )
    assert out.reward == PAYOFFS.T == 5.0


def test_well_formed_CD_gets_S():
    out = compute_reward(
        "<reasoning>x</reasoning><action>C</action>", "D", PAYOFFS
    )
    assert out.reward == PAYOFFS.S == 0.0


def test_well_formed_DD_gets_P():
    out = compute_reward(
        "<reasoning>x</reasoning><action>D</action>", "D", PAYOFFS
    )
    assert out.reward == PAYOFFS.P == 1.0


def test_no_additive_bonus_for_long_reasoning():
    """A 10000-char reasoning earns the same reward as a 5-char one (PRD §4.1)."""
    long_reasoning = "x" * 10000
    out_short = compute_reward(
        "<reasoning>x</reasoning><action>C</action>", "C", PAYOFFS
    )
    out_long = compute_reward(
        f"<reasoning>{long_reasoning}</reasoning><action>C</action>", "C", PAYOFFS
    )
    assert out_short.reward == out_long.reward


def test_invalid_opp_action_raises():
    with pytest.raises(ValueError):
        compute_reward(
            "<reasoning>x</reasoning><action>C</action>", "X", PAYOFFS
        )


# ---------------------------------------------------------------------------
# Bimodality sanity check (PRD §4.1 acceptance: distribution is bimodal)
# ---------------------------------------------------------------------------

def test_reward_value_set_is_bimodal():
    """Over many random completions, rewards land in {0} ∪ {S, P, R, T} only."""
    import random
    rng = random.Random(0)
    expected_nonzero = {PAYOFFS.S, PAYOFFS.P, PAYOFFS.R, PAYOFFS.T}
    rewards = []
    for _ in range(1000):
        if rng.random() < 0.5:
            comp = "garbage no tags"
        else:
            a = rng.choice(["C", "D"])
            comp = f"<reasoning>r</reasoning><action>{a}</action>"
        opp = rng.choice(["C", "D"])
        rewards.append(compute_reward(comp, opp, PAYOFFS).reward)
    unique = set(rewards)
    assert unique <= ({0.0} | expected_nonzero), unique
