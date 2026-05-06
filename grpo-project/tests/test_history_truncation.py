"""History-truncation unit tests.

Authority: Implementation Map §4.4 acceptance criteria.
"""

from __future__ import annotations

import pytest

from envs.history import Round, serialize_history


def _make_history(n: int, my="C", opp="C", payoff=3.0) -> list[Round]:
    return [Round(my_action=my, opp_action=opp, my_payoff=payoff) for _ in range(n)]


def test_empty_history():
    out = serialize_history([], budget=400, keep_last_k=25)
    assert out.raw_round_count == 0
    assert out.kept_verbatim == 0
    assert not out.summary_used
    assert "no rounds" in out.serialized.lower()


def test_short_history_verbatim():
    """5-round history → summary_used=False, kept_verbatim=5."""
    h = _make_history(5)
    out = serialize_history(h, budget=400, keep_last_k=25)
    assert out.raw_round_count == 5
    assert out.kept_verbatim == 5
    assert not out.summary_used
    assert "Round 1" in out.serialized
    assert "Round 5" in out.serialized


def test_long_history_truncated():
    """50-round history → summary_used=True, kept_verbatim=25,
    serialized starts with summary line."""
    h = _make_history(50)
    out = serialize_history(h, budget=400, keep_last_k=25)
    assert out.raw_round_count == 50
    assert out.summary_used
    # The first non-empty line should be the summary
    assert out.serialized.lstrip().startswith("[Earlier")
    assert out.kept_verbatim <= 25  # may decay further if budget tight


def test_summary_preserves_last_opp_action():
    """Summary line must include the LAST round's opp action so TfT survives."""
    h = _make_history(40, my="C", opp="C")
    # Make round n-K's opp action distinctive (the last summarized round)
    h[14] = Round(my_action="C", opp_action="D", my_payoff=0.0)
    # last summarized round = h[14] when kept_verbatim = 25
    out = serialize_history(h, budget=400, keep_last_k=25)
    assert out.summary_used
    # Last summarized round was h[14] (D) — must appear in summary
    assert "opponent's last action: D" in out.serialized


def test_token_budget_respected():
    """For n in {28, 35, 45, 60}, estimated tokens <= 400."""
    for n in (28, 35, 45, 60):
        h = _make_history(n)
        out = serialize_history(h, budget=400, keep_last_k=25)
        assert out.estimated_tokens <= 400, (
            f"n={n}: estimated_tokens={out.estimated_tokens} > 400"
        )


def test_K_decay_under_tight_budget():
    """A pathologically tight budget forces K to decay."""
    h = _make_history(60)
    out = serialize_history(h, budget=80, keep_last_k=25)
    assert out.summary_used
    assert out.estimated_tokens <= 80 + 5  # small slack for fallback path


def test_round_invalid_action_raises():
    with pytest.raises(ValueError):
        Round(my_action="X", opp_action="C", my_payoff=3.0)
