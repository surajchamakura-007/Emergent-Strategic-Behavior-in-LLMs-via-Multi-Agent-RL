"""Tests for TempBumpCallback — Map §5.5.

Coverage:
    - test_skips_at_high_coop: sustained low-advantage but coop-rate high → skip
      (logs but does not latch `fired`).
    - test_fires_at_low_coop: sustained low-advantage AND low coop-rate → fire
      once. `bumped_temp=None` test path so we don't need a real engine.
    - test_one_shot: even after firing, do not fire again.
    - test_short_window_no_fire: only `window_steps - 5` collapsed steps → no
      fire (window not full of below-threshold samples yet).
    - test_skip_does_not_latch: a coop-skip episode followed by coop-drop
      should still allow firing (Map §5.2).
    - test_runtime_mismatch_raises: pre-flight said mutable but mutation
      fails at runtime → R2MitigationFailedError.

We construct the callback with `trainer_ref=None`, so `_patch_sampling_temperature`
is short-circuited to True for the `bumped_temp=None` path. For the
`R2MitigationFailedError` test we inject a fake trainer whose
`_patch_sampling_temperature` returns False.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from training.callbacks.diagnostic_logging import RunState
from training.callbacks.temp_bump_callback import (
    R2MitigationFailedError,
    TempBumpCallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trl_state(step: int) -> SimpleNamespace:
    """Minimal stand-in for transformers.TrainerState."""
    return SimpleNamespace(global_step=step)


def _trl_args() -> SimpleNamespace:
    return SimpleNamespace()


def _trl_control() -> SimpleNamespace:
    return SimpleNamespace()


def _push_window(rs: RunState, *, advantage: float, coop: float, steps: int,
                 start_step: int = 1) -> int:
    """Drive RunState forward `steps` times with constant per-step values.

    Returns the next step index (so callers can chain pushes).
    """
    for s in range(start_step, start_step + steps):
        # Simulate one rollout per step, action=C if coop else D, format ok.
        rs.reset_step_buffers()
        rs.record_rollout(
            opponent_is_snapshot=False,
            action_is_coop=(coop >= 0.5),
            format_ok=True,
        )
        # Synthesize advantage_mean_abs by reading from a TRL-style log dict.
        log_payload = {"advantage_mean_abs": advantage}
        rs.update_from_step(step=s, log_payload=log_payload)
        rs.step_end_aggregates(step=s)
    return start_step + steps


def _make_cb(
    *, threshold: float = 0.10, window: int = 20,
    bumped_temp: float | None = None,
    runtime_mutability_verified: bool = True,
    trainer_ref: Any = None,
) -> TempBumpCallback:
    rs = RunState(window_size=window)
    cb = TempBumpCallback(
        run_state=rs,
        threshold=threshold,
        coop_ceiling=0.85,
        bumped_temp=bumped_temp,
        window_steps=window,
        trainer_ref=trainer_ref,
        wandb_run=None,
        runtime_mutability_verified=runtime_mutability_verified,
    )
    return cb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_skips_at_high_coop_does_not_latch():
    """Sustained advantage collapse with high coop-rate → skip without latching.

    Per Map §5.2: a coop-skip should NOT set `fired = True`. If the run later
    drifts down with coop dropping, R2 must still be available.
    """
    cb = _make_cb()
    rs = cb.run_state
    # Push 25 collapsed steps at high coop.
    next_step = _push_window(rs, advantage=0.02, coop=0.95, steps=25)
    # Drive each via on_step_end.
    for s in range(1, next_step):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is False, "callback latched on coop-skip path"


def test_fires_at_low_coop_log_only_path():
    """Sustained collapse + low coop → fires exactly once on the log-only path.

    With `bumped_temp=None`, the callback emits the 'would-fire' event and
    short-circuits before touching any engine. fired=True after.
    """
    cb = _make_cb(bumped_temp=None, runtime_mutability_verified=False)
    rs = cb.run_state
    next_step = _push_window(rs, advantage=0.01, coop=0.20, steps=22)
    for s in range(1, next_step):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is True


def test_one_shot_does_not_re_fire():
    """Once fired, additional collapsed steps must not re-fire."""
    cb = _make_cb(bumped_temp=None, runtime_mutability_verified=False)
    rs = cb.run_state
    next_step = _push_window(rs, advantage=0.01, coop=0.20, steps=22)
    for s in range(1, next_step):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is True
    # Push another window of collapse and confirm `.fired` stays True (i.e.
    # we don't cycle it) and no exception is raised.
    next_step2 = _push_window(rs, advantage=0.01, coop=0.20, steps=22,
                              start_step=next_step)
    for s in range(next_step, next_step2):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is True


def test_short_window_no_fire():
    """Fewer than `window_steps` samples below threshold → no fire."""
    cb = _make_cb(window=20, bumped_temp=None, runtime_mutability_verified=False)
    rs = cb.run_state
    # 15 collapsed + 5 healthy: the rolling mean over the last 20 is dominated
    # by 15 values at 0.01 and 5 at 0.40 → mean ≈ 0.108; above the 0.10
    # threshold so no fire.
    next_step = _push_window(rs, advantage=0.01, coop=0.20, steps=15)
    next_step = _push_window(rs, advantage=0.40, coop=0.20, steps=5,
                             start_step=next_step)
    for s in range(1, next_step):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is False


def test_skip_then_fire_after_coop_drops():
    """Coop-skip episode does NOT pre-empt later firing once coop drops.

    Map §5.2: skip-gate is a per-step decision, not a latch.
    """
    cb = _make_cb(bumped_temp=None, runtime_mutability_verified=False)
    rs = cb.run_state
    # Phase 1: collapsed advantage but high coop — skip path.
    next_step = _push_window(rs, advantage=0.01, coop=0.95, steps=25)
    for s in range(1, next_step):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is False
    # Phase 2: same collapse, but coop drops — should fire now.
    next_step2 = _push_window(rs, advantage=0.01, coop=0.10, steps=22,
                              start_step=next_step)
    for s in range(next_step, next_step2):
        cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
    assert cb.fired is True


def test_runtime_mismatch_raises():
    """Pre-flight said mutable but mutation fails at runtime → halt loudly.

    Construct a fake trainer whose `_patch_sampling_temperature` returns False
    despite `runtime_mutability_verified=True`. This is the failure mode
    Map §5 wants us to catch instead of silently degrading.
    """
    class _BrokenTrainer:
        def _patch_sampling_temperature(self, new_temp: float) -> bool:
            return False

    fake = _BrokenTrainer()
    cb = _make_cb(
        bumped_temp=1.2,
        runtime_mutability_verified=True,
        trainer_ref=fake,
    )
    rs = cb.run_state
    next_step = _push_window(rs, advantage=0.01, coop=0.10, steps=22)
    with pytest.raises(R2MitigationFailedError):
        for s in range(1, next_step):
            cb.on_step_end(_trl_args(), _trl_state(s), _trl_control())
