"""R2 mitigation: detect Dr. GRPO advantage collapse and bump sampling temp.

Authority: Implementation Map §5 (full design, closes issues #3 and #4) +
PRD v6.1 R2.

Why R2 exists. Under Dr. GRPO (D1), advantage = R_i - R_mean. When group
rollouts converge — either to mutual cooperation (success) or to a
saturated equilibrium (failure mode) — `R_i - R_mean → 0` for all i, and
the policy gradient dies. The mitigation is to inject token-level diversity
by raising the rollout sampling temperature.

The cooperation-rate gate (issue #3 fix). Bumping temp at high coop-rate
would inject noise into a working run. So we gate:
    Fire iff (advantage_mean_abs < threshold for window steps)
         AND (cooperation_rate < coop_ceiling)

When the gate skips, we LOG but do NOT latch `fired = True`. If the run
later drifts down to lower coop-rate with sustained collapse, R2 should
still be available (Map §5.2).

The runtime-mutability fallback (issue #4 fix). If `_patch_sampling_temperature`
returns False, two things can be true:
    (a) Pre-flight 05 verified the temp knob is mutable (`r2_runtime_mutable=True`)
        but the mutation failed mid-run → halt loudly (something is wrong).
    (b) Pre-flight 05 verified the knob is NOT mutable
        (`r2_runtime_mutable=False`); construct the callback with `bumped_temp=None`
        so the callback logs a "would-fire" event and does NOT halt.
The pre-flight script writes `configs/r2_runtime_mutable.json`, which
`Config.from_yaml` reads and propagates to `cfg.r2_runtime_mutable`.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from training.callbacks.diagnostic_logging import RunState


class R2MitigationFailedError(RuntimeError):
    """Pre-flight said temp knob is mutable; runtime says otherwise."""


class TempBumpCallback(TrainerCallback):
    """Detects advantage collapse and (if mutable) bumps sampling temperature."""

    def __init__(
        self,
        run_state: RunState,
        threshold: float,
        *,
        coop_ceiling: float = 0.85,
        bumped_temp: float | None = 1.2,
        window_steps: int = 20,
        trainer_ref: Any = None,
        wandb_run=None,
        runtime_mutability_verified: bool = True,
    ) -> None:
        """
        Args:
            run_state: shared RunState (provides advantage and coop windows).
            threshold: advantage_mean_abs floor below which collapse is suspected.
                       From `analysis/threshold_calibration.py`.
            coop_ceiling: skip-gate. Default 0.85 (Map §5.3 chosen value).
            bumped_temp: target sampling temperature on fire. None = "log only"
                         path (preflight 05 said mutability is False).
            window_steps: how many recent steps to average over (PRD R2: 20).
            trainer_ref: weakref to the FrozenSnapshotGRPOTrainer (avoids
                         retain cycles; trainer outlives callback in normal
                         flow but we must not pin it).
            runtime_mutability_verified: True iff preflight 05 confirmed the
                                         temp knob is runtime-mutable.
        """
        self.run_state = run_state
        self.threshold = float(threshold)
        self.coop_ceiling = float(coop_ceiling)
        self.bumped_temp = bumped_temp
        self.window_steps = int(window_steps)
        # Use weakref if a live trainer is passed; allow `None` for tests.
        if trainer_ref is None:
            self._trainer_ref = lambda: None
        elif isinstance(trainer_ref, weakref.ref):
            self._trainer_ref = trainer_ref
        else:
            self._trainer_ref = weakref.ref(trainer_ref)
        self.wandb_run = wandb_run
        self.runtime_mutability_verified = runtime_mutability_verified
        self.fired = False

    # --- core logic ---

    def on_step_end(self, args, state, control, **kwargs):
        if self.fired:
            return

        adv_window = self.run_state.advantage_mean_abs_window
        coop_window = self.run_state.cooperation_rate_window
        if len(adv_window) < self.window_steps or len(coop_window) < self.window_steps:
            return  # not enough data yet

        adv_mean = float(np.mean(list(adv_window)[-self.window_steps:]))
        coop_mean = float(np.mean(list(coop_window)[-self.window_steps:]))

        # Trigger condition #1: advantage collapsed
        if adv_mean >= self.threshold:
            return

        # Coop-ceiling gate (Map §5.2): log + skip without latching.
        if coop_mean >= self.coop_ceiling:
            self._wblog(state, {
                "r2_callback/skipped_at_step": state.global_step,
                "r2_callback/skip_reason": "coop_at_ceiling",
                "r2_callback/coop_rate": coop_mean,
                "r2_callback/adv_mean_abs": adv_mean,
            })
            return

        # We're firing. Three branches:
        if self.bumped_temp is None:
            # Preflight 05 said: not mutable. Log a "would-fire" once.
            self.fired = True  # one-shot log
            self._wblog(state, {
                "r2_callback/would_fire_at_step": state.global_step,
                "r2_callback/skip_reason": "runtime_temp_not_mutable",
                "r2_callback/coop_rate": coop_mean,
                "r2_callback/adv_mean_abs": adv_mean,
            })
            return

        trainer = self._trainer_ref()
        if trainer is None:
            # Impossible in normal flow — trainer holds the callback.
            raise R2MitigationFailedError(
                f"R2 fired at step {state.global_step} but trainer ref is dead."
            )

        landed = bool(trainer._patch_sampling_temperature(self.bumped_temp))
        if landed:
            self.fired = True
            self.run_state.sampling_temperature = self.bumped_temp
            self._wblog(state, {
                "r2_callback/fired_at_step": state.global_step,
                "r2_callback/old_temp": 0.9,
                "r2_callback/new_temp": self.bumped_temp,
                "r2_callback/coop_rate_at_fire": coop_mean,
                "r2_callback/adv_mean_abs_at_fire": adv_mean,
            })
            return

        # Mutation failed at runtime despite preflight verification.
        # If preflight already said "not mutable", this branch is unreachable
        # (we returned above). So pre-flight verification failed live.
        if self.runtime_mutability_verified:
            self._wblog(state, {
                "r2_callback/halt_at_step": state.global_step,
                "r2_callback/halt_reason": "temp_mutation_failed",
            })
            raise R2MitigationFailedError(
                f"R2 fired at step {state.global_step} but temp mutation did "
                f"not land. Pre-flight should have caught this. Halting."
            )

        # Defensive — shouldn't be reachable.
        self.fired = True
        self._wblog(state, {
            "r2_callback/skipped_at_step": state.global_step,
            "r2_callback/skip_reason": "mutation_failed_unverified",
        })

    # --- helpers ---

    def _wblog(self, state, payload: dict) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(payload, step=state.global_step)
        except Exception:
            pass


__all__ = ["TempBumpCallback", "R2MitigationFailedError"]
