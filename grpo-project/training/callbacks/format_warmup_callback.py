"""Format-warmup R1 callback — halt-and-instruct rather than auto-launch SFT.

Authority: Implementation Map §2.5 + PRD v6.1 R1.

Trigger. After 20 steps of training, if the rolling mean of
`format_violation_rate` exceeds 0.30, the run cannot recover because reward
is gated multiplicatively (PRD §4.1) — gradient signal is too sparse.

Action. Raise `FormatWarmupRequiredError`. The user runs the SFT format-only
warmup as a separate script (out of scope for Stage 1; Map §2.5 explicitly
chose "logged + halt" over "automated SFT" because the SFT script would
add another moving piece to debug under deadline pressure).

Why not at step 1. Format violations in the first ~5 steps are normal as
the model adapts to the new prompt template. The 20-step window in PRD R1
is the validated trigger.
"""

from __future__ import annotations

import numpy as np

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from training.callbacks.diagnostic_logging import RunState


class FormatWarmupRequiredError(RuntimeError):
    """Raised when format-violation rate exceeds the R1 threshold."""


class FormatWarmupCallback(TrainerCallback):
    """Watches `RunState.format_violation_rate_window`; halts on R1 violation.

    Args:
        run_state: shared RunState.
        threshold: violation rate above which to halt. Default 0.30 (PRD R1).
        check_at_step: step number at which to evaluate. Default 20 (PRD R1).
    """

    def __init__(
        self,
        run_state: RunState,
        threshold: float = 0.30,
        check_at_step: int = 20,
        wandb_run=None,
    ) -> None:
        self.run_state = run_state
        self.threshold = threshold
        self.check_at_step = check_at_step
        self.wandb_run = wandb_run
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._fired:
            return
        if state.global_step < self.check_at_step:
            return

        window = self.run_state.format_violation_rate_window
        if len(window) < self.check_at_step:
            return  # not enough samples yet (e.g., resumed run)

        rate = float(np.mean(window))
        if rate > self.threshold:
            self._fired = True
            msg = (
                f"R1 trigger: format-violation rate = {rate:.3f} > "
                f"{self.threshold} after {self.check_at_step} steps. "
                f"Per PRD v6.1 R1, run a brief SFT format-only warmup before "
                f"resuming GRPO. Suggested: 5 epochs over a synthetic dataset "
                f"of well-formed (reasoning + action) examples, no strategic "
                f"content. Then re-launch from step 0."
            )
            if self.wandb_run is not None:
                try:
                    self.wandb_run.log(
                        {"r1_callback/halt_at_step": state.global_step,
                         "r1_callback/format_violation_rate": rate},
                        step=state.global_step,
                    )
                except Exception:
                    pass
            raise FormatWarmupRequiredError(msg)


__all__ = ["FormatWarmupCallback", "FormatWarmupRequiredError"]
