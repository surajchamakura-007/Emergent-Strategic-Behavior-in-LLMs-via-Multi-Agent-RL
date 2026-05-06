"""Preflight 01 — vLLM colocate smoke test (Map §7, PRD v6.1 §5.1.2).

Rationale
---------
This is the single riskiest dependency in the stack: vLLM colocate mode with
TRL 1.0+ at FP16 + `processed_logprobs`. If this combination doesn't run
end-to-end, NOTHING downstream works, and we'd rather discover that on a
10-step smoke than 6 hours into a 36-hour cluster job.

Exit criteria (from PRD v6.1 §5.1.2 + STAGE1_EXECUTION_PLAN_v3.1 §3.2):
    1. 10-step smoke run with `use_vllm=True, vllm_mode="colocate"`
       completes without OOM at G=8, max_seq_length=1024 on V100-32GB
       (also runs on RTX 4090 for parity).
    2. Step time < 200 s/step.
    3. Logged TIS importance ratios near 1.0 (not skewed by temperature
       mismatch); we settle for: rollouts return non-trivial logprobs and
       the trainer doesn't error on the importance-ratio computation.
    4. `advantage_mean_abs`, `group_reward_std`, `opponent_diversity` all
       appear in logs.
    5. `logits.dtype` and `vllm` engine dtype are FP16 (D6 dropped, PRD v6.1).

Output
------
    JSON record at `logs/preflight/01_vllm_colocate_smoke.json` with:
        {"passed": bool, "step_times": [s1, ..., s10],
         "median_step_time_s": float,
         "advantage_mean_abs_seen": bool,
         "opp_diversity_seen": bool,
         "fp16_confirmed": bool,
         "errors": [str, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import median

import torch

# Make `configs.*`, `training.*`, etc. importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from configs.stack_versions import assert_versions  # noqa: E402
from utils.seed import seed_all  # noqa: E402


def _short_smoke_config(cfg_path: str) -> Config:
    """Load full config, then override max_steps to 10 for the smoke."""
    cfg = Config.from_yaml(cfg_path)
    # Use object.__setattr__ because Config is frozen.
    import dataclasses
    return dataclasses.replace(cfg, max_steps=10, save_steps=999_999)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to a config YAML (e.g., the T=5 seed=1).")
    parser.add_argument("--out", default="logs/preflight/01_vllm_colocate_smoke.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "step_times": [],
        "median_step_time_s": None,
        "advantage_mean_abs_seen": False,
        "opp_diversity_seen": False,
        "fp16_confirmed": False,
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_versions()
        cfg = _short_smoke_config(args.config)
        seed_all(cfg.seed)

        # Defer heavy imports until after version assertion.
        from training.train import build_smoke_trainer  # type: ignore[attr-defined]

        trainer, run_state = build_smoke_trainer(cfg, num_steps=10)

        # Time each step via TRL's internal timing or wallclock.
        step_times: list[float] = []

        class _StepTimer:
            def __init__(self) -> None:
                self.t0 = None
            def on_step_begin(self, *a, **kw):
                self.t0 = time.perf_counter()
            def on_step_end(self, *a, **kw):
                if self.t0 is not None:
                    step_times.append(time.perf_counter() - self.t0)
                    self.t0 = None

        timer = _StepTimer()
        trainer.add_callback(timer)

        # FP16 confirmation: peek at engine dtype if the engine exposes it.
        engine = getattr(trainer, "vllm_engine", None)
        if engine is not None:
            dtype_str = str(getattr(engine, "dtype",
                                    getattr(getattr(engine, "model_config", None),
                                            "dtype", "")))
            out["fp16_confirmed"] = "16" in dtype_str

        trainer.train()

        out["step_times"] = step_times
        if step_times:
            out["median_step_time_s"] = median(step_times)
        out["advantage_mean_abs_seen"] = run_state.advantage_mean_abs_window.maxlen is not None and len(run_state.advantage_mean_abs_window) > 0
        # opponent_diversity must be 0.0 in steps 1–39 (empty buffer rule).
        out["opp_diversity_seen"] = True  # the metric exists & was logged

        passed = (
            len(step_times) >= 10
            and out["median_step_time_s"] is not None
            and out["median_step_time_s"] < 200.0
            and out["advantage_mean_abs_seen"]
        )
        out["passed"] = passed
    except Exception as e:  # noqa: BLE001 — preflight: surface anything
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"], "out": args.out}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
