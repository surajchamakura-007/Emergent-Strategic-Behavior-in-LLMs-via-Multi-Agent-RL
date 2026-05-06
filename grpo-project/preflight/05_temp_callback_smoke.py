"""Preflight 05 — Sampling-temperature runtime mutability probe (Map §5, Issue #4).

Rationale
---------
The R2 callback (`temp_bump_callback.py`) wants to bump the rollout sampling
temperature mid-run when advantage collapse is detected. Whether this knob is
actually mutable at runtime depends on the trl/vllm combination — the
trainer's `_patch_sampling_temperature` tries multiple paths
(args.temperature, sampling_params, generation_config, engine.default_sampling_params)
and verifies via readback. If NONE of the paths work, the callback must be
constructed with `bumped_temp=None` to enter "log only" mode rather than
silently no-op.

This preflight runs the probe and writes the verdict to
`configs/r2_runtime_mutable.json`, which `Config.from_yaml` reads on the next
training launch.

Test plan
---------
    1. Build the smoke trainer (fast init).
    2. Synthetically force `advantage_mean_abs < threshold` for 21 steps via
       a calibration_threshold=1.0 hack (any sub-threshold magnitude works).
       But: this preflight ONLY measures whether the temperature mutation
       lands; it does NOT need to fire R2 against real training data.
    3. Call `trainer._patch_sampling_temperature(1.2)` directly and ask the
       method to return True/False.
    4. Write `{"r2_runtime_mutable": <bool>, "probed_at": <timestamp>}`
       to `configs/r2_runtime_mutable.json`.

Output
------
    JSON at logs/preflight/05_temp_callback_smoke.json (preflight diagnostic)
    AND configs/r2_runtime_mutable.json (production artifact read by Config).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from configs.stack_versions import assert_versions  # noqa: E402
from utils.seed import seed_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out",
                        default="logs/preflight/05_temp_callback_smoke.json")
    parser.add_argument("--flag-out",
                        default="configs/r2_runtime_mutable.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "r2_runtime_mutable": False,
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.flag_out).parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_versions()
        cfg = Config.from_yaml(args.config)
        seed_all(cfg.seed)

        from training.train import build_smoke_trainer  # noqa: E402

        trainer, _ = build_smoke_trainer(cfg, num_steps=2)

        # Probe: try to bump from default 0.9 → 1.2 and back to 0.9.
        ok_up = trainer._patch_sampling_temperature(1.2)
        ok_down = trainer._patch_sampling_temperature(cfg.sampling_temp_default)

        out["r2_runtime_mutable"] = bool(ok_up and ok_down)
        out["passed"] = True  # the probe always "passes" — the verdict matters

        with open(args.flag_out, "w") as f:
            json.dump({
                "r2_runtime_mutable": out["r2_runtime_mutable"],
                "probed_at": time.time(),
                "trl_version": __import__("trl").__version__,
                "vllm_version": __import__("vllm").__version__,
            }, f, indent=2)

    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
        out["passed"] = False
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({
            "passed": out["passed"],
            "r2_runtime_mutable": out["r2_runtime_mutable"],
        }, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
