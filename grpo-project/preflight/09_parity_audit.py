"""Preflight 09 — Cluster ↔ RunPod numerical parity audit (R11, S1-11 mitigation).

Rationale
---------
We train on Bridges-2 V100-32GB and evaluate on RunPod RTX 4090. With D6
dropped (PRD v6.1: FP16 throughout, no FP32 LM-head), there is no precision
boundary — both stacks run FP16 — so divergence should be small. But seeded
RNG, BLAS dispatch tables, and FlashAttention quirks can still produce
nontrivial drift. The execution plan §3.2 specifies: log first 10 steps'
reward, advantage_mean_abs, group_reward_std on both. Halt if any metric
diverges >5%.

This script runs the smoke on whatever device it's on and emits a CSV that
gets compared offline. Run once per platform (cluster, RunPod), then diff.

Test plan
---------
    1. Run a deterministic 10-step training pass at fixed seed.
    2. Capture per-step:
        - reward_mean
        - advantage_mean_abs
        - group_reward_std
    3. Write to `logs/preflight/09_parity_<platform>.json`.
    4. (Offline) `analysis/parity_audit.py` consumes the two JSONs and
       reports diffs.

Pass criterion
--------------
    The script itself just records data. Pass = clean run + JSON written.
    The actual parity verdict is in `analysis/parity_audit.py`.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from configs.stack_versions import assert_versions  # noqa: E402
from utils.seed import seed_all  # noqa: E402


def _platform_label() -> str:
    if not torch.cuda.is_available():
        return f"cpu_{platform.node()}"
    name = torch.cuda.get_device_properties(0).name.replace(" ", "_")
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--platform-label", default=None,
                        help="Override autodetect, e.g., 'bridges_v100' or 'runpod_4090'.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    label = args.platform_label or _platform_label()
    out_path = args.out or f"logs/preflight/09_parity_{label}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out = {
        "passed": False,
        "platform_label": label,
        "device_name": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "step_records": [],
        "errors": [],
    }

    try:
        assert_versions()
        cfg = Config.from_yaml(args.config)
        seed_all(cfg.seed)
        if torch.cuda.is_available():
            out["device_name"] = torch.cuda.get_device_properties(0).name

        from training.train import build_smoke_trainer  # noqa: E402
        from transformers import TrainerCallback  # noqa: E402

        records: list[dict] = []

        class _Probe(TrainerCallback):
            def on_log(self, args_, state, control, logs=None, **kwargs):
                if not logs:
                    return
                rec = {"step": state.global_step}
                for k in ("reward", "rewards/mean", "advantage_mean_abs",
                         "group_reward_std"):
                    if k in logs:
                        try:
                            rec[k] = float(logs[k])
                        except (TypeError, ValueError):
                            pass
                if len(rec) > 1:
                    records.append(rec)

        trainer, _ = build_smoke_trainer(cfg, num_steps=10)
        trainer.add_callback(_Probe())
        trainer.train()

        out["step_records"] = records
        out["passed"] = len(records) >= 5
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"],
                          "platform_label": label,
                          "n_records": len(out["step_records"])}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
