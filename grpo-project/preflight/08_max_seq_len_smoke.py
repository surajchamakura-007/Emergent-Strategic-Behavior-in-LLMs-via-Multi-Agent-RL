"""Preflight 08 — VRAM smoke at max_seq_length=1024 (PRD v6.1 §5.1.2 + §6).

Rationale
---------
With max_seq_length raised to 1024 (history budget 400 + completion budget
400 + system tokens), we need to verify that a 5-step training pass fits in
the 32 GB envelope on V100-32GB and the 24 GB envelope on RTX 4090. OOM here
would force us back to G=4 or max_seq_length=768.

Test plan
---------
    1. Build the trainer at the production config (G=8, max_seq=1024).
    2. Run 5 training steps.
    3. Sample peak VRAM via `torch.cuda.max_memory_allocated()`.
    4. Compare to per-device budgets:
        - RTX 4090 (24 GB): peak < 22 GB
        - V100-32GB (32 GB): peak < 30 GB
       Detect device via `torch.cuda.get_device_properties(0).total_memory`.

Output
------
    JSON at logs/preflight/08_max_seq_len_smoke.json with the peak memory
    figure, the device, and the budget that applied.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from configs.stack_versions import assert_versions  # noqa: E402
from utils.seed import seed_all  # noqa: E402


def _device_budget_gb() -> tuple[str, float]:
    """Return (device_name, max_allowed_peak_gb) based on detected device."""
    if not torch.cuda.is_available():
        return ("cpu", 0.0)
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    name = props.name.lower()
    # RTX 4090 = 24 GB; V100 = 32 GB; A100 = 40/80 GB.
    if total_gb < 25:
        return (props.name, 22.0)
    elif total_gb < 35:
        return (props.name, 30.0)
    else:
        # A100 / H100 — generous budget; mainly a sanity check.
        return (props.name, total_gb - 4.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out",
                        default="logs/preflight/08_max_seq_len_smoke.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "device": None,
        "device_total_gb": None,
        "peak_alloc_gb": None,
        "budget_gb": None,
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_versions()
        cfg = Config.from_yaml(args.config)
        assert cfg.max_seq_length == 1024, (
            "preflight 08 expects max_seq_length=1024 in the config under test."
        )
        seed_all(cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        from training.train import build_smoke_trainer  # noqa: E402

        trainer, _ = build_smoke_trainer(cfg, num_steps=5)
        trainer.train()

        device_name, budget_gb = _device_budget_gb()
        out["device"] = device_name
        out["budget_gb"] = budget_gb
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out["device_total_gb"] = props.total_memory / (1024 ** 3)
            peak_bytes = torch.cuda.max_memory_allocated()
            out["peak_alloc_gb"] = peak_bytes / (1024 ** 3)
            out["passed"] = out["peak_alloc_gb"] < budget_gb
        else:
            out["passed"] = False
            out["errors"].append("no CUDA device — VRAM smoke vacuous")
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"],
                          "peak_gb": out["peak_alloc_gb"],
                          "budget_gb": out["budget_gb"]}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
