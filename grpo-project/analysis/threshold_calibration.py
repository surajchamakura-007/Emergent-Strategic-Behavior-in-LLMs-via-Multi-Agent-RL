"""Calibrate the advantage-collapse threshold from the existing 250-step run.

Authority: Implementation Map §6 (closes issue #5).

Background (Map §6). PRD v6.1 mentions a `0.10` threshold "scaled
accordingly" from an E[T]=10 calibration. With D5+D8 (E[T]≈19.1), episode
rewards roughly double, so absolute advantage values double — the threshold
needs to be re-derived from data.

Method. Pull the per-step `advantage_mean_abs` history from the existing
250-step T=5 run on W&B. That run did NOT collapse (it stabilized at
coop ~0.49 with healthy advantage variance throughout), so its
`advantage_mean_abs` distribution defines the LOWER edge of the
"training is alive" regime. The 5th-percentile is the collapse threshold.

Output. `configs/calibrated_threshold.json`, read by `train.py` at startup
via `configs.config.load_calibrated_threshold`.

Acceptance (Map §6). Threshold should land in [0.03, 0.20]; outside that,
inspect the source run before proceeding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def pull_wandb_history(
    run_id: str,
    metric: str = "advantage_mean_abs",
    project: str = "grpo-social-dilemmas",
    entity: str | None = None,
) -> np.ndarray:
    """Pull a single metric series from a W&B run.

    Returns the metric as a 1-D numpy array, in step order.
    """
    import wandb
    api = wandb.Api()
    qualified = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(qualified)
    rows = run.history(keys=[metric], pandas=False)
    series = [r[metric] for r in rows if metric in r and r[metric] is not None]
    if not series:
        raise RuntimeError(f"Metric {metric!r} not found in run {qualified}")
    return np.asarray(series, dtype=np.float64)


def calibrate_threshold(
    history: Iterable[float] | np.ndarray,
    *,
    target_quantile: float = 0.05,
    warmup_steps_to_drop: int = 30,
    floor: float = 0.03,
    ceiling: float = 0.20,
) -> dict:
    """Compute the advantage-collapse threshold and return the full record.

    Args:
        history: Per-step `advantage_mean_abs` from the calibration run.
        target_quantile: 5th-percentile by default — below this is collapse.
        warmup_steps_to_drop: Skip the early steps; not representative of
            the stable regime.
        floor / ceiling: Sanity bounds on the resulting threshold.

    Returns:
        A JSON-serializable dict with the threshold and provenance.
    """
    arr = np.asarray(list(history), dtype=np.float64)
    if arr.size <= warmup_steps_to_drop:
        raise ValueError(
            f"Calibration history has {arr.size} points; need > "
            f"{warmup_steps_to_drop} after warmup drop"
        )
    stable = arr[warmup_steps_to_drop:]
    p5 = float(np.quantile(stable, target_quantile))
    threshold = math.floor(p5 * 100) / 100  # round down to 0.01 for legibility

    record = {
        "threshold": threshold,
        "p5_raw": p5,
        "target_quantile": target_quantile,
        "stable_window_start_step": warmup_steps_to_drop,
        "stable_window_n_samples": int(stable.size),
        "stable_mean": float(np.mean(stable)),
        "stable_std": float(np.std(stable)),
        "stable_min": float(np.min(stable)),
        "stable_max": float(np.max(stable)),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not (floor <= threshold <= ceiling):
        record["WARNING"] = (
            f"threshold={threshold} outside expected [{floor}, {ceiling}]; "
            f"inspect source run before training"
        )
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="W&B run ID of the 250-step T=5 calibration run")
    ap.add_argument("--project", default="grpo-social-dilemmas")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--out", default="configs/calibrated_threshold.json")
    ap.add_argument("--target-quantile", type=float, default=0.05)
    ap.add_argument("--warmup-drop", type=int, default=30)
    args = ap.parse_args()

    print(f"[threshold_calibration] Pulling W&B run {args.run_id} ...")
    history = pull_wandb_history(
        args.run_id,
        metric="advantage_mean_abs",
        project=args.project,
        entity=args.entity,
    )
    print(f"[threshold_calibration] Got {history.size} datapoints")

    record = calibrate_threshold(
        history,
        target_quantile=args.target_quantile,
        warmup_steps_to_drop=args.warmup_drop,
    )
    record["source_run"] = f"{args.entity + '/' if args.entity else ''}{args.project}/{args.run_id}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[threshold_calibration] Wrote {out_path}")
    print(json.dumps(record, indent=2))
    if "WARNING" in record:
        print(f"\n*** WARNING: {record['WARNING']} ***", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
