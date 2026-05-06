"""D7 — sigmoid fits over training reward trajectory (PRD v6.1 §8 / D7).

Authority: PRD v6.1 D7 (sigmoid scaling fits).

Model
-----
For each training run we fit:

    R(C) = R0 + (A - R0) / (1 + (C_mid / C)^B)

where:
    R(C)  = average per-round reward at training step C
    R0    = baseline (untrained-Qwen) reward, fixed input
    A     = asymptotic ceiling
    B     = steepness (>0)
    C_mid = step at which R reaches the midpoint between R0 and A

Inputs
------
    - W&B run id (or local backfill JSON) for each Stage 1 training run
      that exposes per-step `reward` (or `rewards/mean`).
    - Fixed `R0` from the untrained-Qwen baseline (preflight 10).

Output
------
    JSON record per run + a combined CSV table for the writeup:
        {run_name, T, seed, R0, A, A_se, B, B_se, C_mid, C_mid_se,
         RMSE, n_points, c_final_over_c_mid, ceiling_extrapolated}

C_final / C_mid documents whether the run actually reached the fitted
ceiling. If C_final / C_mid < 1.0, the fit is *ceiling-extrapolated* and
should be reported with wide CIs rather than as a point estimate (PRD v4
§8 acceptance criteria, carried through v6.1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Model + fitter
# ---------------------------------------------------------------------------

def _sigmoid(C: np.ndarray, R0: float, A: float, B: float, C_mid: float) -> np.ndarray:
    """Khatri/Madaan-style asymptotic-approach sigmoid."""
    # Avoid division by zero / negative C.
    C = np.clip(C, 1e-9, None)
    return R0 + (A - R0) / (1.0 + (C_mid / C) ** B)


@dataclass
class FitRecord:
    run_name: str
    T: float
    seed: int
    R0: float
    A: float
    A_se: Optional[float]
    B: float
    B_se: Optional[float]
    C_mid: float
    C_mid_se: Optional[float]
    rmse: float
    n_points: int
    c_final: float
    c_mid_ratio: float
    ceiling_extrapolated: bool
    converged: bool
    notes: str = ""


def _fit_one(
    run_name: str,
    T: float,
    seed: int,
    steps: Sequence[float],
    rewards: Sequence[float],
    R0: float,
) -> FitRecord:
    from scipy.optimize import curve_fit
    steps = np.asarray(steps, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    if len(steps) < 5:
        return FitRecord(run_name=run_name, T=T, seed=seed, R0=R0,
                         A=float("nan"), A_se=None, B=float("nan"), B_se=None,
                         C_mid=float("nan"), C_mid_se=None,
                         rmse=float("nan"), n_points=len(steps),
                         c_final=float(steps[-1]) if len(steps) else 0.0,
                         c_mid_ratio=float("nan"),
                         ceiling_extrapolated=False, converged=False,
                         notes="too few points")

    # Initial guesses: A = max observed, B = 2, C_mid = midpoint of x range.
    p0 = [
        float(np.max(rewards)),
        2.0,
        float((steps[0] + steps[-1]) / 2.0),
    ]
    bounds = (
        [R0 - 10.0, 0.1, 1.0],          # A_lo, B_lo, Cmid_lo
        [max(R0 + 50.0, p0[0] + 50.0), 20.0, max(steps[-1] * 2.0, 1000.0)],  # hi
    )

    def f(C, A, B, C_mid):
        return _sigmoid(C, R0, A, B, C_mid)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(f, steps, rewards, p0=p0, bounds=bounds,
                                   maxfev=20000)
        A, B, C_mid = popt
        ses = np.sqrt(np.diag(pcov)) if pcov is not None else [None] * 3
        preds = f(steps, *popt)
        rmse = float(np.sqrt(np.mean((preds - rewards) ** 2)))
        c_final = float(steps[-1])
        ratio = c_final / max(C_mid, 1e-9)
        return FitRecord(
            run_name=run_name, T=T, seed=seed, R0=R0,
            A=float(A), A_se=float(ses[0]) if ses[0] is not None else None,
            B=float(B), B_se=float(ses[1]) if ses[1] is not None else None,
            C_mid=float(C_mid),
            C_mid_se=float(ses[2]) if ses[2] is not None else None,
            rmse=rmse, n_points=len(steps),
            c_final=c_final, c_mid_ratio=ratio,
            ceiling_extrapolated=ratio < 1.0,
            converged=True,
        )
    except Exception as e:  # noqa: BLE001
        return FitRecord(run_name=run_name, T=T, seed=seed, R0=R0,
                         A=float("nan"), A_se=None, B=float("nan"), B_se=None,
                         C_mid=float("nan"), C_mid_se=None,
                         rmse=float("nan"), n_points=len(steps),
                         c_final=float(steps[-1]),
                         c_mid_ratio=float("nan"),
                         ceiling_extrapolated=False, converged=False,
                         notes=f"fit failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_history_json(path: Path) -> tuple[list[float], list[float]]:
    """Load a per-step reward history JSON.

    Expected schema (matches what `analysis/threshold_calibration.py` writes):
        {"history": [{"step": int, "reward": float, ...}, ...]}
    Falls back to a flat list of {"step", "reward"} entries.
    """
    data = json.loads(path.read_text())
    rows = data["history"] if isinstance(data, dict) and "history" in data \
        else data
    steps, rewards = [], []
    for r in rows:
        if "step" not in r:
            continue
        for k in ("reward", "rewards/mean", "train/reward"):
            if k in r and r[k] is not None:
                steps.append(float(r["step"]))
                rewards.append(float(r[k]))
                break
    return steps, rewards


def _pull_wandb(run_id: str, project: str, entity: str) -> tuple[list[float], list[float]]:
    """Pull `reward` history from W&B. Mirrors threshold_calibration.pull_wandb_history."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(keys=["reward", "_step"], pandas=False)
    steps, rewards = [], []
    for h in history:
        if "_step" not in h or "reward" not in h:
            continue
        if h["reward"] is None:
            continue
        steps.append(float(h["_step"]))
        rewards.append(float(h["reward"]))
    return steps, rewards


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R0", type=float, required=True,
                        help="Untrained-Qwen baseline reward (e.g., from preflight 10).")
    parser.add_argument("--runs-json", default=None,
                        help="JSON with the per-run reward histories: "
                             "[{run_name, T, seed, history_path | wandb_run_id}, ...]")
    parser.add_argument("--wandb-project", default="grpo-social-dilemmas")
    parser.add_argument("--wandb-entity",
                        default="suraj_chamakura-university-of-california-berkeley")
    parser.add_argument("--out-json", default="analysis/fits.json")
    parser.add_argument("--out-csv", default="analysis/fits.csv")
    args = parser.parse_args()

    if not args.runs_json:
        raise SystemExit("--runs-json is required (manifest of runs to fit).")

    manifest = json.loads(Path(args.runs_json).read_text())
    fits: list[FitRecord] = []
    for entry in manifest:
        if "history_path" in entry:
            steps, rewards = _load_history_json(Path(entry["history_path"]))
        elif "wandb_run_id" in entry:
            steps, rewards = _pull_wandb(
                entry["wandb_run_id"],
                project=args.wandb_project,
                entity=args.wandb_entity,
            )
        else:
            raise ValueError(f"manifest entry missing source: {entry}")
        fits.append(_fit_one(
            run_name=entry["run_name"],
            T=float(entry.get("T", 5.0)),
            seed=int(entry.get("seed", 0)),
            steps=steps, rewards=rewards, R0=args.R0,
        ))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump([asdict(r) for r in fits], f, indent=2)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(fits[0]).keys()))
        w.writeheader()
        for r in fits:
            w.writerow(asdict(r))

    print(json.dumps([{"run": r.run_name, "A": r.A, "B": r.B,
                       "C_mid": r.C_mid, "RMSE": r.rmse,
                       "ceiling_extrapolated": r.ceiling_extrapolated}
                      for r in fits], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
