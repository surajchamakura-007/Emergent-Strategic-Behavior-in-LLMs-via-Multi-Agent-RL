"""Preflight 07 — Episode horizon distribution check (PRD §6, D5+D8).

Rationale
---------
Episode lengths are drawn from a geometric distribution with continuation
probability p=0.95 (D5), truncated at cap=60 rounds (D8 v6). Acceptance
criteria (PRD v5 §6):
    - mean ≈ 18.5
    - std  ≈ 14
    - truncation rate ≈ 8% (PRD v6.1 with cap=60 brings this lower, ~5%)

If the env's coin-flip is wrong (e.g., we accidentally invert p, or the cap
fires early), the empirical histogram will be off.

Test plan
---------
    1. Simulate 5000 episodes via `simulate_episode_lengths`.
    2. Compute mean, std, truncation rate.
    3. Compare to the analytic targets for geometric(0.05) truncated at 60:
        - E[T] = (1 - (1-p)^N) / p ≈ 19.06 for N=60
        - std  ≈ 14.5
        - P(T = 60) = (1-p)^59 ≈ 0.05

Output
------
    JSON at logs/preflight/07_episode_dist_smoke.json with the empirical
    moments + the analytic targets and pass/fail.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.prisoners_dilemma import simulate_episode_lengths  # noqa: E402


def _analytic_geom_truncated_mean(p: float, N: int) -> float:
    """E[min(T, N)] for T ~ Geometric(p) starting at 1.

    closed form: E[min(T, N)] = (1 - (1-p)^N) / p
    """
    return (1.0 - (1.0 - p) ** N) / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=0.95)
    parser.add_argument("--cap", type=int, default=60)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--out",
                        default="logs/preflight/07_episode_dist_smoke.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "n_episodes": args.n,
        "p": args.p,
        "cap": args.cap,
        "empirical_mean": None,
        "empirical_std": None,
        "empirical_trunc_rate": None,
        "analytic_mean": None,
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        # geometric continuation: stop prob = 1-p
        stop_p = 1.0 - args.p
        lengths = simulate_episode_lengths(
            n=args.n, stop_prob=stop_p, cap=args.cap, seed=0,
        )
        m = statistics.fmean(lengths)
        s = statistics.pstdev(lengths)
        trunc = sum(1 for L in lengths if L >= args.cap) / len(lengths)
        analytic_mean = _analytic_geom_truncated_mean(stop_p, args.cap)

        out["empirical_mean"] = m
        out["empirical_std"] = s
        out["empirical_trunc_rate"] = trunc
        out["analytic_mean"] = analytic_mean

        # Targets (PRD v6.1 monitoring thresholds): mean ~19.1, std ~14.5,
        # trunc ~5%. Allow ±10% tolerances.
        out["passed"] = (
            abs(m - analytic_mean) / analytic_mean < 0.05
            and 12.0 < s < 17.0
            and 0.02 < trunc < 0.08
        )
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"],
                          "mean": out["empirical_mean"],
                          "std": out["empirical_std"],
                          "trunc": out["empirical_trunc_rate"]}, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
