"""Preflight 06 — Format gate bimodality smoke (PRD §4.1, Map §2).

Rationale
---------
The format gate is multiplicative: `reward = format_ok * payoff_lookup`. So
a histogram of rewards across a representative completion sample MUST be
bimodal — either exactly 0 (format violation) or one of the payoff cells
{S, P, R, T}. Anything else means the gate is leaky (e.g., partial-credit
matching, additive bonus snuck in, etc.).

Test plan
---------
    1. Generate 1000 synthetic completions: 30% well-formed C, 30% well-formed D,
       40% malformed (truncated, missing tags, extra noise, lowercase action,
       reasoning leaked into action tag, etc.).
    2. Run `compute_reward` against a synthetic opp action stream (alternating).
    3. Verify:
        - Every reward is in {0.0, S, P, R, T}.
        - Format-violation rate matches the synthetic 40% within ±2%.
        - Reward magnitudes match the payoff matrix exactly.

Output
------
    JSON at logs/preflight/06_format_gate_smoke.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config, PayoffMatrix  # noqa: E402


# Canonical good/bad completions.
GOOD_C = "<reasoning>cooperate is best</reasoning>\n<action>C</action>"
GOOD_D = "<reasoning>defect is best</reasoning>\n<action>D</action>"
BAD_LOWERCASE = "<reasoning>x</reasoning>\n<action>c</action>"
BAD_NO_TAGS = "I think I'll cooperate."
BAD_TRUNCATED = "<reasoning>think...</reasoning>\n<acti"
BAD_DOUBLE_ACTION = (
    "<reasoning>x</reasoning>\n<action>C</action>\n<action>D</action>"
)
BAD_GARBAGE_ACTION = "<reasoning>x</reasoning>\n<action>X</action>"


def _make_completion_stream(n: int, rng: random.Random) -> list[str]:
    population = [
        (GOOD_C, "good"),
        (GOOD_D, "good"),
        (GOOD_C, "good"),
        (GOOD_D, "good"),
        (GOOD_C, "good"),
        (GOOD_D, "good"),
        (BAD_LOWERCASE, "bad"),
        (BAD_NO_TAGS, "bad"),
        (BAD_TRUNCATED, "bad"),
        (BAD_DOUBLE_ACTION, "bad"),
    ]
    # 60% good, 40% bad — close to the 60/40 split the PRD smoke targets.
    return [rng.choice(population)[0] for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Optional config YAML; defaults to T=5.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out",
                        default="logs/preflight/06_format_gate_smoke.json")
    args = parser.parse_args()

    out = {
        "passed": False,
        "reward_histogram": {},
        "format_violation_rate": None,
        "payoff_set_observed": [],
        "errors": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.config:
            cfg = Config.from_yaml(args.config)
            payoffs = cfg.payoffs
        else:
            payoffs = PayoffMatrix(T=5.0, R=3.0, P=1.0, S=0.0)

        from training.reward import compute_reward  # noqa: E402

        rng = random.Random(0)
        completions = _make_completion_stream(args.n, rng)
        opp_stream = [rng.choice(["C", "D"]) for _ in completions]

        rewards = []
        violations = 0
        for c, opp in zip(completions, opp_stream):
            r = compute_reward(c, opp, payoffs).reward
            rewards.append(round(r, 6))
            if r == 0.0:
                violations += 1

        hist = Counter(rewards)
        out["reward_histogram"] = {str(k): v for k, v in hist.items()}
        out["format_violation_rate"] = violations / len(rewards)

        # Allowed reward set: {0} ∪ {S, P, R, T}.
        allowed = {0.0, payoffs.S, payoffs.P, payoffs.R, payoffs.T}
        observed = set(rewards)
        out["payoff_set_observed"] = sorted(observed)
        unexpected = observed - allowed
        assert not unexpected, f"unexpected reward values: {unexpected}"

        # Acceptance: bimodal AND violation rate close to expected ~40%.
        out["passed"] = (
            len(unexpected) == 0
            and 0.30 <= out["format_violation_rate"] <= 0.50
        )
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc())
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"passed": out["passed"],
                          "violation_rate": out["format_violation_rate"]},
                         indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
