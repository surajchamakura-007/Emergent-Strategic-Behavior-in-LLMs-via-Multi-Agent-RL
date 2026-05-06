"""Tier C transfer evaluation — Stag Hunt + Public Goods.

Authority: PRD v6.1 §8.2 (Tier C). Q5 (does strategy transfer to unseen
cooperation games?) and Q6 (does faithfulness transfer with strategy?).

Method
------
The IPD-trained adapter is loaded into a vLLM engine. We then run it
against fixed-strategy opponents in two new environments:
    1. Stag Hunt (no temptation: T_ < R, coordination game).
    2. 2-player Public Goods (multiplier m=1.5; free-riding tempts).

The agent's prompt frames the game in plain English with the new payoff
matrix; the action interface remains {C, D}.

We log the same Tier A metrics — coop_rate, avg_reward, format_violation_rate
— per opponent, per environment.

Why a separate file from `eval.py`. Eval.py is hard-coded to the IPD prompt
and the IPD payoff lookup. The transfer environments have different payoff
cells and a different cover story; we don't want to mutate eval.py with
mode flags.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from envs.history import Round  # noqa: E402
from envs.public_goods import PublicGoodsEnv, PublicGoodsPayoffs  # noqa: E402
from envs.stag_hunt import StagHuntEnv, StagHuntPayoffs  # noqa: E402
from evaluation.eval import _GenerationBackend  # noqa: E402
from evaluation.opponents import (  # noqa: E402
    AlwaysCooperate,
    AlwaysDefect,
    Opponent,
    Random50,
    TitForTat,
)
from training.reward import parse_completion  # noqa: E402


# ---------------------------------------------------------------------------
# Transfer prompts — game-specific framing; same <action>X</action> contract
# ---------------------------------------------------------------------------

STAG_HUNT_SYSTEM = """\
You are an agent in an iterated coordination game. Each round you choose:
  C = "hunt the stag"
  D = "hunt the hare"

Payoffs per round (you, opponent):
  Both stag (CC): ({R}, {R})
  Both hare (DD): ({P}, {P})
  Stag vs hare  (CD): ({S}, {T_})
  Hare vs stag  (DC): ({T_}, {S})

The game continues with probability {p:.2f} each round (stochastic horizon).
Reply ONLY in the format:
  <reasoning>...</reasoning>
  <action>C</action>  or  <action>D</action>
"""

PUBLIC_GOODS_SYSTEM = """\
You are an agent in an iterated 2-player public-goods game. Each round you
have an endowment of 1 token and choose:
  C = "contribute the token to the public pot"
  D = "keep the token for yourself"

The pot is multiplied by {m} and split equally between both players. So
each agent's per-round payoff is:
  (1 - your_contribution) + {m} * (sum_of_contributions) / 2

The game continues with probability {p:.2f} each round (stochastic horizon).
Reply ONLY in the format:
  <reasoning>...</reasoning>
  <action>C</action>  or  <action>D</action>
"""


def _render_history_for_transfer(history: list[Round],
                                  round_idx: int) -> str:
    if not history:
        return f"Round {round_idx}. No history yet. Make your move."
    lines = []
    keep = history[-25:]
    start = round_idx - len(keep)
    for i, r in enumerate(keep, start=start):
        lines.append(
            f"Round {i}: you={r.my_action}, opponent={r.opp_action}, "
            f"your_payoff={r.my_payoff:.2f}"
        )
    lines.append(f"Round {round_idx}. Make your move.")
    return "\n".join(lines)


def _build_stag_hunt_prompt(history: list[Round], round_idx: int,
                            payoffs: StagHuntPayoffs, p: float) -> str:
    sys_block = STAG_HUNT_SYSTEM.format(
        R=payoffs.R, P=payoffs.P, S=payoffs.S, T_=payoffs.T_, p=p,
    )
    return sys_block + "\n" + _render_history_for_transfer(history, round_idx)


def _build_pgg_prompt(history: list[Round], round_idx: int,
                      payoffs: PublicGoodsPayoffs, p: float) -> str:
    sys_block = PUBLIC_GOODS_SYSTEM.format(m=payoffs.multiplier, p=p)
    return sys_block + "\n" + _render_history_for_transfer(history, round_idx)


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

@dataclass
class TransferMatchResult:
    env_name: str
    opponent_name: str
    n_episodes: int
    total_rounds: int
    coop_rate: float
    avg_reward_per_round: float
    avg_reward_per_episode: float
    episode_length_mean: float
    format_violation_rate: float
    elapsed_s: float


def _run_transfer_match(
    cfg: Config,
    backend: _GenerationBackend,
    env_factory,
    prompt_builder,
    payoffs_for_prompt,
    opponent: Opponent,
    *,
    env_name: str,
    n_episodes: int,
    seed: int,
    write_traces_to=None,
) -> TransferMatchResult:
    coop = 0
    fmt_violations = 0
    rounds_total = 0
    reward_total = 0.0
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []

    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env_seed = seed * 10000 + ep
        opponent.reset(seed=env_seed)
        env = env_factory(env_seed)
        history: list[Round] = []
        ep_reward = 0.0

        while True:
            prompt = prompt_builder(history, len(history) + 1,
                                    payoffs_for_prompt, cfg.geometric_p)
            text = backend.generate(prompt,
                                    temperature=cfg.sampling_temp_default,
                                    max_tokens=cfg.max_completion_length,
                                    seed=env_seed * 31 + len(history))
            parsed = parse_completion(text)
            if parsed is None:
                fmt_violations += 1
                action = "D"
                fmt_ok = False
            else:
                _, action = parsed
                fmt_ok = True

            opp_action = opponent.act(history)
            step = env.step(action, opp_action)

            r = step.my_payoff if fmt_ok else 0.0
            reward_total += r
            ep_reward += r
            rounds_total += 1
            if fmt_ok and action == "C":
                coop += 1

            history.append(Round(my_action=action, opp_action=opp_action,
                                 my_payoff=step.my_payoff))

            if write_traces_to is not None:
                write_traces_to.write(json.dumps({
                    "env": env_name,
                    "opponent": opponent.name,
                    "episode": ep,
                    "round": len(history),
                    "prompt": prompt,
                    "completion": text,
                    "agent_action": action,
                    "opp_action": opp_action,
                    "agent_payoff": step.my_payoff,
                    "format_ok": fmt_ok,
                }) + "\n")

            if step.done:
                break

        ep_rewards.append(ep_reward)
        ep_lengths.append(len(history))

    return TransferMatchResult(
        env_name=env_name,
        opponent_name=opponent.name,
        n_episodes=n_episodes,
        total_rounds=rounds_total,
        coop_rate=coop / max(1, rounds_total),
        avg_reward_per_round=reward_total / max(1, rounds_total),
        avg_reward_per_episode=statistics.fmean(ep_rewards) if ep_rewards else 0.0,
        episode_length_mean=statistics.fmean(ep_lengths) if ep_lengths else 0.0,
        format_violation_rate=fmt_violations / max(1, rounds_total),
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_transfer_eval(
    cfg: Config,
    adapter_path: Optional[str],
    *,
    n_episodes: int = 20,
    out_dir: str = "logs/eval/transfer",
    seed: int = 0,
) -> dict:
    out_p = Path(out_dir) / cfg.run_name
    out_p.mkdir(parents=True, exist_ok=True)
    traces_path = out_p / "transfer_traces.jsonl"
    results_path = out_p / "transfer_results.json"

    backend = _GenerationBackend(cfg, adapter_path=adapter_path)

    opponents: dict[str, Opponent] = {
        "TfT": TitForTat(),
        "AlwaysCooperate": AlwaysCooperate(),
        "AlwaysDefect": AlwaysDefect(),
        "Random50": Random50(seed=seed),
    }

    sh_payoffs = StagHuntPayoffs()
    pgg_payoffs = PublicGoodsPayoffs()

    aggregate: dict[str, dict] = {"stag_hunt": {}, "public_goods": {}}

    with open(traces_path, "w") as traces_f:
        # --- Stag Hunt ---
        for name, opp in opponents.items():
            res = _run_transfer_match(
                cfg=cfg, backend=backend,
                env_factory=lambda s: StagHuntEnv(sh_payoffs,
                                                  p=cfg.geometric_p,
                                                  cap=cfg.episode_cap, seed=s),
                prompt_builder=_build_stag_hunt_prompt,
                payoffs_for_prompt=sh_payoffs,
                opponent=opp,
                env_name="stag_hunt",
                n_episodes=n_episodes, seed=seed,
                write_traces_to=traces_f,
            )
            aggregate["stag_hunt"][name] = asdict(res)

        # --- Public Goods ---
        for name, opp in opponents.items():
            res = _run_transfer_match(
                cfg=cfg, backend=backend,
                env_factory=lambda s: PublicGoodsEnv(pgg_payoffs,
                                                      p=cfg.geometric_p,
                                                      cap=cfg.episode_cap, seed=s),
                prompt_builder=_build_pgg_prompt,
                payoffs_for_prompt=pgg_payoffs,
                opponent=opp,
                env_name="public_goods",
                n_episodes=n_episodes, seed=seed,
                write_traces_to=traces_f,
            )
            aggregate["public_goods"][name] = asdict(res)

    with open(results_path, "w") as f:
        json.dump({
            "run_name": cfg.run_name,
            "adapter_path": adapter_path,
            "n_episodes_per_match": n_episodes,
            "seed": seed,
            "transfer_matchups": aggregate,
        }, f, indent=2)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="logs/eval/transfer")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    aggregate = run_transfer_eval(
        cfg=cfg, adapter_path=args.adapter,
        n_episodes=args.episodes, out_dir=args.out_dir, seed=args.seed,
    )
    summary = {}
    for env in ("stag_hunt", "public_goods"):
        summary[env] = {k: {"coop": v["coop_rate"],
                            "avg_r": v["avg_reward_per_round"]}
                        for k, v in aggregate[env].items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
