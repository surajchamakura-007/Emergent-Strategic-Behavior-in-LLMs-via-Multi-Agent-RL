"""Tier B Elo tournament — round-robin payoff comparison across the 4 final
adapters and GPT-4o-mini.

Authority: PRD v6.1 §8.2 (Tier B). Lighter than Melting Pot 2.0; gives a
single ranking number per adapter that aggregates pairwise win rates.

Method
------
For each pair (A, B) of agents, run M episodes where A plays the agent
role and B plays the opponent role. (A and B are LoRA adapters loaded
into a vLLM engine; for GPT-4o-mini we use the API.) Score per episode:

    win = 1 if A's total reward > B's total reward
    loss = 0 if A's total reward < B's total reward
    draw = 0.5 if equal

We use the standard Elo update with K=32, starting at 1000:
    E_a = 1 / (1 + 10^((R_b - R_a)/400))
    R_a' = R_a + K * (S - E_a)

We process matches in a fixed order (sorted) so the tournament is
reproducible per seed.

This is a separate codepath from `evaluation/eval.py` because both agents
are LLMs being generated against each other — neither is a fixed
strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from envs.history import Round  # noqa: E402
from envs.prisoners_dilemma import PrisonersDilemmaEnv  # noqa: E402
from evaluation.eval import _GenerationBackend  # noqa: E402
from training.prompt_builder import build_prompt  # noqa: E402
from training.reward import parse_completion  # noqa: E402


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------

@dataclass
class _Player:
    name: str
    rating: float = 1000.0
    games: int = 0


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update(player: _Player, score: float, expected: float, k: float = 32.0) -> None:
    player.rating += k * (score - expected)
    player.games += 1


# ---------------------------------------------------------------------------
# Match: agent A vs agent B
# ---------------------------------------------------------------------------

def _play_match(
    cfg: Config,
    backend_a: _GenerationBackend,
    backend_b: Optional[_GenerationBackend],
    *,
    n_episodes: int,
    seed: int,
) -> tuple[float, float]:
    """Returns (avg_reward_A, avg_reward_B) over n_episodes.

    `backend_b is None` means agent B is GPT-4o-mini (handled separately).
    """
    if backend_b is None:
        from evaluation.opponents import GPT4oMiniOpponent
        opp = GPT4oMiniOpponent()
    else:
        opp = None

    tokenizer_a = backend_a.tokenizer
    total_a = 0.0
    total_b = 0.0
    rounds_a = 0
    rounds_b = 0

    for ep in range(n_episodes):
        env = PrisonersDilemmaEnv(
            payoffs=cfg.payoffs, p=cfg.geometric_p,
            cap=cfg.episode_cap, seed=seed * 10000 + ep,
        )
        history_a: list[Round] = []
        history_b: list[Round] = []  # mirror of history_a from B's POV

        if opp is not None:
            opp.reset(seed=seed * 10000 + ep)

        while True:
            # A's turn
            prompt_a, _ = build_prompt(
                history=history_a, round_idx=len(history_a) + 1,
                payoffs=cfg.payoffs, p=cfg.geometric_p,
                tokenizer=tokenizer_a,
                history_token_budget=cfg.history_token_budget,
                keep_last_k=cfg.keep_last_k,
            )
            text_a = backend_a.generate(prompt_a, temperature=cfg.sampling_temp_default,
                                        max_tokens=cfg.max_completion_length,
                                        seed=seed * 31 + len(history_a))
            parsed_a = parse_completion(text_a)
            action_a = parsed_a[1] if parsed_a else "D"

            # B's turn
            if backend_b is not None:
                prompt_b, _ = build_prompt(
                    history=history_b, round_idx=len(history_b) + 1,
                    payoffs=cfg.payoffs, p=cfg.geometric_p,
                    tokenizer=backend_b.tokenizer,
                    history_token_budget=cfg.history_token_budget,
                    keep_last_k=cfg.keep_last_k,
                )
                text_b = backend_b.generate(
                    prompt_b, temperature=cfg.sampling_temp_default,
                    max_tokens=cfg.max_completion_length,
                    seed=seed * 37 + len(history_b),
                )
                parsed_b = parse_completion(text_b)
                action_b = parsed_b[1] if parsed_b else "D"
            else:
                action_b = opp.act(history_b)

            step = env.step(action_a, action_b)

            # B's payoff with the IPD matrix:
            #   if (a,b)=(C,C): both get R; (D,D): P; (C,D): a=S, b=T; (D,C): a=T, b=S.
            # Env returns step.my_payoff for A; compute B's payoff from the matrix.
            if action_a == "C" and action_b == "C":
                pay_a, pay_b = cfg.R, cfg.R
            elif action_a == "D" and action_b == "D":
                pay_a, pay_b = cfg.P, cfg.P
            elif action_a == "C" and action_b == "D":
                pay_a, pay_b = cfg.S, cfg.T
            else:
                pay_a, pay_b = cfg.T, cfg.S

            total_a += pay_a if parsed_a else 0.0
            total_b += pay_b
            rounds_a += 1
            rounds_b += 1
            history_a.append(Round(my_action=action_a, opp_action=action_b,
                                   my_payoff=pay_a))
            history_b.append(Round(my_action=action_b, opp_action=action_a,
                                   my_payoff=pay_b))

            if step.done:
                break

    return (total_a / max(1, rounds_a), total_b / max(1, rounds_b))


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

def run_elo_tournament(
    cfg_template: Config,
    adapter_paths: dict[str, str],
    *,
    include_gpt4o: bool = True,
    n_episodes: int = 10,
    seed: int = 0,
) -> list[_Player]:
    """Adapters are {name: path}. cfg_template provides game settings."""
    backends: dict[str, Optional[_GenerationBackend]] = {}
    for name, path in adapter_paths.items():
        backends[name] = _GenerationBackend(cfg_template, adapter_path=path)
    if include_gpt4o:
        backends["GPT-4o-mini"] = None  # special-cased in _play_match

    players: dict[str, _Player] = {n: _Player(n) for n in backends}

    pairs = sorted(combinations(backends, 2))  # deterministic pairing
    for a, b in pairs:
        avg_a, avg_b = _play_match(
            cfg_template, backends[a], backends[b],
            n_episodes=n_episodes, seed=seed,
        )
        if avg_a > avg_b + 1e-9:
            score_a = 1.0
        elif avg_a < avg_b - 1e-9:
            score_a = 0.0
        else:
            score_a = 0.5
        score_b = 1.0 - score_a

        e_a = _expected_score(players[a].rating, players[b].rating)
        e_b = 1.0 - e_a
        _update(players[a], score_a, e_a)
        _update(players[b], score_b, e_b)

    return sorted(players.values(), key=lambda p: -p.rating)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Used for game settings only. T is taken from config.")
    parser.add_argument("--adapters", required=True,
                        help="JSON dict {name: path} for each adapter to enter.")
    parser.add_argument("--include-gpt4o", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs/eval/elo_tournament.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    adapters = json.loads(args.adapters) if args.adapters.startswith("{") \
        else json.loads(Path(args.adapters).read_text())

    leaderboard = run_elo_tournament(
        cfg, adapters, include_gpt4o=args.include_gpt4o,
        n_episodes=args.episodes, seed=args.seed,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([{"name": p.name, "rating": p.rating, "games": p.games}
                   for p in leaderboard], f, indent=2)

    for p in leaderboard:
        print(f"{p.name:<32} {p.rating:7.1f}   ({p.games} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
