"""`uv run stats` — bulk headless evaluation."""

from __future__ import annotations

import argparse

from maze_runner.constants import DEFAULT_RANDOM, DEFAULT_SIZE, MazeConfig
from maze_runner.eval.harness import run_stats


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the baseline policy on many mazes.")
    p.add_argument("--episodes", type=int, default=100, help="Number of mazes to evaluate")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed offset for episodes")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--random", type=int, default=DEFAULT_RANDOM, dest="random_pct")
    args = p.parse_args()

    cfg = MazeConfig(size=args.size, random_pct=args.random_pct)
    run_stats(config=cfg, episodes=args.episodes, base_seed=args.seed)


if __name__ == "__main__":
    main()
