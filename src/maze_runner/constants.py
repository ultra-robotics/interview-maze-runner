"""Shared defaults, validation, and UI ranges for maze generation."""

from __future__ import annotations

from dataclasses import dataclass

# README defaults
DEFAULT_SIZE = 16
DEFAULT_RANDOM = 55

# Viz timing (README)
DEFAULT_STEP_DELAY_S = 0.2

# `uv run viz` sets these in the environment so uvicorn --reload workers inherit CLI flags.
VIZ_ENV_SIZE = "MAZE_RUNNER_VIZ_SIZE"
VIZ_ENV_RANDOM_PCT = "MAZE_RUNNER_VIZ_RANDOM_PCT"
VIZ_ENV_STEP_DELAY = "MAZE_RUNNER_VIZ_STEP_DELAY"
VIZ_ENV_MAX_STEPS = "MAZE_RUNNER_VIZ_MAX_STEPS"

# Sliders / CLI bounds
MIN_SIZE = 8
MAX_SIZE = 64
MIN_RANDOM = 0
MAX_RANDOM = 100

# Gymnasium
DEFAULT_MAX_STEPS = 10_000

# Invalid move penalty (README gap)
STEP_PENALTY = -1.0
INVALID_MOVE_PENALTY = -0.1

# Actions: 0 up, 1 right, 2 down, 3 left
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_DIM = 4

DR = (-1, 0, 1, 0)
DC = (0, 1, 0, -1)


@dataclass(frozen=True, slots=True)
class MazeConfig:
    """Parameters passed from CLI, harness, env, and UI into the generator."""

    size: int = DEFAULT_SIZE
    random_pct: int = DEFAULT_RANDOM
    max_steps: int = DEFAULT_MAX_STEPS

    def __post_init__(self) -> None:
        if not (MIN_SIZE <= self.size <= MAX_SIZE):
            raise ValueError(f"size must be in [{MIN_SIZE}, {MAX_SIZE}], got {self.size}")
        if not (MIN_RANDOM <= self.random_pct <= MAX_RANDOM):
            raise ValueError(
                f"random_pct must be in [{MIN_RANDOM}, {MAX_RANDOM}], got {self.random_pct}"
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @classmethod
    def defaults(cls) -> MazeConfig:
        return cls()


def observation_channels() -> int:
    """Wall + agent + goal channels."""
    return 3
