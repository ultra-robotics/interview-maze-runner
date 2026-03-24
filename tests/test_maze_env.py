from __future__ import annotations

import numpy as np

from maze_runner.constants import MazeConfig, observation_channels
from maze_runner.env.maze_env import MazeEnv


def test_reset_step_shapes(small_config: MazeConfig) -> None:
    env = MazeEnv(small_config)
    obs, info = env.reset(seed=1)
    assert obs.shape == (small_config.size, small_config.size, observation_channels())
    assert obs.dtype == np.float32
    assert "goal" in info


def test_truncation_max_steps() -> None:
    cfg = MazeConfig(size=10, random_pct=30, max_steps=3)
    env = MazeEnv(cfg)
    env.reset(seed=0)
    truncated = False
    for _ in range(10):
        obs, _r, term, trunc, _ = env.step(0)
        if trunc:
            truncated = True
            break
        if term:
            break
    assert truncated


def test_restart_same_layout(small_config: MazeConfig) -> None:
    env = MazeEnv(small_config)
    obs1, _ = env.reset(seed=99)
    grid1 = env.layout.grid.copy() if env.layout else None
    obs2, _ = env.reset(seed=99, options={"regenerate": False})
    assert grid1 is not None and env.layout is not None
    assert np.array_equal(grid1, env.layout.grid)
    assert obs1.shape == obs2.shape
