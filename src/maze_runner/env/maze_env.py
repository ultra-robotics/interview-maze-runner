"""Gymnasium environment: full-world observation, discrete moves."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from maze_runner.constants import (
    ACTION_DIM,
    DR,
    DC,
    INVALID_MOVE_PENALTY,
    MazeConfig,
    STEP_PENALTY,
    observation_channels,
)
from maze_runner.maze.generator import MazeGenerator, MazeLayout


class MazeEnv(gym.Env):
    """Maze navigation with injected `MazeConfig` and optional layout reuse."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: MazeConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config or MazeConfig.defaults()
        self.render_mode = render_mode
        self._generator = MazeGenerator(self._config)

        self._layout: MazeLayout | None = None
        self._agent: tuple[int, int] = (0, 0)
        self._steps = 0

        h = w = self._config.size
        c = observation_channels()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(h, w, c), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

    @property
    def maze_config(self) -> MazeConfig:
        return self._config

    @property
    def layout(self) -> MazeLayout | None:
        return self._layout

    @property
    def agent(self) -> tuple[int, int]:
        return self._agent

    def observation(self) -> np.ndarray:
        """Current full-world observation (walls, agent, goal channels)."""
        return self._build_obs()

    def _build_obs(self) -> np.ndarray:
        assert self._layout is not None
        g = self._layout.grid
        h, w = g.shape
        c = observation_channels()
        obs = np.zeros((h, w, c), dtype=np.float32)
        obs[:, :, 0] = (g == 0).astype(np.float32)
        ar, ac = self._agent
        obs[ar, ac, 1] = 1.0
        gr, gc = self._layout.goal
        obs[gr, gc, 2] = 1.0
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        regenerate = bool(opts.get("regenerate", True))

        if regenerate or self._layout is None:
            gen_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
            self._layout = self._generator.generate(int(gen_seed))
        self._agent = self._layout.start
        self._steps = 0
        obs = self._build_obs()
        info = {
            "start": self._layout.start,
            "goal": self._layout.goal,
            "maze_seed": self._layout.seed,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._layout is not None
        g = self._layout.grid
        h, w = g.shape
        r, c = self._agent
        dr, dc = DR[action], DC[action]
        nr, nc = r + dr, c + dc

        truncated = False
        terminated = False
        reward = float(STEP_PENALTY)

        if not (0 <= nr < h and 0 <= nc < w) or g[nr, nc] == 0:
            reward += float(INVALID_MOVE_PENALTY)
        else:
            self._agent = (nr, nc)

        self._steps += 1

        if self._agent == self._layout.goal:
            terminated = True
            reward = 0.0
        elif self._steps >= self._config.max_steps:
            truncated = True

        obs = self._build_obs()
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info
