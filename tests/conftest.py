"""Shared fixtures."""

from __future__ import annotations

import pytest

from maze_runner.constants import MazeConfig


@pytest.fixture
def small_config() -> MazeConfig:
    return MazeConfig(size=12, random_pct=40, max_steps=500)
