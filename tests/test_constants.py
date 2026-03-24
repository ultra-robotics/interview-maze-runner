from __future__ import annotations

import pytest

from maze_runner.constants import DEFAULT_SIZE, MazeConfig


def test_maze_config_defaults() -> None:
    c = MazeConfig.defaults()
    assert c.size == DEFAULT_SIZE
    assert c.random_pct == 55


def test_maze_config_validation() -> None:
    with pytest.raises(ValueError):
        MazeConfig(size=4)
    with pytest.raises(ValueError):
        MazeConfig(random_pct=-1)
