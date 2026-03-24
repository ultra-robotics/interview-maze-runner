from __future__ import annotations

import numpy as np

from maze_runner.constants import MazeConfig
from maze_runner.maze.generator import (
    DENSITY_CAP,
    MazeGenerator,
    _inner_perimeter_cells,
    opposite_rim_pair,
    outer_perimeter_cells,
    _would_join_existing_path,
)


def _no_walkable_2x2(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if (
                grid[r, c]
                and grid[r + 1, c]
                and grid[r, c + 1]
                and grid[r + 1, c + 1]
            ):
                return False
    return True


def test_inner_perimeter_nonempty() -> None:
    assert len(_inner_perimeter_cells(16)) > 0


def test_outer_perimeter_nonempty() -> None:
    assert len(outer_perimeter_cells(16)) == 4 * 16 - 4


def test_generator_determinism() -> None:
    cfg = MazeConfig(size=14, random_pct=50)
    g = MazeGenerator(cfg)
    a = g.generate(12345)
    b = g.generate(12345)
    assert np.array_equal(a.grid, b.grid)
    assert a.start == b.start and a.goal == b.goal


def test_generator_constraints(small_config: MazeConfig) -> None:
    gen = MazeGenerator(small_config)
    layout = gen.generate(7)
    g = layout.grid
    assert g.shape == (small_config.size, small_config.size)
    assert g[layout.start] == 1 and g[layout.goal] == 1
    peri = set(outer_perimeter_cells(small_config.size))
    assert layout.start in peri and layout.goal in peri
    assert opposite_rim_pair(small_config.size, layout.start, layout.goal)
    assert _no_walkable_2x2(g)


def test_start_goal_always_opposite_rims() -> None:
    cfg = MazeConfig(size=16, random_pct=50)
    gen = MazeGenerator(cfg)
    for seed in range(50):
        layout = gen.generate(seed)
        assert opposite_rim_pair(cfg.size, layout.start, layout.goal)


def test_higher_random_increases_walkable_area() -> None:
    counts_low: list[int] = []
    counts_high: list[int] = []

    low = MazeGenerator(MazeConfig(size=16, random_pct=0))
    high = MazeGenerator(MazeConfig(size=16, random_pct=100))

    for seed in range(10):
        counts_low.append(int((low.generate(seed).grid == 1).sum()))
        counts_high.append(int((high.generate(seed).grid == 1).sum()))

    assert np.mean(counts_high) > np.mean(counts_low) + 30


def test_generator_density_capped() -> None:
    gen = MazeGenerator(MazeConfig(size=24, random_pct=100))
    for seed in range(10):
        layout = gen.generate(seed)
        density = float((layout.grid == 1).sum()) / float(layout.grid.size)
        assert density <= DENSITY_CAP


def test_join_detector_ignores_parent_but_flags_other_connections() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    assert not _would_join_existing_path(grid, (2, 2), 2, 3)
    grid[1, 3] = 1
    assert _would_join_existing_path(grid, (2, 2), 2, 3)
