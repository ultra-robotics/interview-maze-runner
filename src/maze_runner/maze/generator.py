"""Procedural maze generation (perimeter endpoints, narrow corridors)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maze_runner.constants import DR, DC, MazeConfig

DENSITY_CAP = 0.50


@dataclass(frozen=True, slots=True)
class MazeLayout:
    """Immutable maze: walkable mask, start, goal, and generation seed."""

    grid: np.ndarray  # uint8, 0 wall 1 walkable
    start: tuple[int, int]
    goal: tuple[int, int]
    seed: int


def _in_bounds(size: int, r: int, c: int) -> bool:
    return 0 <= r < size and 0 <= c < size


def _inner_perimeter_cells(size: int) -> list[tuple[int, int]]:
    """First ring of cells inside the outer border (legacy helper for tests)."""
    cells: list[tuple[int, int]] = []
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if r == 1 or r == size - 2 or c == 1 or c == size - 2:
                cells.append((r, c))
    return cells


def outer_perimeter_cells(size: int) -> list[tuple[int, int]]:
    """All cells on the outermost row or column of the grid (true perimeter)."""
    cells: list[tuple[int, int]] = []
    for c in range(size):
        cells.append((0, c))
        cells.append((size - 1, c))
    for r in range(1, size - 1):
        cells.append((r, 0))
        cells.append((r, size - 1))
    return cells


def _outer_edge_rims(size: int) -> dict[str, list[tuple[int, int]]]:
    """Split the **outer** grid perimeter into four edges (corners on top/bottom rows only).

    - **top** / **bottom**: full width (rows ``0`` and ``size - 1``)
    - **left** / **right**: vertical sides excluding corners (rows ``1 .. size - 2``)
    """
    return {
        "top": [(0, c) for c in range(0, size)],
        "bottom": [(size - 1, c) for c in range(0, size)],
        "left": [(r, 0) for r in range(1, size - 1)],
        "right": [(r, size - 1) for r in range(1, size - 1)],
    }


def opposite_rim_pair(size: int, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if `a` and `b` lie on opposite outer rims (top↔bottom or left↔right)."""

    def edge(rc: tuple[int, int]) -> str | None:
        r, c = rc
        if r == 0:
            return "top"
        if r == size - 1:
            return "bottom"
        if c == 0:
            return "left"
        if c == size - 1:
            return "right"
        return None

    e1, e2 = edge(a), edge(b)
    if e1 is None or e2 is None:
        return False
    op = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    return op[e1] == e2


def _pick_opposite_rim_endpoints(
    size: int, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose start on one rim and goal on the opposite rim (top↔bottom or left↔right)."""
    rims = _outer_edge_rims(size)
    pairs: list[tuple[str, str]] = []
    if rims["top"] and rims["bottom"]:
        pairs.append(("top", "bottom"))
    if rims["left"] and rims["right"]:
        pairs.append(("left", "right"))
    if not pairs:
        raise ValueError(
            "maze too small for opposite-rim endpoints (need distinct top/bottom or left/right rims)"
        )
    e0, e1 = pairs[int(rng.integers(0, len(pairs)))]
    if rng.random() < 0.5:
        start_edge, goal_edge = e0, e1
    else:
        start_edge, goal_edge = e1, e0
    si = int(rng.integers(0, len(rims[start_edge])))
    gi = int(rng.integers(0, len(rims[goal_edge])))
    return rims[start_edge][si], rims[goal_edge][gi]


def _would_violate_narrow_corridors(grid: np.ndarray, r: int, c: int) -> bool:
    """No walkable 2x2 block (README constraint)."""
    if grid[r, c] != 0:
        return False
    h, w = grid.shape
    grid[r, c] = 1
    bad = False
    for rr in range(h - 1):
        for cc in range(w - 1):
            if (
                grid[rr, cc]
                and grid[rr + 1, cc]
                and grid[rr, cc + 1]
                and grid[rr + 1, cc + 1]
            ):
                bad = True
                break
        if bad:
            break
    grid[r, c] = 0
    return bad


def _carvable_dirs(grid: np.ndarray, r: int, c: int) -> list[int]:
    """Directions that can be carved next without breaking corridor rules."""
    h, w = grid.shape
    dirs: list[int] = []
    for i in range(4):
        nr, nc = r + DR[i], c + DC[i]
        if not _in_bounds(h, nr, nc):
            continue
        if grid[nr, nc] != 0:
            continue
        if _would_violate_narrow_corridors(grid, nr, nc):
            continue
        dirs.append(i)
    return dirs


def _max_walkable_cells(grid: np.ndarray) -> int:
    """Maximum allowed number of walkable cells under the density cap."""
    return max(2, int(grid.size * DENSITY_CAP))


def _would_join_existing_path(
    grid: np.ndarray,
    parent: tuple[int, int],
    r: int,
    c: int,
) -> bool:
    """Whether carving `(r, c)` would connect a branch into some other path.

    The immediate parent cell is allowed; any additional adjacent walkable cell
    means the new carve would merge into existing corridors and create a more
    densely connected region.
    """
    h, w = grid.shape
    for i in range(4):
        nr, nc = r + DR[i], c + DC[i]
        if not _in_bounds(h, nr, nc):
            continue
        if (nr, nc) == parent:
            continue
        if grid[nr, nc] == 1:
            return True
    return False


class MazeGenerator:
    """Builds mazes from a `MazeConfig` (size and random_pct)."""

    def __init__(self, config: MazeConfig) -> None:
        self._config = config

    def generate(self, seed: int | None = None) -> MazeLayout:
        rng = np.random.default_rng(seed)
        s = self._config.size
        if s < 3:
            raise ValueError("size must be >= 3")

        base = int(seed) if seed is not None else int(rng.integers(0, 2**31 - 1))
        for attempt in range(200):
            inner_seed = base + attempt
            rng2 = np.random.default_rng(inner_seed)

            start_rc, goal_rc = _pick_opposite_rim_endpoints(s, rng2)

            grid = np.zeros((s, s), dtype=np.uint8)

            ok = self._carve_main_path(
                grid, start_rc, goal_rc, rng2, self._config.random_pct
            )
            if not ok:
                continue

            self._add_branching(grid, rng2, self._config.random_pct)

            return MazeLayout(grid=grid, start=start_rc, goal=goal_rc, seed=inner_seed)

        raise RuntimeError("failed to generate a maze after many attempts")

    def _carve_main_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        rng: np.random.Generator,
        random_pct: int,
    ) -> bool:
        """Random walk with backtracking from start until goal is carved.

        When stuck, the search backtracks along the path stack; carved exploration
        dead-ends stay on the grid. Neighbor choice uses Manhattan distance to the
        goal with mild stochastic jitter; ``random_pct`` scales extra noise.
        """
        h, w = grid.shape
        max_dim = max(h, w)
        mild = 0.12 * max_dim
        random_scale = (random_pct / 100.0) * max_dim
        path: list[tuple[int, int]] = [start]
        grid[start] = 1
        max_walkable = _max_walkable_cells(grid)

        while path:
            cur = path[-1]
            if cur == goal:
                return True
            if int(grid.sum()) >= max_walkable:
                return False

            neighbors: list[tuple[int, int]] = []
            for i in range(4):
                nr, nc = cur[0] + DR[i], cur[1] + DC[i]
                if not _in_bounds(h, nr, nc):
                    continue
                if grid[nr, nc] != 0:
                    continue
                if _would_violate_narrow_corridors(grid, nr, nc):
                    continue
                neighbors.append((nr, nc))

            if not neighbors:
                path.pop()
                if not path:
                    return False
                continue

            def _score(p: tuple[int, int]) -> float:
                d = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
                return d + rng.random() * random_scale + rng.random() * mild

            neighbors.sort(key=_score)
            nxt = neighbors[0]
            grid[nxt] = 1
            path.append(nxt)

        return False

    def _add_branching(
        self,
        grid: np.ndarray,
        rng: np.random.Generator,
        random_pct: int,
    ) -> None:
        """Grow recursive corridor segments off carved paths.

        `random_pct` now controls both the main-path randomness and how aggressively
        fractal side corridors grow off the carved route.
        """
        if random_pct <= 0:
            return

        h, w = grid.shape
        density = random_pct / 100.0
        frontier = [tuple(rc) for rc in zip(*np.where(grid == 1))]
        rng.shuffle(frontier)

        max_generations = max(1, min(5, 1 + random_pct // 20))
        remaining_capacity = max(0, _max_walkable_cells(grid) - int(grid.sum()))
        if remaining_capacity <= 0:
            return
        extra_budget = min(remaining_capacity, max(1, int(h * w * 0.45 * density)))

        for gen in range(max_generations):
            if extra_budget <= 0 or not frontier:
                break

            source_prob = min(0.95, 0.12 + 0.75 * density * (0.82**gen))
            sources = frontier[:]
            rng.shuffle(sources)
            next_frontier: list[tuple[int, int]] = []

            for cell in sources:
                if extra_budget <= 0:
                    break
                if rng.random() >= source_prob:
                    continue
                carved = self._grow_branch_segment(
                    grid=grid,
                    origin=cell,
                    rng=rng,
                    random_pct=random_pct,
                    generation=gen,
                    budget=extra_budget,
                )
                if not carved:
                    continue
                extra_budget -= len(carved)
                next_frontier.extend(carved)

            frontier = next_frontier

    def _grow_branch_segment(
        self,
        grid: np.ndarray,
        origin: tuple[int, int],
        rng: np.random.Generator,
        random_pct: int,
        generation: int,
        budget: int,
    ) -> list[tuple[int, int]]:
        """Carve one corridor segment, preferring to continue straight."""
        if budget <= 0:
            return []

        h, w = grid.shape
        density = random_pct / 100.0
        cur = origin
        dirs = _carvable_dirs(grid, cur[0], cur[1])
        if not dirs:
            return []

        cur_dir = int(dirs[int(rng.integers(0, len(dirs)))])
        max_len = min(
            budget,
            max(2, min(max(h, w) // 2, 2 + random_pct // 18)),
        )
        target_len = int(rng.integers(1, max_len + 1))

        carved: list[tuple[int, int]] = []
        for _ in range(target_len):
            dirs = _carvable_dirs(grid, cur[0], cur[1])
            if not dirs:
                break
            if rng.random() < 0.85:
                non_joining = []
                for i in dirs:
                    nr, nc = cur[0] + DR[i], cur[1] + DC[i]
                    if not _would_join_existing_path(grid, cur, nr, nc):
                        non_joining.append(i)
                if non_joining:
                    dirs = non_joining

            def _dir_score(i: int) -> float:
                turn_penalty = 0.0 if i == cur_dir else 0.28
                return turn_penalty + rng.random() * (0.9 - 0.3 * density)

            dirs.sort(key=_dir_score)
            cur_dir = dirs[0]
            nr, nc = cur[0] + DR[cur_dir], cur[1] + DC[cur_dir]
            grid[nr, nc] = 1
            cur = (nr, nc)
            carved.append(cur)

            stop_prob = max(0.08, 0.34 - 0.2 * density + 0.12 * generation)
            if rng.random() < stop_prob:
                break

        return carved
