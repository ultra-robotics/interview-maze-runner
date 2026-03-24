from __future__ import annotations

from pathlib import Path

from maze_runner.constants import MazeConfig
from maze_runner.eval.harness import PolicyHarness, maze_config_from_args


def test_harness_writes_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = MazeConfig(size=10, random_pct=20, max_steps=200)
    h = PolicyHarness(cfg)
    summary = h.run(episodes=3, base_seed=1)
    assert summary.csv_path.exists()
    text = summary.csv_path.read_text()
    assert "success_rate" in text


def test_harness_comparison_writes_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = MazeConfig(size=10, random_pct=20, max_steps=200)
    h = PolicyHarness(cfg)
    summary = h.run_comparison(episodes=3, base_seed=1)
    assert summary.csv_path.exists()
    text = summary.csv_path.read_text()
    assert "random_choice" in text
    assert "model" in text
    assert "success_rate" in text


def test_maze_config_from_args() -> None:
    c = maze_config_from_args(size=20)
    assert c.size == 20
    assert c.random_pct == MazeConfig.defaults().random_pct
