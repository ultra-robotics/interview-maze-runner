"""Headless evaluation: success rate, step stats, CSV export."""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from tabulate import tabulate
from tqdm.auto import tqdm

from maze_runner.constants import MazeConfig
from maze_runner.env.maze_env import MazeEnv
from maze_runner.policy import build_policy


@dataclass(frozen=True, slots=True)
class EvalSummary:
    policy_name: str
    episodes: int
    successes: int
    failures: int
    success_rate: float
    steps_success: list[int]
    steps_min: int | None
    steps_max: int | None
    steps_avg: float | None
    steps_median: float | None
    csv_path: Path

    def as_rows(self) -> list[list[Any]]:
        return [
            ["episodes", self.episodes],
            ["successes", self.successes],
            ["failures", self.failures],
            ["success_rate", f"{self.success_rate:.4f}"],
            ["steps_min (success)", self.steps_min],
            ["steps_max (success)", self.steps_max],
            ["steps_avg (success)", None if self.steps_avg is None else f"{self.steps_avg:.4f}"],
            ["steps_median (success)", None if self.steps_median is None else f"{self.steps_median:.4f}"],
        ]


@dataclass(frozen=True, slots=True)
class ComparativeEvalSummary:
    results: dict[str, EvalSummary]
    csv_path: Path

    def as_rows(self) -> list[list[Any]]:
        names = list(self.results)
        metric_rows: list[tuple[str, Callable[[EvalSummary], Any]]] = [
            ("episodes", lambda s: s.episodes),
            ("successes", lambda s: s.successes),
            ("failures", lambda s: s.failures),
            ("success_rate", lambda s: f"{s.success_rate:.4f}"),
            ("steps_min (success)", lambda s: s.steps_min),
            ("steps_max (success)", lambda s: s.steps_max),
            (
                "steps_avg (success)",
                lambda s: None if s.steps_avg is None else f"{s.steps_avg:.4f}",
            ),
            (
                "steps_median (success)",
                lambda s: None if s.steps_median is None else f"{s.steps_median:.4f}",
            ),
        ]
        return [[label, *[getter(self.results[name]) for name in names]] for label, getter in metric_rows]


class PolicyHarness:
    """Runs policies on many episodes with a given `MazeConfig`."""

    def __init__(
        self,
        config: MazeConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self._config = config
        self._device = device or torch.device("cpu")

    def run(
        self,
        episodes: int,
        policy: Any | None = None,
        base_seed: int = 0,
        show_progress: bool = False,
    ) -> EvalSummary:
        h = w = self._config.size
        pol = policy or build_policy("model", h, w, seed=0)
        policy_name = getattr(pol, "policy_name", pol.__class__.__name__.replace("Policy", ""))
        policy_name = str(policy_name).lower()
        stats_dir = Path.cwd() / "stats"
        stats_dir.mkdir(exist_ok=True)

        progress = None
        progress_cb: Callable[..., Any] | None = None
        if show_progress:
            progress = tqdm(total=episodes, desc=f"Evaluating {policy_name}", unit="maze")
            progress_cb = progress.update

        try:
            metrics = self._collect_metrics(
                episodes=episodes,
                policy=pol,
                base_seed=base_seed,
                progress_cb=progress_cb,
            )
        finally:
            if progress is not None:
                progress.close()

        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_path = stats_dir / f"stats-{policy_name}-{ts}.csv"
        self._write_single_policy_csv(csv_path, policy_name=policy_name, metrics=metrics)
        return EvalSummary(policy_name=policy_name, csv_path=csv_path, **metrics)

    def run_comparison(
        self,
        episodes: int,
        base_seed: int = 0,
        show_progress: bool = False,
    ) -> ComparativeEvalSummary:
        """Evaluate the built-in model against a random-action baseline."""
        h = w = self._config.size
        stats_dir = Path.cwd() / "stats"
        stats_dir.mkdir(exist_ok=True)
        policies: list[tuple[str, Any]] = [
            ("model", build_policy("model", h, w, seed=0)),
            ("random_choice", build_policy("random_choice", h, w, seed=base_seed)),
        ]
        results: dict[str, EvalSummary] = {}

        progress = None
        progress_cb: Callable[..., Any] | None = None
        if show_progress:
            progress = tqdm(total=episodes * len(policies), desc="Evaluating policies", unit="run")
            progress_cb = progress.update

        try:
            for name, policy in policies:
                metrics = self._collect_metrics(
                    episodes=episodes,
                    policy=policy,
                    base_seed=base_seed,
                    progress_cb=progress_cb,
                )
                results[name] = EvalSummary(policy_name=name, csv_path=Path(), **metrics)
        finally:
            if progress is not None:
                progress.close()

        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_path = stats_dir / f"stats-compare-{ts}.csv"
        self._write_comparison_csv(csv_path, results)

        results = {
            name: EvalSummary(
                policy_name=summary.policy_name,
                episodes=summary.episodes,
                successes=summary.successes,
                failures=summary.failures,
                success_rate=summary.success_rate,
                steps_success=summary.steps_success,
                steps_min=summary.steps_min,
                steps_max=summary.steps_max,
                steps_avg=summary.steps_avg,
                steps_median=summary.steps_median,
                csv_path=csv_path,
            )
            for name, summary in results.items()
        }
        return ComparativeEvalSummary(results=results, csv_path=csv_path)

    def _collect_metrics(
        self,
        episodes: int,
        policy: Any,
        base_seed: int,
        progress_cb: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        env = MazeEnv(self._config)
        policy.to(self._device)
        policy.eval()

        steps_success: list[int] = []
        successes = 0

        for ep in range(episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.select_action(obs, device=self._device)
                obs, _r, terminated, truncated, _info = env.step(action)
                steps += 1

            if terminated and not truncated:
                successes += 1
                steps_success.append(steps)
            if progress_cb is not None:
                progress_cb(1)

        failures = episodes - successes
        rate = successes / episodes if episodes else 0.0
        smin = min(steps_success) if steps_success else None
        smax = max(steps_success) if steps_success else None
        savg = statistics.mean(steps_success) if steps_success else None
        smed = statistics.median(steps_success) if steps_success else None
        return {
            "episodes": episodes,
            "successes": successes,
            "failures": failures,
            "success_rate": rate,
            "steps_success": steps_success,
            "steps_min": smin,
            "steps_max": smax,
            "steps_avg": savg,
            "steps_median": smed,
        }

    def _write_single_policy_csv(
        self,
        path: Path,
        *,
        policy_name: str,
        metrics: dict[str, Any],
    ) -> None:
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "metric", "value"])
            w.writerow([policy_name, "episodes", metrics["episodes"]])
            w.writerow([policy_name, "successes", metrics["successes"]])
            w.writerow([policy_name, "failures", metrics["failures"]])
            w.writerow([policy_name, "success_rate", f"{metrics['success_rate']:.6f}"])
            w.writerow([policy_name, "steps_min_success", metrics["steps_min"]])
            w.writerow([policy_name, "steps_max_success", metrics["steps_max"]])
            w.writerow([policy_name, "steps_avg_success", metrics["steps_avg"]])
            w.writerow([policy_name, "steps_median_success", metrics["steps_median"]])
            w.writerow(
                [
                    policy_name,
                    "success_episode_steps_csv",
                    ";".join(map(str, metrics["steps_success"])),
                ]
            )

    def _write_comparison_csv(
        self,
        path: Path,
        results: dict[str, EvalSummary],
    ) -> None:
        names = list(results)
        rows = ComparativeEvalSummary(results=results, csv_path=path).as_rows()
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", *names])
            for row in rows:
                w.writerow(row)
            w.writerow(
                [
                    "success_episode_steps_csv",
                    *[";".join(map(str, results[name].steps_success)) for name in names],
                ]
            )


def run_stats(
    config: MazeConfig,
    episodes: int = 1000,
    base_seed: int = 0,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> ComparativeEvalSummary:
    """Convenience: compare the model against a random-action baseline."""
    harness = PolicyHarness(config, device=device)
    summary = harness.run_comparison(
        episodes=episodes,
        base_seed=base_seed,
        show_progress=show_progress,
    )
    print(tabulate(summary.as_rows(), headers=["metric", *summary.results]))
    print(f"Wrote {summary.csv_path}")
    return summary


def maze_config_from_args(
    size: int | None = None,
    random_pct: int | None = None,
    max_steps: int | None = None,
) -> MazeConfig:
    """Build config from CLI overrides with README defaults for omitted fields."""
    return MazeConfig(
        size=size if size is not None else MazeConfig.defaults().size,
        random_pct=random_pct if random_pct is not None else MazeConfig.defaults().random_pct,
        max_steps=max_steps if max_steps is not None else MazeConfig.defaults().max_steps,
    )
