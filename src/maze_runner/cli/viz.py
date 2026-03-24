"""`uv run viz` — local web UI for the maze (FastAPI + browser)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from maze_runner.constants import (
    DEFAULT_RANDOM,
    DEFAULT_SIZE,
    DEFAULT_STEP_DELAY_S,
    DEFAULT_MAX_STEPS,
    VIZ_ENV_MAX_STEPS,
    VIZ_ENV_RANDOM_PCT,
    VIZ_ENV_SIZE,
    VIZ_ENV_STEP_DELAY,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the maze visualizer in your browser.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address")
    p.add_argument("--port", type=int, default=8000, help="HTTP port")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Maze side length")
    p.add_argument("--random", type=int, default=DEFAULT_RANDOM, dest="random_pct", help="Random walk %%")
    p.add_argument("--delay", type=float, default=DEFAULT_STEP_DELAY_S, help="Seconds between policy steps")
    p.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable hot reload (single process, no file watching)",
    )
    args = p.parse_args()

    # Workers spawned by --reload inherit these; required so CLI flags apply after reload.
    os.environ[VIZ_ENV_SIZE] = str(args.size)
    os.environ[VIZ_ENV_RANDOM_PCT] = str(args.random_pct)
    os.environ[VIZ_ENV_STEP_DELAY] = str(args.delay)
    os.environ[VIZ_ENV_MAX_STEPS] = str(DEFAULT_MAX_STEPS)

    reload = not args.no_reload
    src_root = Path(__file__).resolve().parents[2]

    mode = "hot reload" if reload else "no reload"
    print(f"Open http://{args.host}:{args.port}/ in your browser ({mode}; Ctrl+C to stop)")

    run_kw: dict = {
        "app": "maze_runner.viz.web_app:app",
        "host": args.host,
        "port": args.port,
        "reload": reload,
    }
    if reload:
        run_kw["reload_dirs"] = [str(src_root)]

    uvicorn.run(**run_kw)


if __name__ == "__main__":
    main()
