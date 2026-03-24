from __future__ import annotations

from fastapi.testclient import TestClient

from maze_runner.constants import MazeConfig
from maze_runner.viz.web_app import app, state


def setup_function() -> None:
    state.set_config(MazeConfig(size=10, random_pct=30, max_steps=100))
    state.new_maze(seed=3)


def test_web_root() -> None:
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Maze Runner" in r.text


def test_api_new_maze_and_tick() -> None:
    client = TestClient(app)
    r = client.post(
        "/api/new-maze",
        json={"seed": 1, "size": 10, "random_pct": 20, "policy_name": "random_choice"},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["grid"]) == 10
    assert data["config"]["size"] == 10
    assert data["policy_name"] == "random_choice"
    assert "model" in data["available_policies"]
    client.post("/api/play", json={})
    r2 = client.post("/api/tick", json={})
    assert r2.status_code == 200
    d2 = r2.json()
    assert "agent" in d2
    assert d2["last_action"] in (0, 1, 2, 3)
    assert d2["decision_seq"] == 1


def test_api_policy_switch_keeps_same_maze_and_resets_agent() -> None:
    client = TestClient(app)
    r1 = client.post(
        "/api/new-maze",
        json={"seed": 7, "size": 10, "random_pct": 20, "policy_name": "model"},
    )
    assert r1.status_code == 200
    d1 = r1.json()

    client.post("/api/play", json={})
    client.post("/api/tick", json={})

    r2 = client.post("/api/policy", json={"policy_name": "random_choice"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["policy_name"] == "random_choice"
    assert d2["grid"] == d1["grid"]
    assert d2["agent"] == d2["start"]
    assert d2["decision_seq"] == 0
