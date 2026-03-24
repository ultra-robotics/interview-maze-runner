"""Browser-based maze visualization (FastAPI + static HTML/JS)."""

from __future__ import annotations

import os
import threading
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from maze_runner.constants import (
    DEFAULT_RANDOM,
    DEFAULT_SIZE,
    DEFAULT_STEP_DELAY_S,
    DEFAULT_MAX_STEPS,
    MAX_RANDOM,
    MAX_SIZE,
    MIN_RANDOM,
    MIN_SIZE,
    MazeConfig,
    VIZ_ENV_MAX_STEPS,
    VIZ_ENV_RANDOM_PCT,
    VIZ_ENV_SIZE,
    VIZ_ENV_STEP_DELAY,
)
from maze_runner.env.maze_env import MazeEnv
from maze_runner.policy import POLICY_NAMES, build_policy
from maze_runner.policy.model import PolicyName


def _initial_config_from_env() -> MazeConfig:
    return MazeConfig(
        size=int(os.environ.get(VIZ_ENV_SIZE, str(DEFAULT_SIZE))),
        random_pct=int(os.environ.get(VIZ_ENV_RANDOM_PCT, str(DEFAULT_RANDOM))),
        max_steps=int(os.environ.get(VIZ_ENV_MAX_STEPS, str(DEFAULT_MAX_STEPS))),
    )


def _initial_delay_from_env() -> float:
    return max(0.01, float(os.environ.get(VIZ_ENV_STEP_DELAY, str(DEFAULT_STEP_DELAY_S))))


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WebGameState:
    """Single-session state for the demo (one browser tab)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config = _initial_config_from_env()
        self._step_delay_s = _initial_delay_from_env()
        self._device = _device()
        self._env: MazeEnv | None = None
        self._policy_name: PolicyName = "model"
        self._policy: Any | None = None
        self._episode_seed = 0
        self._playing = False
        self._paused_after_success = False
        self._last_action: int | None = None
        self._decision_seq: int = 0

    @property
    def config(self) -> MazeConfig:
        return self._config

    @property
    def step_delay_s(self) -> float:
        return self._step_delay_s

    def set_step_delay(self, value: float) -> None:
        self._step_delay_s = max(0.01, float(value))

    def _ensure(self) -> None:
        if self._env is None or self._env.maze_config != self._config:
            self._env = MazeEnv(self._config)
        if self._policy is None:
            s = self._config.size
            policy_seed = self._episode_seed if self._policy_name == "random_choice" else 0
            self._policy = build_policy(self._policy_name, s, s, seed=policy_seed).to(
                self._device
            )
            self._policy.eval()

    def set_config(self, cfg: MazeConfig, policy_name: PolicyName | None = None) -> None:
        with self._lock:
            if policy_name is not None and policy_name != self._policy_name:
                self._policy_name = policy_name
                self._policy = None
            self._config = cfg
            self._env = None
            self._policy = None

    def set_policy(self, policy_name: PolicyName) -> dict[str, Any]:
        with self._lock:
            if policy_name == self._policy_name:
                return self._snapshot_unlocked()

            self._policy_name = policy_name
            self._policy = None

            if self._env is None:
                self._ensure()
                return self._snapshot_unlocked()

            was_playing = self._playing
            self._ensure()
            assert self._env is not None and self._env.layout is not None
            self._env.reset(seed=self._episode_seed, options={"regenerate": False})
            self._playing = was_playing
            self._paused_after_success = False
            self._last_action = None
            self._decision_seq = 0
            return self._snapshot_unlocked()

    def new_maze(self, seed: int | None = None) -> dict[str, Any]:
        import time

        with self._lock:
            self._ensure()
            assert self._env is not None and self._policy is not None
            if seed is None:
                self._episode_seed = int(time.time() * 1000) % (2**31)
            else:
                self._episode_seed = int(seed)
            self._policy = None
            self._env.reset(seed=self._episode_seed)
            self._playing = True
            self._paused_after_success = False
            self._last_action = None
            self._decision_seq = 0
            return self._snapshot_unlocked()

    def restart_same_maze(self) -> dict[str, Any]:
        with self._lock:
            self._ensure()
            assert self._env is not None and self._policy is not None
            self._policy = None
            self._ensure()
            assert self._policy is not None
            self._env.reset(seed=self._episode_seed, options={"regenerate": False})
            self._playing = True
            self._paused_after_success = False
            self._last_action = None
            self._decision_seq = 0
            return self._snapshot_unlocked()

    def play(self) -> dict[str, Any]:
        with self._lock:
            self._playing = True
            return self._snapshot_unlocked()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._playing = False
            return self._snapshot_unlocked()

    def tick(self) -> dict[str, Any]:
        """Run one policy step if playing and not yet successful."""
        with self._lock:
            self._ensure()
            assert self._env is not None and self._policy is not None
            if not self._playing or self._paused_after_success:
                return self._snapshot_unlocked()

            obs = self._env.observation()
            action = self._policy.select_action(obs, device=self._device)
            self._last_action = int(action)
            self._decision_seq += 1
            _obs, _r, terminated, _truncated, _info = self._env.step(action)
            if terminated:
                self._playing = False
                self._paused_after_success = True
            return self._snapshot_unlocked()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        if self._env is None or self._env.layout is None:
            return {
                "grid": [],
                "agent": [0, 0],
                "start": [0, 0],
                "goal": [0, 0],
                "size": self._config.size,
                "config": {
                    "size": self._config.size,
                    "random_pct": self._config.random_pct,
                },
                "policy_name": self._policy_name,
                "available_policies": list(POLICY_NAMES),
                "playing": False,
                "success": False,
                "step_delay_s": self._step_delay_s,
                "last_action": None,
                "decision_seq": 0,
            }
        layout = self._env.layout
        g = layout.grid
        ar, ac = self._env.agent
        sr, sc = layout.start
        gr, gc = layout.goal
        return {
            "grid": g.astype(int).tolist(),
            "agent": [int(ar), int(ac)],
            "start": [int(sr), int(sc)],
            "goal": [int(gr), int(gc)],
            "size": int(self._config.size),
            "config": {
                "size": self._config.size,
                "random_pct": self._config.random_pct,
            },
            "policy_name": self._policy_name,
            "available_policies": list(POLICY_NAMES),
            "playing": self._playing,
            "success": self._paused_after_success,
            "step_delay_s": self._step_delay_s,
            "last_action": self._last_action,
            "decision_seq": self._decision_seq,
        }


state = WebGameState()


class NewMazeBody(BaseModel):
    size: int | None = Field(default=None, ge=MIN_SIZE, le=MAX_SIZE)
    random_pct: int | None = Field(default=None, ge=MIN_RANDOM, le=MAX_RANDOM)
    policy_name: PolicyName | None = None
    seed: int | None = None


class PolicyBody(BaseModel):
    policy_name: PolicyName


def create_app() -> FastAPI:
    app = FastAPI(title="Maze Runner", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        return JSONResponse(state.state())

    @app.post("/api/new-maze")
    def api_new_maze(body: NewMazeBody) -> JSONResponse:
        cfg = MazeConfig(
            size=body.size if body.size is not None else state.config.size,
            random_pct=body.random_pct
            if body.random_pct is not None
            else state.config.random_pct,
            max_steps=state.config.max_steps,
        )
        state.set_config(cfg, policy_name=body.policy_name)
        return JSONResponse(state.new_maze(seed=body.seed))

    @app.post("/api/policy")
    def api_policy(body: PolicyBody) -> JSONResponse:
        return JSONResponse(state.set_policy(body.policy_name))

    @app.post("/api/restart")
    def api_restart() -> JSONResponse:
        return JSONResponse(state.restart_same_maze())

    @app.post("/api/play")
    def api_play() -> JSONResponse:
        return JSONResponse(state.play())

    @app.post("/api/pause")
    def api_pause() -> JSONResponse:
        return JSONResponse(state.pause())

    @app.post("/api/tick")
    def api_tick() -> JSONResponse:
        return JSONResponse(state.tick())

    return app


app = create_app()

# Embedded UI — avoids wheel packaging issues for non-Python assets.
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Maze Runner</title>
  <style>
    :root {
      --bg: #12141a;
      --panel: #1c2028;
      --text: #e8eaef;
      --muted: #9aa3b2;
      --wall: #373b47;
      --path: #d2d8e6;
      --start-outline: #3b82f6;
      --goal: #5adc6e;
      --btn: #4668a8;
      --btn-hi: #5a7ec4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
    }
    .wrap { display: flex; flex-wrap: wrap; gap: 1.25rem; padding: 1.25rem; max-width: 1100px; margin: 0 auto; }
    .page-header {
      width: 100%;
      display: flex;
      justify-content: center;
      margin-bottom: 0.1rem;
    }
    .wordmark {
      margin: 0;
      text-align: center;
      font-weight: 900;
      font-size: clamp(2.1rem, 6vw, 4.4rem);
      line-height: 0.95;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #f8fafc;
      text-shadow:
        0 2px 0 #1d4ed8,
        0 4px 14px rgba(59, 130, 246, 0.35),
        0 0 28px rgba(90, 220, 110, 0.18);
    }
    .wordmark .maze {
      display: inline-block;
      color: #dbeafe;
      transform: skew(-8deg);
    }
    .wordmark .runner {
      display: inline-block;
      margin-left: 0.28em;
      color: #bbf7d0;
      transform: skew(-8deg);
    }
    .board-wrap {
      background: var(--panel); padding: 0.75rem; border-radius: 10px;
      box-shadow: 0 8px 32px rgba(0,0,0,.35);
    }
    .board {
      display: grid; gap: 1px; background: #0e1016;
      width: min(72vmin, 560px); height: min(72vmin, 560px);
    }
    .cell {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 0;
      min-height: 0;
      line-height: 1;
      background: var(--path);
      font-size: clamp(7px, min(2.8vmin, 2.8vw), 22px);
    }
    .cell.wall { background: var(--wall); }
    .cell.start {
      background:
        linear-gradient(0deg, rgba(59, 130, 246, 0.24), rgba(59, 130, 246, 0.24)),
        var(--path);
      box-shadow: inset 0 0 0 3px var(--start-outline);
    }
    .cell.goal {
      background:
        linear-gradient(0deg, rgba(90, 220, 110, 0.28), rgba(90, 220, 110, 0.28)),
        var(--path);
      box-shadow: inset 0 0 0 3px var(--goal);
    }
    .cell.start.goal {
      box-shadow: inset 0 0 0 3px var(--start-outline), inset 0 0 0 6px var(--goal);
    }
    .cell .robot {
      pointer-events: none;
      user-select: none;
      line-height: 1;
      display: block;
      max-width: 100%;
      max-height: 100%;
      font-size: 0.85em;
      z-index: 1;
    }
    .panel {
      flex: 1 1 280px;
    }
    .row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; align-items: center; }
    button {
      background: var(--btn); color: var(--text); border: none; padding: 0.45rem 0.9rem;
      border-radius: 6px; cursor: pointer; font-size: 0.9rem;
    }
    button:hover { background: var(--btn-hi); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .btn-play-pause { min-width: 5.5rem; }
    .btn-play-pause.playing {
      background: #2d6a4f;
      color: #f0fdf4;
    }
    .btn-play-pause.playing:hover { background: #40916c; }
    .btn-play-pause.paused {
      background: var(--btn);
      color: var(--text);
    }
    .btn-play-pause.paused:hover { background: var(--btn-hi); }
    label { font-size: 0.85rem; color: var(--muted); display: block; margin-bottom: 0.2rem; }
    input[type=range] { width: 100%; }
    .status { font-size: 0.9rem; color: var(--muted); margin-top: 0.5rem; }
    .hint { font-size: 0.8rem; color: var(--muted); margin-top: 0.75rem; line-height: 1.4; }
    .decision-panel {
      display: flex; align-items: center; gap: 0.65rem;
      padding: 0.65rem 0.85rem; margin-bottom: 0.75rem;
      background: rgba(0,0,0,.22); border-radius: 8px;
      border: 1px solid rgba(255,255,255,.06);
      min-height: 2.75rem;
    }
    .decision-panel .robot-decision {
      font-size: 1.35rem; line-height: 1;
      user-select: none;
    }
    .decision-panel .action-arrow {
      font-size: 1.5rem; font-weight: 600; color: var(--goal);
      min-width: 1.25em; text-align: center;
      line-height: 1;
      user-select: none;
      transition: color 0.12s ease;
    }
    .decision-panel .decision-label {
      font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
      letter-spacing: 0.04em; margin-right: 0.25rem;
    }
    @keyframes decisionPulse {
      0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(90, 220, 110, 0.45); }
      55% { transform: scale(1.03); box-shadow: 0 0 0 10px rgba(90, 220, 110, 0); }
      100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(90, 220, 110, 0); }
    }
    .decision-panel.pulse { animation: decisionPulse 0.38s ease-out; }
    .control-block { margin-bottom: 0.75rem; }
    .control-well {
      background: rgba(0,0,0,.2);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 10px;
      padding: 0.9rem;
      margin-bottom: 0.9rem;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
    }
    .well-title {
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin: 0 0 0.75rem;
    }
    .control-well .row:last-child,
    .control-well .control-block:last-child,
    .control-well .hint:last-child,
    .control-well .decision-panel:last-child {
      margin-bottom: 0;
    }
    select {
      width: 100%;
      background: #141821;
      color: var(--text);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 6px;
      padding: 0.55rem 0.7rem;
      font-size: 0.9rem;
    }
    .confetti-canvas {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 50;
    }
  </style>
</head>
<body>
  <canvas id="confettiCanvas" class="confetti-canvas"></canvas>
  <div class="wrap">
    <div class="page-header">
      <h1 class="wordmark"><span class="maze">Maze</span> <span class="runner">Runner</span></h1>
    </div>
    <div>
      <div class="board-wrap"><div id="board" class="board"></div></div>
    </div>
    <div class="panel">
      <div class="control-well">
        <p class="well-title">Robot Controls</p>
        <div class="row">
          <button id="btnPlayPause" type="button" class="btn-play-pause paused">Play</button>
          <button id="btnRestart" type="button">Restart</button>
        </div>
        <div id="decisionPanel" class="decision-panel" aria-live="polite">
          <span class="decision-label">Move</span>
          <span class="robot-decision" title="Agent">\U0001F916</span>
          <span id="actionArrow" class="action-arrow" title="Last chosen direction">—</span>
        </div>
        <div class="control-block">
          <label for="policySelect">Policy</label>
          <select id="policySelect"></select>
        </div>
      </div>
      <div class="control-well">
        <p class="well-title">Maze Controls</p>
        <div class="row">
          <button id="btnNew" type="button">New Maze</button>
        </div>
        <div class="row">
          <label>Size (<span id="vSize">16</span>)<input id="size" type="range" min="8" max="64" step="1" value="16"/></label>
        </div>
        <div class="row">
          <label>Random % (<span id="vRand">55</span>)<input id="rand" type="range" min="0" max="100" step="1" value="55"/></label>
        </div>
        <p class="hint">Controls apply on <strong>New Maze</strong>. Higher <strong>Random %</strong> makes the main walk less direct and grows more side corridors. Restart keeps the current maze and resets the agent.</p>
      </div>
    </div>
  </div>
  <script>
    const board = document.getElementById('board');
    const decisionPanel = document.getElementById('decisionPanel');
    const actionArrowEl = document.getElementById('actionArrow');
    const policySelectEl = document.getElementById('policySelect');
    const confettiCanvas = document.getElementById('confettiCanvas');
    const confettiCtx = confettiCanvas.getContext('2d');
    const ARROWS = ['\u2191', '\u2192', '\u2193', '\u2190'];
    let tickTimer = null;
    let tickLoopActive = false;
    let lastSeenDecisionSeq = 0;
    let lastSuccess = false;
    let confettiBurstId = 0;

    function resizeConfettiCanvas() {
      const dpr = window.devicePixelRatio || 1;
      confettiCanvas.width = Math.floor(window.innerWidth * dpr);
      confettiCanvas.height = Math.floor(window.innerHeight * dpr);
      confettiCanvas.style.width = window.innerWidth + 'px';
      confettiCanvas.style.height = window.innerHeight + 'px';
      confettiCtx.setTransform(1, 0, 0, 1, 0, 0);
      confettiCtx.scale(dpr, dpr);
    }

    function labelForPolicy(name) {
      return name.split('_').map(part => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
    }

    function syncPolicySelect(s) {
      const names = Array.isArray(s.available_policies) ? s.available_policies : [];
      if (!names.length) return;
      const current = s.policy_name || 'model';
      const existing = Array.from(policySelectEl.options).map(opt => opt.value);
      if (existing.join('|') !== names.join('|')) {
        policySelectEl.innerHTML = '';
        for (const name of names) {
          const opt = document.createElement('option');
          opt.value = name;
          opt.textContent = labelForPolicy(name);
          policySelectEl.appendChild(opt);
        }
      }
      policySelectEl.value = current;
    }

    function fireConfetti() {
      resizeConfettiCanvas();
      const burstId = ++confettiBurstId;
      const pieces = Array.from({ length: 140 }, () => ({
        x: window.innerWidth * (0.15 + Math.random() * 0.7),
        y: -20 - Math.random() * 40,
        vx: (Math.random() - 0.5) * 7,
        vy: 2 + Math.random() * 3.5,
        size: 5 + Math.random() * 8,
        color: ['#5adc6e', '#60a5fa', '#fbbf24', '#f472b6', '#f87171'][Math.floor(Math.random() * 5)],
        rot: Math.random() * Math.PI,
        spin: (Math.random() - 0.5) * 0.35,
      }));
      const durationMs = 2400;
      const start = performance.now();

      function frame(now) {
        if (burstId !== confettiBurstId) return;
        const elapsed = now - start;
        confettiCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
        for (const piece of pieces) {
          piece.x += piece.vx;
          piece.y += piece.vy;
          piece.vy += 0.06;
          piece.rot += piece.spin;
          confettiCtx.save();
          confettiCtx.translate(piece.x, piece.y);
          confettiCtx.rotate(piece.rot);
          confettiCtx.fillStyle = piece.color;
          confettiCtx.fillRect(-piece.size / 2, -piece.size / 2, piece.size, piece.size * 0.65);
          confettiCtx.restore();
        }
        if (elapsed < durationMs) {
          requestAnimationFrame(frame);
        } else {
          confettiCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
        }
      }

      requestAnimationFrame(frame);
    }

    function updateSuccessEffects(s) {
      const success = !!s.success;
      if (success && !lastSuccess) fireConfetti();
      lastSuccess = success;
    }

    function triggerDecisionPulse() {
      decisionPanel.classList.remove('pulse');
      void decisionPanel.offsetWidth;
      decisionPanel.classList.add('pulse');
    }

    function updateDecisionPanel(s, fromTick) {
      const seq = s.decision_seq != null ? s.decision_seq : 0;
      const a = s.last_action;
      if (a == null || a === undefined || a < 0 || a > 3) {
        actionArrowEl.textContent = '\u2014';
        actionArrowEl.title = 'No move yet';
        lastSeenDecisionSeq = seq;
        return;
      }
      actionArrowEl.textContent = ARROWS[a];
      const names = ['up', 'right', 'down', 'left'];
      actionArrowEl.title = 'Last move: ' + names[a];
      if (fromTick && seq !== lastSeenDecisionSeq) {
        lastSeenDecisionSeq = seq;
        triggerDecisionPulse();
      } else if (!fromTick) {
        lastSeenDecisionSeq = seq;
      }
    }

    function syncSlidersFromState(s) {
      const c = s.config || {};
      syncPolicySelect(s);
      if (c.size != null) {
        document.getElementById('size').value = c.size;
        document.getElementById('vSize').textContent = c.size;
      }
      if (c.random_pct != null) {
        document.getElementById('rand').value = c.random_pct;
        document.getElementById('vRand').textContent = c.random_pct;
      }
    }

    function renderBoard(s) {
      const n = s.size || (s.grid && s.grid.length) || 0;
      if (!n || !s.grid) { board.innerHTML = ''; return; }
      board.style.gridTemplateColumns = 'repeat(' + n + ', minmax(0, 1fr))';
      board.style.gridTemplateRows = 'repeat(' + n + ', minmax(0, 1fr))';
      board.innerHTML = '';
      const [ar, ac] = s.agent || [0, 0];
      const [sr, sc] = s.start || [ar, ac];
      const [gr, gc] = s.goal || [0, 0];
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const d = document.createElement('div');
          d.className = 'cell' + (s.grid[r][c] === 0 ? ' wall' : '');
          if (r === sr && c === sc) d.classList.add('start');
          if (r === gr && c === gc) d.classList.add('goal');
          if (r === ar && c === ac) {
            const bot = document.createElement('span');
            bot.className = 'robot';
            bot.textContent = '\U0001F916';
            d.appendChild(bot);
          }
          board.appendChild(d);
        }
      }
    }

    function setStatus(s) {
      syncPlayPauseButton(s);
    }

    function syncPlayPauseButton(s) {
      const btn = document.getElementById('btnPlayPause');
      const on = !!(s.playing && !s.success);
      btn.textContent = on ? 'Pause' : 'Play';
      btn.classList.toggle('playing', on);
      btn.classList.toggle('paused', !on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    }

    async function fetchState() {
      const r = await fetch('/api/state');
      const s = await r.json();
      renderBoard(s);
      setStatus(s);
      syncSlidersFromState(s);
      updateDecisionPanel(s, false);
      updateSuccessEffects(s);
      return s;
    }

    async function post(path, body) {
      const r = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : '{}'
      });
      const s = await r.json();
      renderBoard(s);
      setStatus(s);
      // Only sync sliders when the server applied a new maze config — not on every tick/play/pause.
      if (path === '/api/new-maze') {
        syncSlidersFromState(s);
      }
      updateDecisionPanel(s, path === '/api/tick');
      updateSuccessEffects(s);
      return s;
    }

    function stopPlayLoop() {
      tickLoopActive = false;
      if (tickTimer) {
        clearTimeout(tickTimer);
        tickTimer = null;
      }
    }

    /** Runs policy inference on an interval; first step happens immediately. */
    async function startPlayLoop(delayMs) {
      stopPlayLoop();
      tickLoopActive = true;
      async function doStep() {
        if (!tickLoopActive) return;
        const s = await post('/api/tick', {});
        if (!s.playing || s.success) {
          stopPlayLoop();
          return;
        }
        tickTimer = setTimeout(doStep, delayMs);
      }
      await doStep();
    }

    async function applyCurrentSettings(extra = {}) {
      stopPlayLoop();
      const payload = {
        size: parseInt(document.getElementById('size').value, 10),
        random_pct: parseInt(document.getElementById('rand').value, 10),
        policy_name: policySelectEl.value || 'model',
        ...extra,
      };
      const s = await post('/api/new-maze', payload);
      const ms = Math.max(10, Math.round((s.step_delay_s || 0.2) * 1000));
      if (s.playing) await startPlayLoop(ms);
    }

    async function applyPolicySelection() {
      stopPlayLoop();
      const s = await post('/api/policy', {
        policy_name: policySelectEl.value || 'model',
      });
      const ms = Math.max(10, Math.round((s.step_delay_s || 0.2) * 1000));
      if (s.playing) await startPlayLoop(ms);
    }

    document.getElementById('btnPlayPause').onclick = async () => {
      const s0 = await fetchState();
      if (s0.playing && !s0.success) {
        stopPlayLoop();
        await post('/api/pause', {});
      } else {
        const ms = Math.max(10, Math.round((s0.step_delay_s || 0.2) * 1000));
        await post('/api/play', {});
        await startPlayLoop(ms);
      }
    };
    document.getElementById('btnRestart').onclick = async () => {
      stopPlayLoop();
      const s = await post('/api/restart', {});
      const ms = Math.max(10, Math.round((s.step_delay_s || 0.2) * 1000));
      if (s.playing) await startPlayLoop(ms);
    };
    document.getElementById('btnNew').onclick = async () => {
      await applyCurrentSettings();
    };
    policySelectEl.onchange = async () => {
      await applyPolicySelection();
    };

    ['size','rand'].forEach(id => {
      document.getElementById(id).addEventListener('input', (e) => {
        const map = { size: 'vSize', rand: 'vRand' };
        document.getElementById(map[id]).textContent = e.target.value;
      });
    });
    window.addEventListener('resize', resizeConfettiCanvas);
    resizeConfettiCanvas();

    (async () => {
      await applyCurrentSettings({ seed: 0 });
    })();
  </script>
</body>
</html>
"""
