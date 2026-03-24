"""Single-layer baseline policy (README: very basic untrained)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from maze_runner.constants import ACTION_DIM, observation_channels

PolicyName = Literal["model", "random_choice"]
POLICY_NAMES: tuple[PolicyName, ...] = ("model", "random_choice")


class BaselinePolicy(nn.Module):
    """Flattened observation -> logits over four directions."""

    policy_name = "model"

    def __init__(self, height: int, width: int, channels: int | None = None) -> None:
        super().__init__()
        c = channels if channels is not None else observation_channels()
        self._flat = int(height * width * c)
        self.fc = nn.Linear(self._flat, ACTION_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        b = x.shape[0]
        flat = x.reshape(b, -1)
        return self.fc(flat)

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        device: torch.device | str | None = None,
    ) -> int:
        """Return discrete action index from observation (H,W,C) float."""
        dev = device or torch.device("cpu")
        t = torch.from_numpy(obs).float().to(dev)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        logits = self.forward(t)
        return int(torch.argmax(logits, dim=-1).item())


class RandomChoicePolicy:
    """Baseline that picks a random direction each step."""

    policy_name = "random_choice"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def to(self, device: torch.device | str | None = None) -> RandomChoicePolicy:
        return self

    def eval(self) -> RandomChoicePolicy:
        return self

    def select_action(
        self,
        obs: np.ndarray,
        device: torch.device | str | None = None,
    ) -> int:
        del obs, device
        return int(self._rng.integers(0, ACTION_DIM))


def build_policy(
    policy_name: PolicyName,
    height: int,
    width: int,
    *,
    seed: int = 0,
) -> BaselinePolicy | RandomChoicePolicy:
    """Construct a supported policy by name."""
    if policy_name == "model":
        # Keep model initialization reproducible without mutating global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return BaselinePolicy(height, width)
    if policy_name == "random_choice":
        return RandomChoicePolicy(seed=seed)
    raise ValueError(f"unknown policy_name: {policy_name}")
