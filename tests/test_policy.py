from __future__ import annotations

import numpy as np
import torch

from maze_runner.constants import observation_channels
from maze_runner.policy.model import POLICY_NAMES, BaselinePolicy, RandomChoicePolicy, build_policy


def test_baseline_forward() -> None:
    h, w = 9, 9
    pol = BaselinePolicy(h, w)
    x = torch.randn(2, h, w, observation_channels())
    y = pol(x)
    assert y.shape == (2, 4)


def test_select_action() -> None:
    pol = BaselinePolicy(8, 8)
    obs = np.zeros((8, 8, observation_channels()), dtype=np.float32)
    a = pol.select_action(obs)
    assert 0 <= a < 4


def test_random_choice_action() -> None:
    pol = RandomChoicePolicy(seed=0)
    obs = np.zeros((8, 8, observation_channels()), dtype=np.float32)
    a = pol.select_action(obs)
    assert 0 <= a < 4


def test_build_policy_supports_all_registered_names() -> None:
    built = [build_policy(name, 8, 8, seed=0) for name in POLICY_NAMES]
    assert [p.policy_name for p in built] == list(POLICY_NAMES)
