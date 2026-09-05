"""Tests for the DexGrasp online metric terms."""

import math
from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dexgrasp.mdp.metrics import (
  joint_pos_mean,
  mean_arm_action_magnitude,
  object_tilt_angle,
  object_tilt_deg,
)


def test_arm_action_and_joint_metrics() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(joint_pos=torch.tensor([[0.2, 0.4], [1.0, 3.0]]))
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": robot},
      action_manager=SimpleNamespace(
        action=torch.tensor([[1.0, -0.5, 0.0], [0.2, 0.4, 0.6]])
      ),
    ),
  )
  asset_cfg = cast(SceneEntityCfg, SimpleNamespace(name="robot", joint_ids=[1]))

  assert torch.equal(mean_arm_action_magnitude(env, 2), torch.tensor([0.75, 0.3]))
  assert torch.equal(joint_pos_mean(env, asset_cfg), torch.tensor([0.4, 3.0]))


def test_object_tilt() -> None:
  half = math.sqrt(0.5)
  quat = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [half, half, 0.0, 0.0]]
  )  # upright, on side
  obj = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=quat))
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene={"object": obj}))

  torch.testing.assert_close(
    object_tilt_angle(quat), torch.tensor([0.0, math.pi / 2]), atol=1e-6, rtol=0
  )
  torch.testing.assert_close(
    object_tilt_deg(env, "object"), torch.tensor([0.0, 90.0]), atol=1e-4, rtol=0
  )
