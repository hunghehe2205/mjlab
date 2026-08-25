"""Unit tests for DexGrasp termination terms."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import OBJECT_WORKSPACE_BOUNDS
from mjlab.tasks.dexgrasp.mdp.terminations import (
  hand_below_table,
  object_out_of_workspace,
)

_BOUNDS = OBJECT_WORKSPACE_BOUNDS


def _env_with_object(positions: torch.Tensor, origins: torch.Tensor):
  obj = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions))
  scene = MagicMock()
  scene.__getitem__.return_value = obj
  scene.env_origins = origins
  return SimpleNamespace(scene=scene)


def test_hand_below_table_uses_each_environment_origin() -> None:
  positions = torch.tensor(
    [
      [[0.0, 0.0, 0.80], [0.0, 0.0, 0.79]],
      [[0.0, 0.0, 1.30], [0.0, 0.0, 1.24]],
    ]
  )
  robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=positions))
  scene = MagicMock()
  scene.__getitem__.return_value = robot
  scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.50]])
  env = SimpleNamespace(scene=scene)
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.body_ids = [0, 1]

  terminated = hand_below_table(
    cast(ManagerBasedRlEnv, env), table_top_z=0.771, asset_cfg=asset_cfg
  )

  assert torch.equal(terminated, torch.tensor([False, True]))


def test_object_out_of_workspace_flags_each_axis_violation() -> None:
  positions = torch.tensor(
    [
      [0.0, -0.6, 0.85],  # Inside the box.
      [0.8, -0.6, 0.85],  # x high.
      [0.0, -1.5, 0.85],  # y low.
      [0.0, -0.6, 0.70],  # z low (below the tabletop).
      [0.0, -0.6, 1.30],  # z high (flung up).
    ]
  )
  env = _env_with_object(positions, torch.zeros(5, 3))

  terminated = object_out_of_workspace(cast(ManagerBasedRlEnv, env), bounds=_BOUNDS)

  assert torch.equal(terminated, torch.tensor([False, True, True, True, True]))


def test_object_out_of_workspace_uses_each_environment_origin() -> None:
  origins = torch.tensor([[0.0, 0.0, 0.0], [2.5, 2.5, 0.0]])
  positions = torch.tensor([[0.0, -0.6, 0.85], [2.5, 1.9, 0.85]])

  terminated = object_out_of_workspace(
    cast(ManagerBasedRlEnv, _env_with_object(positions, origins)), bounds=_BOUNDS
  )

  assert torch.equal(terminated, torch.tensor([False, False]))


def test_object_out_of_workspace_catches_hammer_dump_before_divergence() -> None:
  position = torch.tensor([[-0.97924, -2.13356, 1.40455]])
  env = _env_with_object(position, torch.zeros(1, 3))

  terminated = object_out_of_workspace(cast(ManagerBasedRlEnv, env), bounds=_BOUNDS)

  assert terminated.item()
