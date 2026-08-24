"""Unit tests for DexGrasp termination terms."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dexgrasp.mdp.terminations import hand_below_table


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
