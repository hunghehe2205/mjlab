"""Termination terms for the DexGrasp teacher task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def hand_below_table(
  env: ManagerBasedRlEnv,
  table_top_z: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate when any monitored hand keypoint goes below the tabletop."""
  robot: Entity = env.scene[asset_cfg.name]
  keypoint_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  table_z = env.scene.env_origins[:, 2] + table_top_z
  return (keypoint_z < table_z.unsqueeze(1)).any(dim=1)
