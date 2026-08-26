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
  tolerance: float = 0.0,
) -> torch.Tensor:
  """Terminate when a monitored hand keypoint penetrates the table tolerance."""
  robot: Entity = env.scene[asset_cfg.name]
  keypoint_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  table_z = env.scene.env_origins[:, 2] + table_top_z - tolerance
  return (keypoint_z < table_z.unsqueeze(1)).any(dim=1)


def object_out_of_workspace(
  env: ManagerBasedRlEnv,
  bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
  object_entity: str = "object",
) -> torch.Tensor:
  """Terminate when the object root leaves the env-local task workspace."""
  obj: Entity = env.scene[object_entity]
  local = obj.data.root_link_pos_w - env.scene.env_origins
  limits = torch.as_tensor(bounds, device=local.device)
  return ((local < limits[:, 0]) | (local > limits[:, 1])).any(dim=1)
