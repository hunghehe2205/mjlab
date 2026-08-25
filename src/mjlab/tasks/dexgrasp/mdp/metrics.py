from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_linear_speed(
  env: ManagerBasedRlEnv, object_entity: str = "object"
) -> torch.Tensor:
  """Object root linear speed in metres per second."""
  obj: Entity = env.scene[object_entity]
  return torch.linalg.vector_norm(obj.data.root_link_lin_vel_w, dim=-1)


def object_angular_speed(
  env: ManagerBasedRlEnv, object_entity: str = "object"
) -> torch.Tensor:
  """Object root angular speed in radians per second."""
  obj: Entity = env.scene[object_entity]
  return torch.linalg.vector_norm(obj.data.root_link_ang_vel_w, dim=-1)
