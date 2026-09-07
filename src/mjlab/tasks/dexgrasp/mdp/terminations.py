"""Termination terms for the DexGrasp teacher task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_out_of_workspace(
  env: ManagerBasedRlEnv,
  bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
  object_entity: str = "object",
) -> torch.Tensor:
  """Object root left the env-local task workspace (a flung/spun object)."""
  obj: Entity = env.scene[object_entity]
  local = obj.data.root_link_pos_w - env.scene.env_origins
  limits = torch.as_tensor(bounds, device=local.device)
  return ((local < limits[:, 0]) | (local > limits[:, 1])).any(dim=1)
