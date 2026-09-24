"""Joint targets sampled once per policy step, held during decimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GraspActionCfg(JointPositionActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> GraspAction:
    return GraspAction(self, env)


class GraspAction(JointPositionAction):
  @property
  def target(self) -> torch.Tensor:
    return self._processed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    q = self._entity.data.joint_pos[:, self.target_ids]
    limits = self._entity.data.joint_pos_limits[:, self.target_ids]
    self._processed_actions = torch.clamp(
      q + actions.clamp(-1.0, 1.0) * self._scale,
      min=limits[..., 0],
      max=limits[..., 1],
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._raw_actions[ids] = 0.0
    self._processed_actions[ids] = self._entity.data.joint_pos[ids][:, self.target_ids]
    self._entity.set_joint_position_target(
      self._processed_actions[ids], joint_ids=self.target_ids, env_ids=ids
    )
