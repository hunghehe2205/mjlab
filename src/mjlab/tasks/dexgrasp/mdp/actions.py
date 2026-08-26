"""DexGrasp action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import (
  RelativeJointPositionAction,
  RelativeJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "ReferenceRelativeJointPositionAction",
  "ReferenceRelativeJointPositionActionCfg",
]


@dataclass(kw_only=True)
class ReferenceRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
  """Relative joint targets with the reference one-substep action delay."""

  first_substep_delay_prob: float = 0.5

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 <= self.first_substep_delay_prob <= 1.0:
      raise ValueError("first_substep_delay_prob must be in [0, 1].")

  def build(self, env: ManagerBasedRlEnv) -> ReferenceRelativeJointPositionAction:
    return ReferenceRelativeJointPositionAction(self, env)


class ReferenceRelativeJointPositionAction(RelativeJointPositionAction):
  """Hold the previous target for the first substep with probability 0.5."""

  def __init__(
    self, cfg: ReferenceRelativeJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self._delay_prob = cfg.first_substep_delay_prob
    self._previous_target = torch.zeros_like(self._target)
    self._delay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._substep = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._previous_target[ids] = self._target[ids]
    self._delay_mask[ids] = False
    self._substep = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    self._previous_target.copy_(self._target)
    super().process_actions(actions)
    self._delay_mask = torch.rand(self.num_envs, device=self.device) < self._delay_prob
    self._substep = 0

  def apply_actions(self) -> None:
    target = self._target
    if self._substep == 0:
      target = torch.where(
        self._delay_mask.unsqueeze(-1), self._previous_target, self._target
      )
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
    self._substep += 1
