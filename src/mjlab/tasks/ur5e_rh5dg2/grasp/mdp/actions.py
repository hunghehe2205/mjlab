"""Joint targets sampled once per policy step, held during decimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.events import lift_delta
from mjlab.tasks.ur5e_rh5dg2.grasp.pregrasp import ARM_JOINTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GraspActionCfg(JointPositionActionCfg):
  grasp_time: float | None = None
  """Start of the lift test; None trains grasp-only without it."""
  lift_ramp: float = 1.0
  """Seconds to ramp the arm from the grasp to the raised pose."""

  def build(self, env: ManagerBasedRlEnv) -> GraspAction:
    return GraspAction(self, env)


class GraspAction(JointPositionAction):
  """Policy joint deltas; in the lift test the arm follows a scripted raise."""

  def __init__(self, cfg: GraspActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._env = env
    self._grasp_steps = (
      None if cfg.grasp_time is None else round(cfg.grasp_time / env.step_dt)
    )
    self._ramp_steps = max(1, math.ceil(cfg.lift_ramp / env.step_dt))
    self._arm = torch.tensor(
      [self.target_names.index(name) for name in ARM_JOINTS], device=self.device
    )
    self._lift_start = torch.zeros(self.num_envs, len(ARM_JOINTS), device=self.device)

  @property
  def target(self) -> torch.Tensor:
    return self._processed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    q = self._entity.data.joint_pos[:, self.target_ids]
    limits = self._entity.data.joint_pos_limits[:, self.target_ids]
    target = q + actions.clamp(-1.0, 1.0) * self._scale
    if self._grasp_steps is not None:
      step = self._env.episode_length_buf - self._grasp_steps
      start = step == 0
      self._lift_start[start] = q[start][:, self._arm]
      progress = ((step + 1) / self._ramp_steps).clamp(0.0, 1.0)[:, None]
      arm = self._lift_start + progress * lift_delta(self._env)
      target[:, self._arm] = torch.where(step[:, None] >= 0, arm, target[:, self._arm])
    self._processed_actions = torch.clamp(
      target, min=limits[..., 0], max=limits[..., 1]
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._raw_actions[ids] = 0.0
    self._processed_actions[ids] = self._entity.data.joint_pos[ids][:, self.target_ids]
    self._entity.set_joint_position_target(
      self._processed_actions[ids], joint_ids=self.target_ids, env_ids=ids
    )
