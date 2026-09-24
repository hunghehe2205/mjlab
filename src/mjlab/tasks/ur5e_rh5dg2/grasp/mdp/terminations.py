"""Continuous lift-and-hold success, with per-environment reset state."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.scenes.workstation import TABLE_CENTER, TABLE_SIZE, TABLE_TOP_Z
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import stable_hold

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LiftSuccess:
  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self.required_steps = math.ceil(cfg.params["hold_time"] / env.step_dt)
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last_step = torch.zeros_like(self.count)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.count[ids] = 0
    self.last_step[ids] = 0

  def __call__(
    self, env: ManagerBasedRlEnv, height: float, hold_time: float
  ) -> torch.Tensor:
    del hold_time
    advance = env.episode_length_buf != self.last_step
    next_count = torch.where(stable_hold(env, height), self.count + 1, 0)
    self.count[:] = torch.where(advance, next_count, self.count)
    self.last_step[:] = env.episode_length_buf
    return self.count >= self.required_steps


def dropped(env: ManagerBasedRlEnv) -> torch.Tensor:
  pos = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  return (
    (pos[:, 2] < TABLE_TOP_Z - 0.05)
    | ((pos[:, 0] - TABLE_CENTER[0]).abs() > TABLE_SIZE[0])
    | ((pos[:, 1] - TABLE_CENTER[1]).abs() > TABLE_SIZE[1])
  )


def hold_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Observation dimensions are probed before the termination manager is built.
  if not hasattr(env, "termination_manager"):
    return torch.zeros(env.num_envs, device=env.device)
  term = env.termination_manager.get_term_cfg("success").func
  assert isinstance(term, LiftSuccess)
  return (term.count / term.required_steps).clamp(max=1.0)


def success(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.get_term("success").float()
