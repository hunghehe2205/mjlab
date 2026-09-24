"""Nonnegative task incentives and costs; signs live in reward weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.scenes.workstation import TABLE_TOP_Z
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import FORCE_THRESHOLD
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.events import object_start
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import (
  box_surface_vectors,
  finger_contacts,
  lift_height,
  normal_force,
  stable_hold,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.terminations import success

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reach(env: ManagerBasedRlEnv, tips: SceneEntityCfg) -> torch.Tensor:
  obj = env.scene["object"].data
  points = env.scene["robot"].data.body_link_pos_w[:, tips.body_ids]
  vectors = box_surface_vectors(points, obj.root_link_pos_w, obj.root_link_quat_w)
  return torch.exp(-vectors.norm(dim=-1).mean(dim=-1) / 0.05)


def contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  force = normal_force(env, "hand_object")
  return (
    (force > FORCE_THRESHOLD).float() * (0.5 + 0.5 * (force / 5.0).clamp(max=1.0))
  ).mean(dim=-1)


def lift(env: ManagerBasedRlEnv, height: float) -> torch.Tensor:
  support = finger_contacts(env).sum(dim=-1) >= 2
  return (lift_height(env) / height).clamp(0.0, 1.0) * support


def hold(env: ManagerBasedRlEnv, height: float) -> torch.Tensor:
  return stable_hold(env, height).float()


def completion(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Cancel manager dt scaling for a one-time terminal bonus."""
  return success(env) / env.step_dt


def height_cost(env: ManagerBasedRlEnv, links: SceneEntityCfg) -> torch.Tensor:
  z = env.scene["robot"].data.body_link_pos_w[:, links.body_ids, 2]
  h = z - env.scene.env_origins[:, None, 2] - TABLE_TOP_Z
  return clearance_cost(h).mean(dim=-1)


def clearance_cost(height: torch.Tensor) -> torch.Tensor:
  return ((0.02 - height).clamp_min(0.0) / 0.02).square()


def undesired_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  forces = torch.cat(
    [
      normal_force(env, name, history=True)
      for name in ("robot_table", "hand_self", "arm_object")
    ],
    dim=-1,
  )
  return ((forces > FORCE_THRESHOLD).float() + (forces / 10.0).clamp(0.0, 1.0)).sum(
    dim=-1
  )


def horizontal_displacement(env: ManagerBasedRlEnv) -> torch.Tensor:
  pos = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  return (pos[:, :2] - object_start(env)[:, :2]).square().sum(dim=-1)


def horizontal_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["object"].data.root_link_lin_vel_w[:, :2].square().sum(dim=-1)


def joint_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.joint_vel.square().mean(dim=-1)


def palm_velocity(env: ManagerBasedRlEnv, palm: SceneEntityCfg) -> torch.Tensor:
  return (
    env.scene["robot"]
    .data.body_link_vel_w[:, palm.body_ids]
    .square()
    .sum(dim=-1)
    .mean(dim=-1)
  )


def peak_contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  return normal_force(env, "hand_object").amax(dim=-1)
