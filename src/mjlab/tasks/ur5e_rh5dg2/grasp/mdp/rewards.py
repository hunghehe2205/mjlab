"""Nonnegative grasp incentives and costs after RobustDexGrasp; signs live in weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.scenes.workstation import TABLE_TOP_Z
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import (
  FORCE_THRESHOLD,
  GRIP_FORCE,
  HAND_BODIES,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.events import object_start
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import (
  box_surface_vectors,
  normal_force,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def link_weights(tip: float, thumb: float, thumb_tip: float) -> tuple[float, ...]:
  """Per hand body weights summing to 1; the palm gets none."""
  weights = []
  for name in HAND_BODIES:
    w = 0.0 if name == "R_hand_palm" else 1.0
    w *= tip if name.endswith("_dip") else 1.0
    w *= thumb if name.startswith("R_thumb") else 1.0
    w *= thumb_tip if name == "R_thumb_dip" else 1.0
    weights.append(w)
  return tuple(w / sum(weights) for w in weights)


def contact_weights() -> tuple[float, ...]:
  """Reference contact weights: fingertips x3, thumb x2, thumb tip x2 more."""
  return link_weights(3.0, 2.0, 2.0)


def distance_weights() -> tuple[float, ...]:
  """Reference distance weights: fingertips x4, thumb tip x2 more."""
  return link_weights(4.0, 1.0, 2.0)


def _weighted(score: torch.Tensor) -> torch.Tensor:
  return score @ score.new_tensor(contact_weights())


def distance(env: ManagerBasedRlEnv, anchors: SceneEntityCfg) -> torch.Tensor:
  """Weighted hand-link distance to the box, scaled by the finger link count."""
  obj = env.scene["object"].data
  points = env.scene["robot"].data.body_link_pos_w[:, anchors.body_ids]
  vectors = box_surface_vectors(points, obj.root_link_pos_w, obj.root_link_quat_w)
  weights = points.new_tensor(distance_weights())
  return vectors.norm(dim=-1) @ weights * (len(HAND_BODIES) - 1)


def contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Weighted fraction of hand links touching the object."""
  return _weighted((normal_force(env, "hand_object") > FORCE_THRESHOLD).float())


def grip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Weighted horizontal contact force, capped per link (thumb cap doubled)."""
  sensor = env.scene["hand_object_world"]
  assert isinstance(sensor, ContactSensor) and sensor.data.force is not None
  cap = torch.tensor(
    [GRIP_FORCE * (2.0 if n.startswith("R_thumb") else 1.0) for n in HAND_BODIES],
    device=env.device,
  )
  force = sensor.data.force[..., :2].norm(dim=-1)
  return _weighted(torch.minimum(force, cap) / cap)


def crash(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Cancel manager dt scaling for a one-time terminal penalty."""
  return env.termination_manager.get_term("hand_below_table").float() / env.step_dt


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


def object_displacement(env: ManagerBasedRlEnv) -> torch.Tensor:
  pos = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  return (pos - object_start(env)).norm(dim=-1)


def object_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["object"].data.root_link_lin_vel_w.square().sum(dim=-1)


def object_angular_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["object"].data.root_link_ang_vel_w.square().sum(dim=-1)


def wrist_velocity(env: ManagerBasedRlEnv, palm: SceneEntityCfg) -> torch.Tensor:
  """Squared palm speed, ten times above 0.25 m/s."""
  vel = env.scene["robot"].data.body_link_lin_vel_w[:, palm.body_ids].squeeze(1)
  cost = vel.square().sum(dim=-1)
  return torch.where(vel.norm(dim=-1) > 0.25, 10.0 * cost, cost)


def wrist_angular_velocity(
  env: ManagerBasedRlEnv, palm: SceneEntityCfg
) -> torch.Tensor:
  vel = env.scene["robot"].data.body_link_ang_vel_w[:, palm.body_ids].squeeze(1)
  return vel.square().sum(dim=-1)


def arm_joint_velocity(env: ManagerBasedRlEnv, arm: SceneEntityCfg) -> torch.Tensor:
  """Squared arm joint speed, amplified four times beyond 0.5 rad/s."""
  vel = env.scene["robot"].data.joint_vel[:, arm.joint_ids]
  vel = torch.where(vel.abs() > 0.5, 4.0 * vel, vel)
  return vel.square().sum(dim=-1)


def peak_contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  return normal_force(env, "hand_object").amax(dim=-1)
