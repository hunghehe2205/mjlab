"""Device-resident geometry and filtered contact signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import (
  BOX_SIZE,
  FORCE_THRESHOLD,
  OBJECT_POS,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def box_surface_vectors(
  points: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
  half_size: tuple[float, float, float] = BOX_SIZE,
) -> torch.Tensor:
  """Anchor-to-nearest-surface vectors, including anchors inside the box."""
  rotation = quat[:, None, :].expand(-1, points.shape[1], -1)
  local = quat_apply_inverse(rotation, points - pos[:, None, :])
  size = local.new_tensor(half_size)
  nearest = torch.clamp(local, min=-size, max=size)
  inside = (local.abs() <= size).all(dim=-1)
  face = (size - local.abs()).argmin(dim=-1, keepdim=True)
  sign = torch.where(local >= 0, 1.0, -1.0)
  surface = local.scatter(-1, face, (sign * size).gather(-1, face))
  nearest = torch.where(inside[..., None], surface, nearest)
  return quat_apply(rotation, nearest - local)


def normal_force(
  env: ManagerBasedRlEnv, name: str, history: bool = False
) -> torch.Tensor:
  sensor = env.scene[name]
  assert isinstance(sensor, ContactSensor)
  data = sensor.data
  assert data.force is not None
  force = data.force[..., 0].clamp_min(0.0)
  if history and data.force_history is not None:
    force = torch.maximum(force, data.force_history[..., 0].amax(dim=-1))
  return force


def finger_contacts(env: ManagerBasedRlEnv) -> torch.Tensor:
  return normal_force(env, "finger_object") > FORCE_THRESHOLD


def lift_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (
    env.scene["object"].data.root_link_pos_w[:, 2]
    - env.scene.env_origins[:, 2]
    - OBJECT_POS[2]
  )


def object_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["object"].data.root_link_lin_vel_w.norm(dim=-1)


def stable_hold(env: ManagerBasedRlEnv, height: float) -> torch.Tensor:
  obj = env.scene["object"].data
  return (
    (lift_height(env) >= height)
    & (object_speed(env) <= 0.05)
    & (obj.root_link_ang_vel_w.norm(dim=-1) <= 1.0)
    & (finger_contacts(env).sum(dim=-1) >= 2)
    & (normal_force(env, "object_table").amax(dim=-1) < FORCE_THRESHOLD)
  )


def fingers_in_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  return finger_contacts(env).sum(dim=-1).float()
