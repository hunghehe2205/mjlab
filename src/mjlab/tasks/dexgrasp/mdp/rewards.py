"""DexGrasp reward terms (Phase 1 §F), reference coeffs kept as baseline.

Python-side terms (train.py): affordance distance, table log-barrier, arm
height log-barrier, arm collision. C++-side terms (Environment.hpp step()):
weighted contact/impulse rewards, object stability, wrist and arm joint
velocity penalties. Terms return unweighted values; the reference coeffs from
cfg_reg.yaml are applied as mjlab reward weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.dexgrasp.mdp.metrics import object_root_pos_state
from mjlab.tasks.dexgrasp.mdp.observations import (
  FLAG_IMPULSE,
  REFERENCE_IMPULSE_DT,
  keypoint_min_distances,
  sensor_impulse,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "REWARD_COEFFS",
  "affordance_weights",
  "contact_weights",
  "AffordanceDistance",
  "TableLogBarrier",
  "ArmHeightLogBarrier",
  "ContactReward",
  "EnclosureGatedContact",
  "ArmCollision",
  "ObjectDisplacement",
  "object_velocity",
  "object_angular_velocity",
  "wrist_velocity",
  "wrist_angular_velocity",
  "arm_joint_velocity",
]

# cfg_reg.yaml environment.reward coefficients (baseline).
REWARD_COEFFS = {
  "affordance_distance": 0.5,
  # Contact/impulse are enclosure-gated (thumb + >=2 fingers): they fire only on
  # a wrapping grip, not palm/single-finger farming. Impulse halved so squeeze
  # stops dominating the shaped reward once gated.
  "affordance_contact": 1.5,
  "affordance_impulse": 0.5,
  "table_logbarrier": -0.03,
  "table_contact": -1.0,
  "table_impulse": -0.5,
  "arm_height_logbarrier": -0.05,
  "arm_contact": -0.1,
  "arm_impulse": -0.1,
  "arm_collision": -1.0,
  # Reference object-stability coefficients; the softened -5/-0.1/-2 let the
  # policy push and tip the object (8yelo0qc displacement 3 -> 8 cm).
  "object_velocity": -15.0,
  "object_angular_velocity": -0.2,
  "object_displacement": -5.0,
  "wrist_velocity": -1.0,
  "wrist_angular_velocity": -0.1,
  "arm_joint_velocity": -1.0,
}

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def affordance_weights(
  tip_indices: tuple[int, ...],
  thumb_tip_index: int,
  wrist_index: int,
  num: int = 24,
  device: str = "cpu",
) -> torch.Tensor:
  """Distance-reward keypoint weights (tips x4, thumb tip x8, wrist 0)."""
  w = torch.ones(num, dtype=torch.float32, device=device)
  w[list(tip_indices)] *= 4.0
  w[thumb_tip_index] *= 2.0
  w[wrist_index] = 0.0
  w /= w.sum()
  w *= 16.0
  return w


def contact_weights(
  tip_indices: tuple[int, ...],
  thumb_indices: tuple[int, ...],
  thumb_tip_index: int,
  palm_index: int,
  num: int = 16,
  device: str = "cpu",
) -> torch.Tensor:
  """Contact-reward body weights (tips x3, thumb x2, thumb tip x2, palm 0)."""
  w = torch.ones(num, dtype=torch.float32, device=device)
  w[list(tip_indices)] *= 3.0
  w[list(thumb_indices)] *= 2.0
  w[thumb_tip_index] *= 2.0
  w[palm_index] = 0.0
  w /= w.sum()
  w *= 16.0
  return w


class AffordanceDistance:
  """-sum(weights * min_dist) over the hand keypoints."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    names = tuple(
      cfg.params.get("object_names")
      or (cfg.params.get("object_name", "potted_meat_can"),)
    )
    self._pcd = torch.as_tensor(
      np.stack([oc.PHASE1_OBJECTS[name].load_surface_points() for name in names]),
      dtype=torch.float32,
      device=env.device,
    )
    self._robot = env.scene[cfg.params["asset_cfg"].name]
    self._object = env.scene[cfg.params["object_entity"]]
    self._keypoint_ids = cfg.params["asset_cfg"].body_ids
    self._weights = affordance_weights(
      cfg.params["tip_indices"],
      cfg.params["thumb_tip_index"],
      cfg.params["wrist_index"],
      device=env.device,
    )
    self._variant_ids = env.sim.world_to_variant.get(cfg.params["object_entity"])
    if self._variant_ids is None:
      self._variant_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    obj_pos = self._object.data.root_link_pos_w
    obj_quat = self._object.data.root_link_quat_w
    keypoints_w = self._robot.data.body_link_pos_w[:, self._keypoint_ids]
    min_dist = keypoint_min_distances(
      keypoints_w, obj_pos, obj_quat, self._pcd[self._variant_ids]
    )
    return -(min_dist * self._weights).sum(dim=-1)


class TableLogBarrier:
  """-sum(weights * log(50 * clip(h, 0.002, 0.02))) over the keypoints."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._robot = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._table_top_z = float(cfg.params["table_top_z"])
    self._weights = affordance_weights(
      cfg.params["tip_indices"],
      cfg.params["thumb_tip_index"],
      cfg.params["wrist_index"],
      device=env.device,
    )

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    heights = self._robot.data.body_link_pos_w[:, self._body_ids, 2]
    heights = heights - self._table_top_z
    clipped = torch.clamp(heights, min=0.002, max=0.02)
    return -(torch.log(50.0 * clipped) * self._weights).sum(dim=-1)


class ArmHeightLogBarrier:
  """-sum(log(50 * clip(h, 0.002, 0.02))) over the 4 last arm links."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._robot = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._table_top_z = float(cfg.params["table_top_z"])

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    heights = self._robot.data.body_link_pos_w[:, self._body_ids, 2]
    heights = heights - self._table_top_z
    clipped = torch.clamp(heights[:, 2:6], min=0.002, max=0.02)
    return -torch.log(50.0 * clipped).sum(dim=-1)


class ContactReward:
  """Weighted contact flags or clipped impulse reward from a contact sensor.

  mode "flags": sum(weights * (impulse_norm > 0.01)) / divisor.
  mode "impulse_xy": sum(weights * clamp(|impulse_xy|, high)).
  mode "impulse": sum(weights * clamp(|impulse|, high)).
  Without weights, the per-body vector is reduced by norm (arm terms).
  Pad sensor slots fold into their parent bodies via pad_parent_indices.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    sensor_names = cfg.params.get("sensor_names")
    if sensor_names is None:
      sensor_names = (cfg.params["sensor_name"],)
    self._sensors: tuple[ContactSensor, ...] = tuple(
      env.scene[name] for name in sensor_names
    )
    self._dt = REFERENCE_IMPULSE_DT
    self._pad_parents = cfg.params.get("pad_parent_indices")
    self._mode = cfg.params["mode"]
    self._divisor = float(cfg.params.get("divisor", 1.0))
    clip_high = cfg.params.get("clip_high")
    self._clip_high = (
      torch.as_tensor(clip_high, dtype=torch.float32, device=env.device)
      if clip_high is not None
      else None
    )
    self._weights = (
      torch.as_tensor(cfg.params["weights"], dtype=torch.float32, device=env.device)
      if "weights" in cfg.params
      else None
    )

  def _impulse(self) -> torch.Tensor:
    """Summed per-body impulse [E, B, 3] across the term's sensors."""
    impulses = [
      sensor_impulse(sensor, self._dt, self._pad_parents) for sensor in self._sensors
    ]
    return torch.stack(impulses).sum(dim=0)

  def _reduce(self, impulse: torch.Tensor) -> torch.Tensor:
    """Mode-specific reduction of a per-body impulse to a per-env scalar."""
    if self._mode == "flags":
      value = (impulse.norm(dim=-1) > FLAG_IMPULSE).float()
    elif self._mode == "impulse_xy":
      value = impulse[..., :2].norm(dim=-1)
    else:
      value = impulse.norm(dim=-1)
    if self._mode != "flags" and self._clip_high is not None:
      value = value.clamp(max=self._clip_high)
    if self._weights is not None:
      return (value * self._weights).sum(dim=-1) / self._divisor
    return value.norm(dim=-1)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    return self._reduce(self._impulse())


class EnclosureGatedContact(ContactReward):
  """ContactReward gated on an opposition grip.

  Zero unless the thumb tip and at least ``min_fingers`` non-thumb fingertips
  are in contact, so palm-only or single-finger presses score nothing -- only a
  wrapping (force-closure) grip is rewarded.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._thumb_tip = int(cfg.params["thumb_tip_index"])
    self._finger_tips = list(cfg.params["finger_tip_indices"])
    self._min_fingers = int(cfg.params.get("min_fingers", 2))

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    impulse = self._impulse()
    value = self._reduce(impulse)
    flags = impulse.norm(dim=-1) > FLAG_IMPULSE
    thumb = flags[:, self._thumb_tip]
    fingers = flags[:, self._finger_tips].sum(dim=-1)
    gate = (thumb & (fingers >= self._min_fingers)).float()
    return value * gate


class ArmCollision:
  """Sum of any-contact flags for the arm links (index 1..4)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    found = self._sensor.data.found
    assert found is not None
    return (found[:, 1:5] > 0).float().sum(dim=-1)


class ObjectDisplacement:
  """Distance of the object from its position at episode reset."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._object = env.scene[cfg.params["object_entity"]]
    n = env.num_envs
    self._init_pos = torch.zeros((n, 3), device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._init_pos[ids] = object_root_pos_state(self._object)[ids]

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    return (self._object.data.root_link_pos_w - self._init_pos).norm(dim=-1)


def object_velocity(
  env: ManagerBasedRlEnv,
  object_entity: str,
) -> torch.Tensor:
  """Squared object linear velocity."""
  obj: Entity = env.scene[object_entity]
  return obj.data.root_link_lin_vel_w.square().sum(dim=-1)


def object_angular_velocity(
  env: ManagerBasedRlEnv,
  object_entity: str,
) -> torch.Tensor:
  """Squared object angular velocity."""
  obj: Entity = env.scene[object_entity]
  return obj.data.root_link_ang_vel_w.square().sum(dim=-1)


def wrist_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared wrist linear velocity, x10 when above 0.25 m/s."""
  robot: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  assert not isinstance(body_ids, slice)
  vel = robot.data.body_link_lin_vel_w[:, body_ids[0]]
  squared = vel.square().sum(dim=-1)
  return torch.where(vel.norm(dim=-1) > 0.25, squared * 10.0, squared)


def wrist_angular_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared wrist angular velocity."""
  robot: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  assert not isinstance(body_ids, slice)
  vel = robot.data.body_link_ang_vel_w[:, body_ids[0]]
  return vel.square().sum(dim=-1)


def arm_joint_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared arm joint velocity, x4 per joint when above 0.5 rad/s."""
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  boosted = torch.where(vel.abs() > 0.5, vel * 4.0, vel)
  return boosted.square().sum(dim=-1)
