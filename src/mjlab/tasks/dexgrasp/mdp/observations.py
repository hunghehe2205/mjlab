"""DexGrasp teacher observation terms, native MuJoCo sensing.

Absolute qpos, PD error, pad contact flags + force magnitudes, per-link contact
flags, keypoint/arm-link heights above the table, hand center, wrist frame axes,
and the hand-centric affordance distance vectors (af_vec).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dexgrasp.mdp.contacts import contact_flags, contact_force
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

__all__ = [
  "joint_pos",
  "pd_error",
  "link_heights",
  "hand_center_pos",
  "wrist_rot",
  "PadContactObs",
  "LinkContactObs",
  "nearest_affordance_points",
  "keypoint_min_distances",
  "compute_af_vec",
  "AffordanceVectors",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute joint positions."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.joint_pos[:, asset_cfg.joint_ids]


def pd_error(
  env: ManagerBasedRlEnv,
  action_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """PD position error (clipped target - current qpos), a torque proxy."""
  from mjlab.envs.mdp.actions import JointPositionAction, RelativeJointPositionAction

  robot: Entity = env.scene[asset_cfg.name]
  term = env.action_manager.get_term(action_name)
  if isinstance(term, RelativeJointPositionAction):
    target = term.target[:, asset_cfg.joint_ids]
  else:
    assert isinstance(term, JointPositionAction)
    target = term._processed_actions - robot.data.encoder_bias[:, asset_cfg.joint_ids]
  return target - robot.data.joint_pos[:, asset_cfg.joint_ids]


def link_heights(
  env: ManagerBasedRlEnv,
  table_top_z: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Link-frame heights above the table top."""
  robot: Entity = env.scene[asset_cfg.name]
  z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return z - table_top_z


def hand_center_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Grasp-center site position in the environment-local frame."""
  robot: Entity = env.scene[asset_cfg.name]
  world_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return world_pos - env.scene.env_origins


def wrist_rot(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """World x-axis and z-axis of the wrist body frame (6D rotation)."""
  robot: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  assert not isinstance(body_ids, slice)
  rot = matrix_from_quat(robot.data.body_link_quat_w[:, body_ids[0]])
  return torch.cat([rot[:, :, 0], rot[:, :, 2]], dim=-1)


class PadContactObs:
  """Per-pad contact flags then force magnitudes (2 x 6 pads = 12)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    mag = contact_force(self._sensor).norm(dim=-1)
    return torch.cat([contact_flags(self._sensor), mag], dim=-1)


class LinkContactObs:
  """Per-link instantaneous contact flags (15 finger links)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    found = self._sensor.data.found
    assert found is not None
    return (found > 0).float()


def nearest_affordance_points(
  keypoints_obj: torch.Tensor, pcd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Nearest cloud point per keypoint (object frame) and the distance to it.

  Args:
    keypoints_obj: (..., K, 3) keypoint positions in the object frame.
    pcd: (..., P, 3) affordance cloud in the object frame.

  Returns:
    ((..., K, 3) nearest points, (..., K) distances).
  """
  dists = torch.cdist(keypoints_obj, pcd)
  min_dist, min_idx = torch.min(dists, dim=-1)
  if pcd.ndim == 2:
    nearest = pcd[min_idx]
  else:
    nearest = torch.gather(
      pcd,
      dim=1,
      index=min_idx.unsqueeze(-1).expand(-1, -1, pcd.shape[-1]),
    )
  return nearest, min_dist


def keypoint_min_distances(
  keypoints_w: torch.Tensor,
  obj_pos: torch.Tensor,
  obj_quat: torch.Tensor,
  pcd: torch.Tensor,
) -> torch.Tensor:
  """Per-keypoint distance to the nearest affordance cloud point.

  Args:
    keypoints_w: (B, K, 3) keypoint positions in the world frame.
    obj_pos: (B, 3) object position in the world frame.
    obj_quat: (B, 4) object orientation (wxyz) in the world frame.
    pcd: (P, 3) or (B, P, 3) affordance cloud in the object frame.

  Returns:
    (B, K) distances.
  """
  obj_quat_k = obj_quat.unsqueeze(1).expand(-1, keypoints_w.shape[1], -1)
  keypoints_obj = quat_apply(quat_inv(obj_quat_k), keypoints_w - obj_pos.unsqueeze(1))
  _, min_dist = nearest_affordance_points(keypoints_obj, pcd)
  return min_dist


def compute_af_vec(
  keypoints_w: torch.Tensor,
  obj_pos: torch.Tensor,
  obj_quat: torch.Tensor,
  pcd: torch.Tensor,
) -> torch.Tensor:
  """Flattened hand-centric affordance vectors for one batch.

  Keypoints move to the object frame, the nearest cloud point per keypoint
  gives the vector ``p - kp``, which is rotated back to the world frame.

  Args:
    keypoints_w: (B, K, 3) keypoint positions in the world frame.
    obj_pos: (B, 3) object position in the world frame.
    obj_quat: (B, 4) object orientation (wxyz) in the world frame.
    pcd: (P, 3) or (B, P, 3) affordance cloud in the object frame.

  Returns:
    (B, K * 3) distance vectors in the world frame.
  """
  obj_quat_k = obj_quat.unsqueeze(1).expand(-1, keypoints_w.shape[1], -1)
  keypoints_obj = quat_apply(quat_inv(obj_quat_k), keypoints_w - obj_pos.unsqueeze(1))
  nearest, _ = nearest_affordance_points(keypoints_obj, pcd)
  af_vec_obj = nearest - keypoints_obj
  af_vec_w = quat_apply(obj_quat_k, af_vec_obj)
  return af_vec_w.reshape(af_vec_w.shape[0], -1)


class AffordanceVectors:
  """Hand-centric distance vectors to the nearest affordance cloud point.

  Keypoints are the wrist, the 18 finger joints, and the 5 fingertip pads;
  the surface cloud is precomputed in the object frame and tracked with the
  live object pose (privileged teacher information).
  """

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
    self._variant_ids = env.sim.world_to_variant.get(cfg.params["object_entity"])
    if self._variant_ids is None:
      self._variant_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    obj_pos = self._object.data.root_link_pos_w
    obj_quat = self._object.data.root_link_quat_w
    keypoints_w = self._robot.data.body_link_pos_w[:, self._keypoint_ids]
    return compute_af_vec(keypoints_w, obj_pos, obj_quat, self._pcd[self._variant_ids])
