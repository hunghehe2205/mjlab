"""DexGrasp teacher observation terms (Phase 1 §E).

Layout mirrors the reference environment: absolute qpos, PD error, per-body
contact flags + accumulated impulse magnitudes, keypoint/arm-link heights above
the table, hand center position, unwrapped wrist euler and the init-relative
wrist euler, plus the hand-centric affordance distance vectors (af_vec).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.dexgrasp.rotations import euler_from_rotmat, unwrap_euler
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "SKIP_IMPULSE",
  "FLAG_IMPULSE",
  "joint_pos",
  "pd_error",
  "link_heights",
  "hand_center_pos",
  "HandObjectContacts",
  "WristOrientation",
  "nearest_affordance_points",
  "keypoint_min_distances",
  "compute_af_vec",
  "AffordanceVectors",
]

# Reference impulse thresholds (N*s): contributions below SKIP_IMPULSE are
# dropped and a body flags as in-contact above FLAG_IMPULSE. Baseline for
# MuJoCo; recalibrate if the soft contact solver changes the impulse scale.
SKIP_IMPULSE = 0.001
FLAG_IMPULSE = 0.01

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
  from mjlab.envs.mdp.actions import RelativeJointPositionAction

  robot: Entity = env.scene[asset_cfg.name]
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, RelativeJointPositionAction)
  target = term.target[:, asset_cfg.joint_ids]
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
  """Grasp-center site position in the world frame."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


class HandObjectContacts:
  """Contact flags + accumulated impulse magnitudes per hand contact body.

  The impulse is the net contact force summed over the control step's
  substeps, scaled by the physics timestep, so it matches the reference
  accumulated-impulse bookkeeping.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    self._sensor = sensor
    self._dt = float(env.sim.cfg.mujoco.timestep)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    history = self._sensor.data.force_history
    assert history is not None
    impulse = history.sum(dim=2) * self._dt  # (B, P, 3)
    magnitude = impulse.norm(dim=-1)
    # The reference drops per-contact impulses below SKIP_IMPULSE before
    # accumulating; netforce sums first, so floor the accumulated magnitude.
    magnitude = torch.where(
      magnitude < SKIP_IMPULSE, torch.zeros_like(magnitude), magnitude
    )
    flags = (magnitude > FLAG_IMPULSE).float()
    return torch.cat([flags, magnitude], dim=-1)


class WristOrientation:
  """Wrist euler (unwrapped) + init-relative wrist euler.

  The init rotation snapshot and the unwrap memory are stateful; the snapshot
  is (re)captured on the first compute after a reset, when forward kinematics
  of the new state are available.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._robot = env.scene[cfg.params["asset_cfg"].name]
    self._wrist_ids = cfg.params["asset_cfg"].body_ids
    n = env.num_envs
    self._rot_init = torch.eye(3, device=env.device).expand(n, 3, 3).contiguous()
    self._euler_prev = torch.zeros((n, 3), device=env.device)
    self._pending = torch.ones(n, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._pending[ids] = True
    self._euler_prev[ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    quat = self._robot.data.body_link_quat_w[:, self._wrist_ids[0]]
    rot = matrix_from_quat(quat)
    if bool(self._pending.any()):
      self._rot_init[self._pending] = rot[self._pending]
      self._pending[:] = False
    euler = unwrap_euler(euler_from_rotmat(rot), self._euler_prev)
    self._euler_prev[:] = euler
    diff = euler_from_rotmat(self._rot_init.transpose(-1, -2) @ rot)
    return torch.cat([euler, diff], dim=-1)


def nearest_affordance_points(
  keypoints_obj: torch.Tensor, pcd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Nearest cloud point per keypoint (object frame) and the distance to it.

  Args:
    keypoints_obj: (..., K, 3) keypoint positions in the object frame.
    pcd: (P, 3) affordance cloud in the object frame.

  Returns:
    ((..., K, 3) nearest points, (..., K) distances).
  """
  dists = torch.cdist(keypoints_obj, pcd)
  min_dist, min_idx = torch.min(dists, dim=-1)
  nearest = pcd[min_idx]
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
    pcd: (P, 3) affordance cloud in the object frame.

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
    pcd: (P, 3) affordance cloud in the object frame.

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
  the 200-point cloud is precomputed in the object frame and tracked with the
  live object pose (privileged teacher information).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    obj = oc.PHASE1_OBJECTS[cfg.params["object_name"]]
    self._pcd = torch.as_tensor(
      obj.load_surface_points(), dtype=torch.float32, device=env.device
    )
    self._robot = env.scene[cfg.params["asset_cfg"].name]
    self._object = env.scene[cfg.params["object_entity"]]
    self._keypoint_ids = cfg.params["asset_cfg"].body_ids

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    obj_pos = self._object.data.root_link_pos_w
    obj_quat = self._object.data.root_link_quat_w
    keypoints_w = self._robot.data.body_link_pos_w[:, self._keypoint_ids]
    return compute_af_vec(keypoints_w, obj_pos, obj_quat, self._pcd)
