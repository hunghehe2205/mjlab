"""Online diagnostics for the DexGrasp teacher, logged as ``Episode_Metrics``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_root_pos_state(obj: Entity) -> torch.Tensor:
  """Root position read from qpos/mocap state, valid before forward().

  Reset-time readers must use this: events write qpos inside _reset_idx and
  kinematics (xpos) refresh only at the later forward() call.
  """
  if not obj.is_fixed_base:
    q_adr = obj.indexing.free_joint_q_adr[:3]
    return obj.data.data.qpos[:, q_adr]
  mocap_id = obj.indexing.mocap_id
  if mocap_id is not None:
    return obj.data.data.mocap_pos[:, mocap_id]
  return obj.data.root_link_pos_w


def object_tilt_angle(quat: torch.Tensor) -> torch.Tensor:
  """Angle (rad) between the object body z-axis and world z."""
  rot = matrix_from_quat(quat)
  return torch.arccos(rot[..., 2, 2].clamp(-1.0, 1.0))


def object_tilt_deg(
  env: ManagerBasedRlEnv, object_entity: str = "object"
) -> torch.Tensor:
  """Object tilt from upright in degrees; 45+ means it was knocked over."""
  obj: Entity = env.scene[object_entity]
  return object_tilt_angle(obj.data.root_link_quat_w) * (180.0 / math.pi)


def joint_pos_mean(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Mean position of the selected joints."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=-1)


def mean_arm_action_magnitude(
  env: ManagerBasedRlEnv, arm_action_dim: int
) -> torch.Tensor:
  """Mean magnitude of raw policy commands for the arm action prefix."""
  return env.action_manager.action[:, :arm_action_dim].abs().mean(dim=1)
