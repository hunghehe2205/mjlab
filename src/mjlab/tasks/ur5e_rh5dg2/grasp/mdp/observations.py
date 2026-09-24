"""Privileged observations from batched simulation state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.scenes.workstation import TABLE_TOP_Z
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import FORCE_THRESHOLD
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspAction
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import (
  box_surface_vectors,
  normal_force,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def tracking_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = env.action_manager.get_term("joint_pos")
  assert isinstance(action, GraspAction)
  return env.scene["robot"].data.joint_pos[:, action.target_ids] - action.target


def teacher_state(
  env: ManagerBasedRlEnv, anchors: SceneEntityCfg, palm: SceneEntityCfg
) -> torch.Tensor:
  robot = env.scene["robot"].data
  obj = env.scene["object"].data
  points = robot.body_link_pos_w[:, anchors.body_ids]
  distances = box_surface_vectors(points, obj.root_link_pos_w, obj.root_link_quat_w)
  heights = points[..., 2] - env.scene.env_origins[:, None, 2] - TABLE_TOP_Z
  palm_pose = robot.body_link_pose_w[:, palm.body_ids].squeeze(1).clone()
  palm_pose[:, :3] -= env.scene.env_origins
  obj_pose = obj.root_link_pose_w.clone()
  obj_pose[:, :3] -= env.scene.env_origins
  force = torch.cat(
    [
      normal_force(env, name)
      for name in ("hand_object", "robot_table", "hand_self", "arm_object")
    ],
    dim=-1,
  )
  contacts = (force > FORCE_THRESHOLD).float()
  return torch.cat(
    [
      distances.flatten(1),
      heights.clamp(-0.1, 1.5),
      palm_pose,
      tracking_error(env),
      robot.joint_pos,
      robot.joint_vel,
      obj_pose,
      obj.root_link_vel_w,
      contacts,
      torch.log1p(force.clamp(max=20.0)),
    ],
    dim=-1,
  )
