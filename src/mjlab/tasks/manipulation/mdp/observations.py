from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
  PickPlaceCommand,
  get_pick_place_command,
)
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to object in base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  distance_vec_w = obj_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def object_to_goal_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from object to goal in base frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (LiftingCommand, PickPlaceCommand)):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand or PickPlaceCommand, "
      f"got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos
  distance_vec_w = goal_pos_w - obj_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def ee_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """EE linear velocity in EE frame."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 6)
  ee_vel_linear_w = ee_vel_w[:, :3]
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_linear_ee = quat_apply(quat_inv(ee_quat_w), ee_vel_linear_w)
  return ee_vel_linear_ee


def target_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Target position in EE frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(
    command, (LiftingCommand, MultiCubeLiftingCommand, PickPlaceCommand)
  ):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, "
      f"MultiCubeLiftingCommand or PickPlaceCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  target_pos_w = command.target_pos
  target_pos_rel_w = target_pos_w - ee_pos_w
  target_pos_ee = quat_apply(quat_inv(ee_quat_w), target_pos_rel_w)
  return target_pos_ee


def fingertip_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-fingertip contact flag against the manipulated object. Shape (B, P).

  ``ContactData.found`` is a match *count*, not a boolean -- values of 1, 2, 3,
  6, 7, 8 and 10-17 all occur for a single fingertip -- so it is thresholded
  rather than cast. Kept in the actor group deliberately: the real YAM senses
  the same thing through gripper current, so this does not open a sim-to-real
  gap the way a privileged pose observation would.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def contact_force_magnitude(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-primary contact force magnitude. Shape (B, P). Critic-side."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return torch.norm(sensor.data.force, dim=-1)


def object_velocity(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  """Object linear and angular velocity in world frame. Shape (B, 6).

  Privileged: it tells the value function whether a placement is about to
  settle or about to roll off, which is exactly the distinction the reward's
  ``at_rest`` predicate turns on.
  """
  obj: Entity = env.scene[object_name]
  return torch.cat([obj.data.root_link_lin_vel_w, obj.data.root_link_ang_vel_w], dim=-1)


def pick_place_stage_flags(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Latched stage flags ``[ever_grasped, lifted, placed_done]``. Shape (B, 3).

  Required, not optional. Whether the policy should hold or let go depends on
  latched history, so without these the observation is not a Markov state and
  the correct policy is not representable.
  """
  command = get_pick_place_command(env, command_name)
  return torch.stack(
    [command.ever_grasped, command.lifted, command.placed_done], dim=-1
  )


def pick_place_hold_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Fraction of the success hold window completed. Shape (B, 1).

  Also a Markov requirement: success needs the predicates to hold continuously,
  so the policy has to know how much of the window it has banked.
  """
  command = get_pick_place_command(env, command_name)
  progress = command.hold_counter / max(command.cfg.hold_steps, 1)
  return progress.clamp(0.0, 1.0).unsqueeze(-1)


def pick_place_hold_counter(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Raw hold-window counter, in control steps. Shape (B,). Metrics only.

  Log this with ``reduce="mean"`` and ``reduce="max"`` from the very first
  smoke run. It is the only signal that separates the two failure modes: if the
  max rarely exceeds ~5 the stage flags are flickering and no amount of policy
  training will fix it, whereas a healthy max that sits at ``hold_steps`` with a
  low mean simply means success is still rare.
  """
  return get_pick_place_command(env, command_name).hold_counter


def pick_place_flag(
  env: ManagerBasedRlEnv, command_name: str, flag: str
) -> torch.Tensor:
  """Named stage flag as a (B,) float, for the metrics manager."""
  command = get_pick_place_command(env, command_name)
  value = getattr(command, flag)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Flag '{flag}' on PickPlaceCommand is not a tensor.")
  return value.float()


def contact_force_peak(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Largest contact force magnitude across primaries. Shape (B,). Metrics only.

  Paired with ``reduce="max"`` this reports the per-episode peak, which is how
  you tell a termination threshold that is correctly parked from one you have
  merely retuned into a new margin.

  Reads ``force_history`` when the sensor has one, so that it measures the same
  quantity ``illegal_contact`` decides on. That function scans every substep of
  the history (``terminations.py:19-23``), while ``data.force`` holds only the
  last of the four. Auditing the guard against the instantaneous value
  under-reports its own decision variable by 25-27% at the peaks that matter.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    # [B, N, H, 3] -> max over both primaries and substeps.
    return torch.norm(sensor.data.force_history, dim=-1).amax(dim=-1).amax(dim=-1)
  assert sensor.data.force is not None
  return torch.norm(sensor.data.force, dim=-1).amax(dim=-1)


def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """RGB observation in CNN-compatible format (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb  # (B, H, W, 3)
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2)  # (B, 3, H, W)
  return rgb_data.float() / 255.0


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)


def camera_segmentation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Per-pixel typed segmentation in (B, 2, H, W) format."""
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  return seg_data.permute(0, 3, 1, 2)  # (B, 2, H, W)


def camera_target_cube_mask(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Binary mask of the target cube selected by a MultiCubeLiftingCommand.

  Output shape: (B, 1, H, W) float32.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  obj_ids = seg_data[..., 0]  # (B, H, W)
  obj_types = seg_data[..., 1]  # (B, H, W)

  command = env.command_manager.get_term(command_name)
  assert isinstance(command, MultiCubeLiftingCommand)
  target_ids = command.target_geom_ids  # (B, K)

  # Only geom hits should participate in the target mask.
  is_geom = obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)
  mask = (obj_ids.unsqueeze(-1) == target_ids.unsqueeze(1).unsqueeze(1)).any(-1)
  mask = mask & is_geom
  return mask.float().unsqueeze(1)  # (B, 1, H, W)
