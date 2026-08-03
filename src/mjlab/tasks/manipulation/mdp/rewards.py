from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  MultiCubeLiftingCommand,
  get_pick_place_command,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def multi_cube_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for the target cube selected by MultiCubeLiftingCommand."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = command.target_object_pos()
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def multi_cube_bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian reward for bringing the selected target cube to goal."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, MultiCubeLiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a MultiCubeLiftingCommand, got {type(command)}"
    )
  obj_pos_w = command.target_object_pos()
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  return torch.exp(-position_error / std**2)


def pick_place_reach(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Approach the cube, but only while not holding it and not finished.

  Gating on ``grasping`` rather than on a latched flag is what removes the
  grasp-and-park optimum: holding the cube still at the pick location pays
  nothing for reaching, while a policy that drops the cube gets its dense
  guidance back and can go collect it again.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = get_pick_place_command(env, command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  error = torch.sum(torch.square(ee_pos_w - obj.data.root_link_pos_w), dim=-1)
  gate = (1.0 - command.grasping) * (1.0 - command.placed_done)
  return gate * torch.exp(-error / std**2)


def pick_place_grasp(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Both fingertips loaded against the cube."""
  return get_pick_place_command(env, command_name).grasping


def pick_place_lift(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Height above the floor, ramped to 1 at ``lift_height``, gated on a grasp."""
  obj: Entity = env.scene[object_name]
  command = get_pick_place_command(env, command_name)
  floor_z = command.cfg.floor_z
  span = max(command.cfg.lift_height - floor_z, 1e-6)
  height = (obj.data.root_link_pos_w[:, 2] - floor_z) / span
  return height.clamp(0.0, 1.0) * command.grasping


def pick_place_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  """Bring the cube to the goal, gated on currently holding it.

  The gate is the anti-push and anti-drag invariant. With the goal on the floor
  an ungated kernel is maximized by bulldozing the cube along the ground, which
  needs no grasp at all and is far easier to find than a pick. Used twice, with
  a wide kernel for transport and a sharp one for the final placement.
  """
  obj: Entity = env.scene[object_name]
  command = get_pick_place_command(env, command_name)
  error = torch.sum(torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1)
  return command.grasping * torch.exp(-error / std**2)


def pick_place_release(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Retreat from a cube that is already down, settled and on target.

  Ramps with end-effector distance once ``placed_ok`` holds, so it starts
  paying while the cube is being set down and saturates once the hand is clear.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = get_pick_place_command(env, command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - obj.data.root_link_pos_w, dim=-1)
  ramp = (distance / command.cfg.release_dist).clamp(0.0, 1.0)
  return command.placed_ok * ramp


def pick_place_hold(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Dense payment for every step the finished state holds.

  This, not the one-shot bonus, is what discriminates a policy that finishes
  from one that finishes and then re-grips to farm the dense terms: a one-shot
  is latched, so both policies collect it exactly once and it cancels out of
  every comparison between them no matter how large its weight.
  """
  return get_pick_place_command(env, command_name).success_now


def pick_place_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-shot bonus on the rising edge of a completed hold window.

  Rewards are scaled by ``weight * dt`` with ``dt = 0.02``, so a one-shot needs
  a weight around ``fraction_active * T`` to be commensurate with a dense term
  -- roughly 400 here. Treat it as a tiebreaker on top of :func:`pick_place_hold`
  rather than as the primary signal.
  """
  return get_pick_place_command(env, command_name).fired_now


def pick_place_dropped(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  height: float = 0.06,
) -> torch.Tensor:
  """Penalize a cube left airborne without a grasp after it has been lifted.

  Gated on the latched ``lifted`` so it cannot tax the approach phase, and on
  height so a cube resting at the goal never triggers it.
  """
  obj: Entity = env.scene[object_name]
  command = get_pick_place_command(env, command_name)
  airborne = obj.data.root_link_pos_w[:, 2] > height
  return command.lifted * airborne.float() * (1.0 - command.grasping)


def action_rate_l2_subset(
  env: ManagerBasedRlEnv,
  indices: tuple[int, ...],
) -> torch.Tensor:
  """Action-rate penalty restricted to selected action dimensions.

  A grasp reward at weight 2.0 held at 50% duty collects 20.0 per episode while
  chattering one action dimension by the full range every step costs only 0.80
  at the shipped ``action_rate_l2`` weight of -0.01. Break-even is around 0.25 --
  25x the global value, which the lift task trains fine at -- so the gripper
  dimension needs its own penalty rather than a global increase.
  """
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta[:, list(indices)]), dim=1)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)
