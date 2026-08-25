from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_linear_speed(
  env: ManagerBasedRlEnv, object_entity: str = "object"
) -> torch.Tensor:
  """Object root linear speed in metres per second."""
  obj: Entity = env.scene[object_entity]
  return torch.linalg.vector_norm(obj.data.root_link_lin_vel_w, dim=-1)


def object_angular_speed(
  env: ManagerBasedRlEnv, object_entity: str = "object"
) -> torch.Tensor:
  """Object root angular speed in radians per second."""
  obj: Entity = env.scene[object_entity]
  return torch.linalg.vector_norm(obj.data.root_link_ang_vel_w, dim=-1)


def hand_keypoint_below_table_depth(
  env: ManagerBasedRlEnv,
  table_top_z: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Greatest distance any monitored hand keypoint is below the tabletop height.

  This is the same global-Z guard as ``hand_below_table``; it deliberately does
  not imply physical penetration of the finite XY table collider. A value of
  zero means every monitored keypoint is at or above the tabletop height.
  """
  robot: Entity = env.scene[asset_cfg.name]
  keypoint_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  table_z = env.scene.env_origins[:, 2] + table_top_z
  return (table_z.unsqueeze(1) - keypoint_z).clamp_min(0.0).amax(dim=1)


def mean_arm_action_magnitude(
  env: ManagerBasedRlEnv, arm_action_dim: int
) -> torch.Tensor:
  """Mean magnitude of raw policy commands for the arm action prefix."""
  return env.action_manager.action[:, :arm_action_dim].abs().mean(dim=1)


class ObjectLiftHeight:
  """Track object vertical displacement from each episode's reset pose."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._object: Entity = env.scene[cfg.params.get("object_entity", "object")]
    self._initial_z = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    return self._object.data.root_link_pos_w[:, 2] - self._initial_z

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initial_z[env_ids] = self._object.data.root_link_pos_w[env_ids, 2]


class LiftSuccess(ObjectLiftHeight):
  """Whether the object has risen by ``success_height`` during the episode.

  This is an online rollout diagnostic. The checkpoint evaluator remains the
  benchmark metric because it runs the post-grasp scripted lift.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._success_height = float(cfg.params["success_height"])

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    return (super().__call__(env, **kwargs) > self._success_height).float()
