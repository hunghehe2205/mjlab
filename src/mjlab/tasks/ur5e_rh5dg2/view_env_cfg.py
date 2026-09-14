"""Minimal viewable environment for the UR5e + rh5dg2 hand workstation.

No task reward yet — a trivial (zero-weight) reward keeps the manager happy so
the env can be viewed with ``play.py --agent zero`` / ``--agent random``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots import (
  UR5E_RH5DG2_ACTION_SCALE,
  get_ur5e_rh5dg2_robot_cfg,
)
from mjlab.asset_zoo.scenes.workstation import get_workstation_spec
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def ur5e_rh5dg2_view_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  robot_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  props_cfg = SceneEntityCfg("props")

  actor_terms = {
    "joint_pos": ObservationTermCfg(func=joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=UR5E_RH5DG2_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  events = {
    # Position each env's fixed-base entities at their env origin.
    "reset_robot_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}, "asset_cfg": robot_cfg},
    ),
    "reset_props_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}, "asset_cfg": props_cfg},
    ),
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": robot_cfg,
      },
    ),
  }

  rewards = {"alive": RewardTermCfg(func=_alive, weight=0.0)}
  terminations = {"time_out": TerminationTermCfg(func=time_out, time_out=True)}

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": get_ur5e_rh5dg2_robot_cfg(),
        "props": EntityCfg(spec_fn=get_workstation_spec),
      },
      num_envs=1,
      env_spacing=2.5,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=2.0,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=1e10,
  )
  if play:
    cfg.scene.num_envs = 4
    cfg.observations["actor"].enable_corruption = False
  return cfg


def ur5e_rh5dg2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="ur5e_rh5dg2_view",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
