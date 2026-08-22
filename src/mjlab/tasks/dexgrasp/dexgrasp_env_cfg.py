"""DexGrasp teacher env skeleton (robot-agnostic).

Builds the scene (table + arm pedestal + one grasp object), delta-joint action,
minimal proprio observations, and a time-out termination so the env runs with
zero action. Privileged observations (§E), the reward stack (§F), and the
analytic-IK pre-grasp reset (§C) are filled in later phases. Per-robot wiring
(robot entity, action scale, object placement) lives in ``config/<robot>/``.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.dexgrasp import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

##
# Scene geometry (world frame). Table top and pedestal top are flush; the arm
# base mounts on the pedestal at ``ARM_MOUNT_Z``. Tunable in the viewer.
##

TABLE_TOP_Z = 0.771
ARM_MOUNT_Z = 0.771  # Arm base height (pedestal top, flush with the table top).
TABLE_FRICTION = 0.2

TABLE_HALF = (0.40, 0.35, TABLE_TOP_Z / 2)
TABLE_CENTER = (0.0, -0.55, TABLE_TOP_Z / 2)
PEDESTAL_HALF = (0.12, 0.12, ARM_MOUNT_Z / 2)
PEDESTAL_CENTER = (0.0, 0.0, ARM_MOUNT_Z / 2)

# Control: 5 Hz policy (control_dt 0.2 s = decimation 20 x timestep 0.01 s),
# 70-step episode (14 s).
SIM_TIMESTEP = 0.01
DECIMATION = 20
EPISODE_LENGTH_S = 14.0

_TABLE_RGBA = (0.55, 0.40, 0.28, 1.0)
_PEDESTAL_RGBA = (0.30, 0.30, 0.32, 1.0)


def get_arena_spec() -> mujoco.MjSpec:
  """Static table + arm pedestal as one fixed body (auto-wrapped mocap)."""
  spec = mujoco.MjSpec()
  spec.add_material(name="table", rgba=_TABLE_RGBA)
  spec.add_material(name="pedestal", rgba=_PEDESTAL_RGBA)
  body = spec.worldbody.add_body(name="arena")
  # priority=1 so the table's low friction wins over contacting geoms; equal
  # priority would take the element-wise max, letting the object's default 1.0
  # override the intended 0.2.
  body.add_geom(
    name="table",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_HALF,
    pos=TABLE_CENTER,
    material="table",
    friction=[TABLE_FRICTION, 0.005, 0.0001],
    priority=1,
  )
  body.add_geom(
    name="pedestal",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=PEDESTAL_HALF,
    pos=PEDESTAL_CENTER,
    material="pedestal",
  )
  return spec


def make_dexgrasp_env_cfg() -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  # Teacher obs are privileged and clean; corruption is on (repo convention) but
  # no term has a noise model, so it is a no-op. Critic = actor (both privileged).
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,  # Override per-robot (arm 0.005 / finger 0.015).
    )
  }

  # Position mocap-wrapped fixed-base entities at env origins; object at default.
  events = {
    "reset_arena": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "asset_cfg": SceneEntityCfg("arena")},
    ),
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_object": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("object"),
      },
    ),
  }

  rewards = {
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.5,
      entities={"arena": EntityCfg(spec_fn=get_arena_spec)},
      sensors=(),
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.9,
      elevation=-18.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      # Headroom for a multi-finger grasp (lift_cube's 2-finger scene uses
      # 55/600); revisit once §H measures peak contacts.
      nconmax=150,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=SIM_TIMESTEP,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
  )
