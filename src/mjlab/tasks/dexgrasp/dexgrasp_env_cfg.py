"""Robot-agnostic DexGrasp teacher env config.

Scene (table, pedestal, one grasp object), delta-joint action, privileged
observation terms and time-out termination. Per-robot wiring (robot entity,
action scale, rewards, sensors, pre-grasp reset) lives in ``config/<robot>/``.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.dexgrasp import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# Scene geometry (world frame); arm base sits 4 cm below tabletop for a
# pre-grasp view centred at table height.

TABLE_TOP_Z = 0.771
ARM_MOUNT_Z = TABLE_TOP_Z - 0.04
# Reference friction: object-table pair is 0.2, every other pair the 0.8 default.
TABLE_FRICTION = 0.2
DEFAULT_FRICTION = 0.8

# 1.2m x 1.1m table; near edge at y=-0.2 keeps it clear of the pedestal.
TABLE_HALF = (0.60, 0.55, TABLE_TOP_Z / 2)
TABLE_CENTER = (0.0, -0.75, TABLE_TOP_Z / 2)
PEDESTAL_HALF = (0.12, 0.12, ARM_MOUNT_Z / 2)
PEDESTAL_CENTER = (0.0, 0.0, ARM_MOUNT_Z / 2)

# Control: 5 Hz policy (decimation 40 x timestep 0.005s), 70-step episode (14s).
SIM_TIMESTEP = 0.005
DECIMATION = 40  # control_dt stays 0.2 s (5 Hz)
EPISODE_LENGTH_S = 14.0

_TABLE_RGBA = (0.55, 0.40, 0.28, 1.0)
_PEDESTAL_RGBA = (0.30, 0.30, 0.32, 1.0)


def get_arena_spec() -> mujoco.MjSpec:
  """Static table + arm pedestal as one fixed body (auto-wrapped mocap).

  No friction priority is set: MuJoCo combines equal-priority frictions via
  elementwise max, so table 0.2 + object 0.2 -> 0.2 while hand 0.8 -> 0.8,
  matching the reference exactly. Priority=1 would leak the table's 0.2 onto
  hand-table contacts too.
  """
  spec = mujoco.MjSpec()
  spec.add_material(name="table", rgba=_TABLE_RGBA)
  spec.add_material(name="pedestal", rgba=_PEDESTAL_RGBA)
  body = spec.worldbody.add_body(name="arena")
  body.add_geom(
    name="table",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_HALF,
    pos=TABLE_CENTER,
    material="table",
    friction=[TABLE_FRICTION, 0.005, 0.0001],
  )
  body.add_geom(
    name="pedestal",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=PEDESTAL_HALF,
    pos=PEDESTAL_CENTER,
    material="pedestal",
    friction=[DEFAULT_FRICTION, 0.005, 0.0001],
  )
  return spec


def make_dexgrasp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Build the robot-agnostic teacher env config.

  ``reward_clip_min`` is left unset: the reference computes a -2.0 floor but
  discards it (train.py's ``reward_r.clip(...)`` return value is unused) and
  overwrites the -10 terminal with the raw sum, so neither reaches PPO. Both
  would be actively harmful here too -- affordance_distance alone is -1.87
  per step at the pre-grasp pose, leaving only 0.13 of headroom before the
  floor flattens the gradient (22.9% of steps clipped under a random policy).
  """
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "pd_error": ObservationTermCfg(
      func=mdp.pd_error,
      params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
    ),
    "contacts": ObservationTermCfg(
      func=mdp.HandObjectContacts,
      params={"sensor_name": "hand_object_contact"},
    ),
    "keypoint_heights": ObservationTermCfg(
      func=mdp.link_heights,
      params={"table_top_z": TABLE_TOP_Z, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "arm_link_heights": ObservationTermCfg(
      func=mdp.link_heights,
      params={"table_top_z": TABLE_TOP_Z, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "hand_center": ObservationTermCfg(
      func=mdp.hand_center_pos,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "wrist_orientation": ObservationTermCfg(
      func=mdp.WristOrientation,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "af_vec": ObservationTermCfg(
      func=mdp.AffordanceVectors,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "object_entity": "object",
      },
    ),
  }
  # Privileged, clean teacher obs; runner maps both actor and critic to this group.
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.ReferenceRelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,  # Override per-robot (arm 0.005 / finger 0.015).
      # Clamp target to soft limits (no finger ctrlrange); delta unclipped, see §D.
      clip_to_joint_limits=True,
      first_substep_delay_prob=0.5,
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

  rewards = {}  # §F reward stack is wired per-robot in config/<robot>/env_cfgs.py.

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
      # Headroom for multi-finger grasp (cf. lift_cube's 55/600); revisit after §H.
      nconmax=150,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=SIM_TIMESTEP,
        iterations=10,
        ls_iterations=20,
        impratio=1.0,
        cone="pyramidal",
      ),
    ),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
    scale_rewards_by_dt=False,  # Reference rewards are per control step.
    reward_clip_min=None,  # See docstring: reference's floor/terminal are no-ops.
    termination_reward=0.0,
  )
