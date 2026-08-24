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
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.dexgrasp import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

##
# Scene geometry (world frame). The arm base sits 4 cm below the tabletop so
# the wrist/hand are visually centred at the table height in the pre-grasp view.
##

TABLE_TOP_Z = 0.771
ARM_MOUNT_Z = TABLE_TOP_Z - 0.04
TABLE_FRICTION = 0.2

# 1.2 m x 1.1 m table. Keep its near edge at y=-0.2 so it remains separated
# from the pedestal while expanding the usable area away from the robot.
TABLE_HALF = (0.60, 0.55, TABLE_TOP_Z / 2)
TABLE_CENTER = (0.0, -0.75, TABLE_TOP_Z / 2)
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
      # Clamp the absolute target to soft joint limits (fingers have no actuator
      # ctrlrange to bound them; the delta itself stays unclipped -- see §D).
      clip_to_joint_limits=True,
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
    scale_rewards_by_dt=False,  # Reference rewards are per control step.
    reward_clip_min=-2.0,  # Reference total-reward floor.
  )
