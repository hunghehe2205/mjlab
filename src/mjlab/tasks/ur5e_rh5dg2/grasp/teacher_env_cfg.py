"""Privileged, fully policy-controlled grasp-and-lift baseline."""

from copy import deepcopy

from mjlab.asset_zoo.objects.primitives import get_box_cfg
from mjlab.asset_zoo.robots import get_ur5e_rh5dg2_robot_cfg
from mjlab.asset_zoo.robots.universal_robots_ur5e.ur5e_constants import (
  SIZE1_JOINTS,
  SIZE3_JOINTS,
)
from mjlab.asset_zoo.scenes.workstation import get_workstation_spec
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_joints_by_offset, reset_root_state_uniform, time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import (
  ARM_ACTION_SCALE,
  ARM_BODIES,
  BOX_SIZE,
  FINGER_ROOTS,
  HAND_ACTION_SCALE,
  HAND_BODIES,
  HOLD_TIME,
  LIFT_HEIGHT,
  OBJECT_POS,
  OPEN_HAND,
  PREGRASP_ARM,
  TIP_BODIES,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp import (
  observations,
  rewards,
  signals,
  terminations,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspActionCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.robot import teacher_robot_spec
from mjlab.tasks.ur5e_rh5dg2.view_env_cfg import ur5e_rh5dg2_ppo_runner_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


def teacher_env_cfg(
  play: bool = False, lift_height: float = LIFT_HEIGHT, hold_time: float = HOLD_TIME
) -> ManagerBasedRlEnvCfg:
  if lift_height <= 0 or hold_time <= 0:
    raise ValueError("Lift height and hold time must be positive.")
  robot = deepcopy(get_ur5e_rh5dg2_robot_cfg())
  robot.spec_fn = teacher_robot_spec
  joint_names = (
    *SIZE3_JOINTS,
    *SIZE1_JOINTS,
    *(f"{name}_joint" for name in HAND_BODIES[1:]),
  )
  robot.init_state.joint_pos = dict(
    zip(joint_names, (*PREGRASP_ARM, *OPEN_HAND), strict=True)
  )
  obj = get_box_cfg(BOX_SIZE, mass=0.08)
  obj.init_state.pos = OBJECT_POS
  tips = SceneEntityCfg("robot", body_names=TIP_BODIES, preserve_order=True)
  palm = SceneEntityCfg("robot", body_names=("R_hand_palm",))
  anchors = SceneEntityCfg(
    "robot",
    body_names=tuple(
      name.replace("_dip", "_force_sensor") if name.endswith("_dip") else name
      for name in HAND_BODIES
    ),
    preserve_order=True,
  )
  links = SceneEntityCfg(
    "robot", body_names=tuple(name for name in HAND_BODIES if not name.endswith("_dip"))
  )
  object_match = ContactMatch(mode="body", pattern="object", entity="object")
  sensors = (
    ContactSensorCfg(
      name="hand_object",
      primary=ContactMatch(mode="body", pattern=HAND_BODIES, entity="robot"),
      secondary=object_match,
      secondary_policy="error",
    ),
    ContactSensorCfg(
      name="finger_object",
      primary=ContactMatch(mode="subtree", pattern=FINGER_ROOTS, entity="robot"),
      secondary=object_match,
      secondary_policy="error",
    ),
    ContactSensorCfg(
      name="object_table",
      primary=object_match,
      secondary=ContactMatch(mode="body", pattern="table", entity="props"),
      secondary_policy="error",
    ),
    ContactSensorCfg(
      name="robot_table",
      primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="table", entity="props"),
      secondary_policy="error",
      history_length=10,
    ),
    ContactSensorCfg(
      name="hand_self",
      primary=ContactMatch(mode="body", pattern=HAND_BODIES, entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
      secondary_policy="error",
      history_length=10,
    ),
    ContactSensorCfg(
      name="arm_object",
      primary=ContactMatch(mode="body", pattern=ARM_BODIES, entity="robot"),
      secondary=object_match,
      secondary_policy="error",
      history_length=10,
    ),
  )
  events = {
    f"reset_{name}": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "asset_cfg": SceneEntityCfg(name)},
    )
    for name in ("robot", "props", "object")
  }
  events["reset_joints"] = EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  obs = {
    "state": ObservationTermCfg(
      func=observations.teacher_state, params={"anchors": anchors, "palm": palm}
    )
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={
        "robot": robot,
        "props": EntityCfg(spec_fn=get_workstation_spec),
        "object": obj,
      },
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=sensors,
      num_envs=1 if play else 256,
      env_spacing=2.5,
    ),
    observations={
      "actor": ObservationGroupCfg(deepcopy(obs), enable_corruption=not play),
      "critic": ObservationGroupCfg(deepcopy(obs), enable_corruption=False),
    },
    actions={
      "joint_pos": GraspActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        use_default_offset=False,
        scale={
          **{name: ARM_ACTION_SCALE for name in (*SIZE3_JOINTS, *SIZE1_JOINTS)},
          "R_.*_joint": HAND_ACTION_SCALE,
        },
      )
    },
    events=events,
    rewards={
      "reach": RewardTermCfg(func=rewards.reach, weight=1.0, params={"tips": tips}),
      "contact": RewardTermCfg(func=rewards.contact, weight=0.5),
      "lift": RewardTermCfg(
        func=rewards.lift, weight=6.0, params={"height": lift_height}
      ),
      "hold": RewardTermCfg(
        func=rewards.hold, weight=2.0, params={"height": lift_height}
      ),
      "success": RewardTermCfg(func=rewards.completion, weight=150.0),
      "clearance": RewardTermCfg(
        func=rewards.height_cost, weight=-0.1, params={"links": links}
      ),
      "undesired_contact": RewardTermCfg(func=rewards.undesired_contact, weight=-0.2),
      "object_displacement_xy": RewardTermCfg(
        func=rewards.horizontal_displacement, weight=-5.0
      ),
      "object_velocity_xy": RewardTermCfg(
        func=rewards.horizontal_velocity, weight=-1.0
      ),
      "joint_velocity": RewardTermCfg(func=rewards.joint_velocity, weight=-0.01),
      "palm_velocity": RewardTermCfg(
        func=rewards.palm_velocity, weight=-0.01, params={"palm": palm}
      ),
    },
    terminations={
      "success": TerminationTermCfg(
        func=terminations.LiftSuccess,
        params={"height": lift_height, "hold_time": hold_time},
      ),
      "dropped": TerminationTermCfg(func=terminations.dropped),
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
    },
    metrics={
      "lift_height": MetricsTermCfg(func=signals.lift_height),
      "object_speed": MetricsTermCfg(func=signals.object_speed),
      "contact_force": MetricsTermCfg(func=rewards.peak_contact_force),
      "hold_progress": MetricsTermCfg(func=terminations.hold_progress, reduce="last"),
      "success": MetricsTermCfg(func=terminations.success, reduce="last"),
    },
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005, iterations=50, ls_iterations=20, cone="elliptic", impratio=10
      ),
      nconmax=256,
      njmax=1024,
    ),
    decimation=10,
    episode_length_s=1e9 if play else 12.0,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="object",
      body_name="object",
      distance=1.5,
      elevation=-20.0,
      azimuth=130.0,
    ),
  )


def teacher_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = ur5e_rh5dg2_ppo_runner_cfg()
  cfg.actor.hidden_dims = (128, 128)
  cfg.critic.hidden_dims = (128, 128)
  cfg.algorithm.gamma = 0.996
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.learning_rate = 3e-4
  cfg.algorithm.value_loss_coef = 0.5
  cfg.algorithm.max_grad_norm = 0.5
  cfg.algorithm.num_learning_epochs = 4
  cfg.num_steps_per_env = 64
  cfg.max_iterations = 5000
  cfg.clip_actions = 1.0
  cfg.experiment_name = "ur5e_rh5dg2_grasp_teacher"
  return cfg
