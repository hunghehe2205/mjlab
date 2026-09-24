"""Privileged grasp-only teacher after RobustDexGrasp, with its lift test."""

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
from mjlab.envs.mdp import reset_root_state_uniform, time_out
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
  ARM_IK_SEED,
  ARM_MAX_OFFSET,
  BOX_SIZE,
  FINGER_ROOTS,
  GRASP_TIME,
  HAND_ACTION_SCALE,
  HAND_BODIES,
  HAND_MAX_OFFSET,
  LIFT_RAMP,
  LIFT_TIME,
  MIN_ACTION_STD,
  OBJECT_POS,
  OPEN_HAND,
  POOL_SIZE,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp import (
  observations,
  rewards,
  signals,
  terminations,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspActionCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.events import PregraspReset
from mjlab.tasks.ur5e_rh5dg2.grasp.robot import teacher_robot_spec
from mjlab.tasks.ur5e_rh5dg2.view_env_cfg import ur5e_rh5dg2_ppo_runner_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


def teacher_env_cfg(
  play: bool = False, lift_test: bool | None = None
) -> ManagerBasedRlEnvCfg:
  """Grasp-only episodes; the lift test (default in play) appends a scripted raise."""
  lift_test = play if lift_test is None else lift_test
  robot = deepcopy(get_ur5e_rh5dg2_robot_cfg())
  robot.spec_fn = teacher_robot_spec
  joint_names = (
    *SIZE3_JOINTS,
    *SIZE1_JOINTS,
    *(f"{name}_joint" for name in HAND_BODIES[1:]),
  )
  robot.init_state.joint_pos = dict(
    zip(joint_names, (*ARM_IK_SEED, *OPEN_HAND), strict=True)
  )
  obj = get_box_cfg(BOX_SIZE, mass=0.08)
  obj.init_state.pos = OBJECT_POS
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
  hand = SceneEntityCfg("robot", body_names=HAND_BODIES)
  arm = SceneEntityCfg("robot", joint_names=(*SIZE3_JOINTS, *SIZE1_JOINTS))
  object_match = ContactMatch(mode="body", pattern="object", entity="object")
  hand_match = ContactMatch(mode="body", pattern=HAND_BODIES, entity="robot")
  sensors = (
    ContactSensorCfg(
      name="hand_object",
      primary=hand_match,
      secondary=object_match,
      secondary_policy="error",
    ),
    ContactSensorCfg(
      name="hand_object_world",
      primary=hand_match,
      secondary=object_match,
      secondary_policy="error",
      fields=("found", "force", "normal", "tangent"),
      global_frame=True,
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
    for name in ("robot", "props")
  }
  events["reset_pregrasp"] = EventTermCfg(
    func=PregraspReset,
    mode="reset",
    params={"pool_size": POOL_SIZE, "edge_biased": not (play or lift_test)},
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
        max_offset={
          **{name: ARM_MAX_OFFSET for name in (*SIZE3_JOINTS, *SIZE1_JOINTS)},
          "R_.*_joint": HAND_MAX_OFFSET,
        },
        random_delay=True,
        grasp_time=GRASP_TIME if lift_test else None,
        lift_ramp=LIFT_RAMP,
      )
    },
    events=events,
    rewards={
      "contact": RewardTermCfg(func=rewards.contact, weight=1.5),
      "grip": RewardTermCfg(func=rewards.grip, weight=1.0),
      "distance": RewardTermCfg(
        func=rewards.distance, weight=-0.5, params={"anchors": anchors}
      ),
      "crash": RewardTermCfg(func=rewards.crash, weight=-1.0),
      "clearance": RewardTermCfg(
        func=rewards.height_cost, weight=-0.1, params={"links": links}
      ),
      "undesired_contact": RewardTermCfg(func=rewards.undesired_contact, weight=-0.2),
      "object_displacement": RewardTermCfg(
        func=rewards.object_displacement, weight=-5.0
      ),
      "object_velocity": RewardTermCfg(func=rewards.object_velocity, weight=-15.0),
      "object_angular_velocity": RewardTermCfg(
        func=rewards.object_angular_velocity, weight=-0.2
      ),
      "wrist_velocity": RewardTermCfg(
        func=rewards.wrist_velocity, weight=-1.0, params={"palm": palm}
      ),
      "wrist_angular_velocity": RewardTermCfg(
        func=rewards.wrist_angular_velocity, weight=-0.1, params={"palm": palm}
      ),
      "arm_joint_velocity": RewardTermCfg(
        func=rewards.arm_joint_velocity, weight=-1.0, params={"arm": arm}
      ),
    },
    terminations={
      "hand_below_table": TerminationTermCfg(
        func=terminations.hand_below_table, params={"links": hand}
      ),
      "dropped": TerminationTermCfg(func=terminations.dropped),
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
    },
    metrics={
      "fingers_in_contact": MetricsTermCfg(
        func=signals.fingers_in_contact, reduce="last"
      ),
      "object_displacement": MetricsTermCfg(
        func=rewards.object_displacement, reduce="last"
      ),
      "object_speed": MetricsTermCfg(func=signals.object_speed),
      "contact_force": MetricsTermCfg(func=rewards.peak_contact_force, reduce="max"),
      **(
        {
          "lift_height": MetricsTermCfg(func=signals.lift_height, reduce="last"),
          "success": MetricsTermCfg(func=terminations.lifted, reduce="last"),
        }
        if lift_test
        else {}
      ),
    },
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005, iterations=50, ls_iterations=20, cone="elliptic", impratio=10
      ),
      nconmax=256,
      njmax=1024,
    ),
    decimation=10,
    episode_length_s=1e9 if play else GRASP_TIME + (LIFT_TIME if lift_test else 0.0),
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
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["std_range"] = (MIN_ACTION_STD, 1e6)
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
