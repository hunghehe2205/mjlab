"""UR5e + RH5-DG2 wiring for the DexGrasp env."""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.dexgrasp import mdp
from mjlab.tasks.dexgrasp.dexgrasp_env_cfg import (
  ARM_MOUNT_Z,
  DECIMATION,
  TABLE_CENTER,
  TABLE_HALF,
  TABLE_TOP_Z,
  make_dexgrasp_env_cfg,
)
from mjlab.tasks.dexgrasp.mdp.rewards import (
  REWARD_COEFFS,
  contact_weights,
)

# Actuator-name regexes -> delta-action scale (arm 0.005, finger 0.015 rad).
# Like RobustDexGrasp, scale is a residual gain, not a per-step cap: the delta is
# unbounded and only the absolute target is limited (arm actuator ctrlrange;
# fingers by joint limits). §D adds explicit soft-limit target clipping.
ACTION_SCALE = {
  r"(shoulder|elbow|wrist).*": rc.ACTION_SCALE_ARM,
  r"R_.*": rc.ACTION_SCALE_FINGER,
}

SKELETON_OBJECT = "potted_meat_can"
OBJECT_XY = (0.0, -0.6)  # Polar r~0.6, theta=-0.5pi (in the sampling region).
HAND_TABLE_TERMINATION_TOLERANCE = 0.005
PHASE1_OBJECT_NAMES = (
  "potted_meat_can",
  "tomato_soup_can",
  "tuna_fish_can",
  "sugar_box",
  "pudding_box",
)

# Reset unrecoverable objects before off-table motion reaches unstable high spin.
_OBJECT_XY_MARGIN = 0.10
OBJECT_WORKSPACE_BOUNDS = (
  (
    TABLE_CENTER[0] - TABLE_HALF[0] - _OBJECT_XY_MARGIN,
    TABLE_CENTER[0] + TABLE_HALF[0] + _OBJECT_XY_MARGIN,
  ),
  (
    TABLE_CENTER[1] - TABLE_HALF[1] - _OBJECT_XY_MARGIN,
    TABLE_CENTER[1] + TABLE_HALF[1] + _OBJECT_XY_MARGIN,
  ),
  (TABLE_TOP_Z - 0.05, TABLE_TOP_Z + 0.50),
)


def get_dexgrasp_robot_cfg() -> EntityCfg:
  """UR5e + RH5-DG2 with its base raised onto the pedestal.

  Fingers start at the cupped pre-grasp so reset lands on a valid pose; leaving
  them at 0 gets silently clamped up to the soft joint limits. §C overrides the
  arm via IK per reset.
  """
  joint_pos = {**(rc.HOME_KEYFRAME.joint_pos or {}), **rc.INIT_FINGER_POSE}
  init_state = dataclasses.replace(
    rc.HOME_KEYFRAME, pos=(0.0, 0.0, ARM_MOUNT_Z), joint_pos=joint_pos
  )
  return EntityCfg(
    init_state=init_state,
    spec_fn=rc.get_spec,
    articulation=rc.ARTICULATION,
  )


def get_skeleton_object_cfg(name: str, fixed: bool = False) -> EntityCfg:
  obj = oc.PHASE1_OBJECTS[name]
  init_state = EntityCfg.InitialStateCfg(
    pos=(OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z - obj.lowest_point),
  )
  return EntityCfg(
    init_state=init_state,
    spec_fn=lambda: oc.get_mesh_object_spec(name, fixed=fixed),
  )


def get_dexgrasp_object_cfg(object_names: tuple[str, ...]) -> EntityCfg:
  """Create a fixed object or a per-world mesh-variant object entity."""
  if len(object_names) == 1:
    return get_skeleton_object_cfg(object_names[0])
  if object_names == oc.ROBUST_DEXGRASP_TRAIN_OBJECTS:
    return oc.get_robustdexgrasp_variant_cfg()
  return oc.get_phase1_variant_cfg(object_names)


# Body-mode slots only see the body's own geoms, so the welded pad bodies get
# the 6 trailing slots; consumers fold them into the canonical 16 via
# rc.PAD_PARENT_INDICES.
_HAND_SENSOR_BODIES = rc.CONTACT_BODIES + rc.PAD_BODIES


def get_hand_object_contact_sensor() -> ContactSensorCfg:
  """Hand-vs-object contact sensor: 16 contact bodies + 6 pad slots.

  Literal compiled names keep the canonical CONTACT_BODIES-then-PAD_BODIES
  order on the per-primary axis; net-force + history gives one force vector
  per body per substep.
  """
  return ContactSensorCfg(
    name="hand_object_contact",
    primary=ContactMatch(
      mode="body",
      pattern=tuple(f"robot/{rc.HAND_PREFIX}{b}" for b in _HAND_SENSOR_BODIES),
    ),
    secondary=ContactMatch(mode="subtree", pattern="object", entity="object"),
    fields=("force",),
    reduce="netforce",
    history_length=DECIMATION,
  )


def _hand_table_sensor(name: str, secondary: ContactMatch) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(
      mode="body",
      pattern=tuple(f"robot/{rc.HAND_PREFIX}{b}" for b in _HAND_SENSOR_BODIES),
    ),
    secondary=secondary,
    fields=("force",),
    reduce="netforce",
    history_length=DECIMATION,
  )


def get_hand_table_contact_sensor() -> ContactSensorCfg:
  """Hand-vs-table contact sensor (table log-barrier / contact penalties)."""
  return _hand_table_sensor(
    "hand_table_contact",
    ContactMatch(mode="geom", pattern="table", entity="arena"),
  )


def _arm_sensor(
  name: str,
  secondary: ContactMatch | None,
  fields: tuple[str, ...],
  history_length: int,
) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(
      mode="body", pattern=tuple(f"robot/{b}" for b in rc.ARM_LINK_BODIES)
    ),
    secondary=secondary,
    fields=fields,
    reduce="netforce",
    history_length=history_length,
  )


def get_arm_world_contact_sensor() -> ContactSensorCfg:
  """Arm any-contact flags (arm_collision reward)."""
  return _arm_sensor("arm_world_contact", None, ("found",), 0)


def get_arm_table_contact_sensor() -> ContactSensorCfg:
  """Arm-vs-table impulse sensor."""
  return _arm_sensor(
    "arm_table_contact",
    ContactMatch(mode="geom", pattern="table", entity="arena"),
    ("force",),
    DECIMATION,
  )


def get_arm_object_contact_sensor() -> ContactSensorCfg:
  """Arm-vs-object impulse sensor."""
  return _arm_sensor(
    "arm_object_contact",
    ContactMatch(mode="subtree", pattern="object", entity="object"),
    ("force",),
    DECIMATION,
  )


def get_contact_clip_high() -> tuple[float, ...]:
  """Per-body impulse clip: 0.2 for the thumb links, 0.1 otherwise."""
  high = [0.1] * 16
  for i in rc.CONTACT_THUMB_INDICES:
    high[i] = 0.2
  return tuple(high)


def get_dexgrasp_rewards(object_names: tuple[str, ...]) -> dict[str, RewardTermCfg]:
  """§F reward stack (reference coeffs from cfg_reg.yaml)."""
  keypoints = SceneEntityCfg(
    "robot", body_names=rc.KEYPOINT_BODIES, preserve_order=True
  )
  arm_links = SceneEntityCfg(
    "robot", body_names=rc.ARM_LINK_BODIES, preserve_order=True
  )
  arm_joints = SceneEntityCfg(
    "robot", joint_names=rc.ARM_JOINT_NAMES, preserve_order=True
  )
  wrist = SceneEntityCfg("robot", body_names=("right_hand",))
  af_params = {
    "tip_indices": rc.KEYPOINT_TIP_INDICES,
    "thumb_tip_index": rc.KEYPOINT_THUMB_TIP_INDEX,
    "wrist_index": 0,
  }
  con_weights = contact_weights(
    rc.CONTACT_TIP_INDICES,
    rc.CONTACT_THUMB_INDICES,
    rc.CONTACT_THUMB_TIP_INDEX,
    0,
  ).tolist()
  clip_high = get_contact_clip_high()
  # Enclosure gate: thumb tip + >=2 non-thumb fingertips must be in contact.
  finger_tips = tuple(
    i for i in rc.CONTACT_TIP_INDICES if i != rc.CONTACT_THUMB_TIP_INDEX
  )
  gate_params = {
    "thumb_tip_index": rc.CONTACT_THUMB_TIP_INDEX,
    "finger_tip_indices": finger_tips,
    "min_fingers": 2,
  }
  return {
    "affordance_distance": RewardTermCfg(
      func=mdp.AffordanceDistance,
      weight=REWARD_COEFFS["affordance_distance"],
      params={
        "asset_cfg": keypoints,
        "object_entity": "object",
        "object_names": object_names,
        **af_params,
      },
    ),
    "table_logbarrier": RewardTermCfg(
      func=mdp.TableLogBarrier,
      weight=REWARD_COEFFS["table_logbarrier"],
      params={"asset_cfg": keypoints, "table_top_z": TABLE_TOP_Z, **af_params},
    ),
    "arm_height_logbarrier": RewardTermCfg(
      func=mdp.ArmHeightLogBarrier,
      weight=REWARD_COEFFS["arm_height_logbarrier"],
      params={"asset_cfg": arm_links, "table_top_z": TABLE_TOP_Z},
    ),
    "affordance_contact": RewardTermCfg(
      func=mdp.EnclosureGatedContact,
      weight=REWARD_COEFFS["affordance_contact"],
      params={
        "sensor_name": "hand_object_contact",
        "pad_parent_indices": rc.PAD_PARENT_INDICES,
        "mode": "flags",
        "divisor": 16.0,
        "weights": con_weights,
        **gate_params,
      },
    ),
    "affordance_impulse": RewardTermCfg(
      func=mdp.EnclosureGatedContact,
      weight=REWARD_COEFFS["affordance_impulse"],
      params={
        "sensor_name": "hand_object_contact",
        "pad_parent_indices": rc.PAD_PARENT_INDICES,
        "mode": "impulse_xy",
        "clip_high": clip_high,
        "weights": con_weights,
        **gate_params,
      },
    ),
    "table_contact": RewardTermCfg(
      func=mdp.ContactReward,
      weight=REWARD_COEFFS["table_contact"],
      params={
        "sensor_name": "hand_table_contact",
        "pad_parent_indices": rc.PAD_PARENT_INDICES,
        "mode": "flags",
        "divisor": 16.0,
        "weights": con_weights,
      },
    ),
    "table_impulse": RewardTermCfg(
      func=mdp.ContactReward,
      weight=REWARD_COEFFS["table_impulse"],
      params={
        "sensor_name": "hand_table_contact",
        "pad_parent_indices": rc.PAD_PARENT_INDICES,
        "mode": "impulse",
        "clip_high": clip_high,
        "weights": con_weights,
      },
    ),
    "arm_contact": RewardTermCfg(
      func=mdp.ContactReward,
      weight=REWARD_COEFFS["arm_contact"],
      params={
        "sensor_names": ("arm_table_contact", "arm_object_contact"),
        "mode": "flags",
      },
    ),
    "arm_impulse": RewardTermCfg(
      func=mdp.ContactReward,
      weight=REWARD_COEFFS["arm_impulse"],
      params={
        "sensor_names": ("arm_table_contact", "arm_object_contact"),
        "mode": "impulse",
      },
    ),
    "arm_collision": RewardTermCfg(
      func=mdp.ArmCollision,
      weight=REWARD_COEFFS["arm_collision"],
      params={"sensor_name": "arm_world_contact"},
    ),
    "object_velocity": RewardTermCfg(
      func=mdp.object_velocity,
      weight=REWARD_COEFFS["object_velocity"],
      params={"object_entity": "object"},
    ),
    "object_angular_velocity": RewardTermCfg(
      func=mdp.object_angular_velocity,
      weight=REWARD_COEFFS["object_angular_velocity"],
      params={"object_entity": "object"},
    ),
    "object_displacement": RewardTermCfg(
      func=mdp.ObjectDisplacement,
      weight=REWARD_COEFFS["object_displacement"],
      params={"object_entity": "object"},
    ),
    "wrist_velocity": RewardTermCfg(
      func=mdp.wrist_velocity,
      weight=REWARD_COEFFS["wrist_velocity"],
      params={"asset_cfg": wrist},
    ),
    "wrist_angular_velocity": RewardTermCfg(
      func=mdp.wrist_angular_velocity,
      weight=REWARD_COEFFS["wrist_angular_velocity"],
      params={"asset_cfg": wrist},
    ),
    "arm_joint_velocity": RewardTermCfg(
      func=mdp.arm_joint_velocity,
      weight=REWARD_COEFFS["arm_joint_velocity"],
      params={"asset_cfg": arm_joints},
    ),
  }


def dexgrasp_ur5e_rh5dg2_env_cfg(
  play: bool = False,
  object_name: str | None = None,
  object_names: tuple[str, ...] | None = None,
  non_uniform_sampling: bool = True,
) -> ManagerBasedRlEnvCfg:
  if object_name is not None and object_names is not None:
    raise ValueError("Specify either object_name or object_names, not both")
  if object_names is not None and not object_names:
    raise ValueError("object_names must contain at least one object")

  cfg = make_dexgrasp_env_cfg()

  cfg.scene.entities["robot"] = get_dexgrasp_robot_cfg()
  selected_object_names = object_names or (
    (object_name,) if object_name is not None else oc.ROBUST_DEXGRASP_TRAIN_OBJECTS
  )
  if play:
    if len(selected_object_names) > 1:
      selected_object_names = (SKELETON_OBJECT,)
    cfg.scene.entities["object"] = get_skeleton_object_cfg(
      next(iter(selected_object_names)), fixed=True
    )
  else:
    cfg.scene.entities["object"] = get_dexgrasp_object_cfg(selected_object_names)
  if object_name is None and object_names is None and not play:
    cfg.scene.num_envs = oc.ROBUST_DEXGRASP_BASELINE_NUM_ENVS
  cfg.scene.sensors = cfg.scene.sensors + (
    get_hand_object_contact_sensor(),
    get_hand_table_contact_sensor(),
    get_arm_world_contact_sensor(),
    get_arm_table_contact_sensor(),
    get_arm_object_contact_sensor(),
  )
  cfg.rewards = get_dexgrasp_rewards(selected_object_names)
  # Per-robot observation scopes. preserve_order locks the documented frame
  # order (KEYPOINT_BODIES / CONTACT_BODIES / ALL_JOINT_NAMES) against model
  # layout changes, so §F finger weights stay aligned with the obs columns.
  actor_terms = cfg.observations["actor"].terms
  joints = SceneEntityCfg("robot", joint_names=rc.ALL_JOINT_NAMES, preserve_order=True)
  keypoints = SceneEntityCfg(
    "robot", body_names=rc.KEYPOINT_BODIES, preserve_order=True
  )
  if not play:
    # Grasp-type diagnostics at episode end. model_700 (0% lift) vs model_1000
    # (64%) of run 8yelo0qc differed in thumb yaw 0.10 vs 0.82 and squeeze 52
    # vs 163 N while the contact reward was flat (documents/08 §6).
    grip_params = {
      "sensor_name": "hand_object_contact",
      "pad_parent_indices": rc.PAD_PARENT_INDICES,
    }
    cfg.metrics.update(
      {
        "thumb_yaw_last": MetricsTermCfg(
          func=mdp.joint_pos_mean,
          params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(rc.THUMB_YAW_JOINT,))
          },
          reduce="last",
        ),
        "grip_bodies_last": MetricsTermCfg(
          func=mdp.HandObjectGrip,
          params={**grip_params, "quantity": "bodies"},
          reduce="last",
        ),
        "grip_squeeze_xy_last": MetricsTermCfg(
          func=mdp.HandObjectGrip,
          params={**grip_params, "quantity": "squeeze_xy"},
          reduce="last",
        ),
        "grip_net_z_last": MetricsTermCfg(
          func=mdp.HandObjectGrip,
          params={**grip_params, "quantity": "net_z"},
          reduce="last",
        ),
        "object_linear_speed_max": MetricsTermCfg(
          func=mdp.object_linear_speed,
          params={"object_entity": "object"},
          reduce="max",
        ),
        "object_angular_speed_max": MetricsTermCfg(
          func=mdp.object_angular_speed,
          params={"object_entity": "object"},
          reduce="max",
        ),
        "hand_keypoint_below_table_depth_max": MetricsTermCfg(
          func=mdp.hand_keypoint_below_table_depth,
          params={"table_top_z": TABLE_TOP_Z, "asset_cfg": keypoints},
          reduce="max",
        ),
        "arm_action_magnitude_mean": MetricsTermCfg(
          func=mdp.mean_arm_action_magnitude,
          params={"arm_action_dim": len(rc.ARM_JOINT_NAMES)},
        ),
        "object_lift_height_max": MetricsTermCfg(
          func=mdp.ObjectLiftHeight,
          params={"object_entity": "object"},
          reduce="max",
        ),
        "lift_success": MetricsTermCfg(
          func=mdp.LiftSuccess,
          params={"object_entity": "object", "success_height": 0.10},
          reduce="max",
        ),
      }
    )
  cfg.terminations["hand_below_table"] = TerminationTermCfg(
    func=mdp.hand_below_table,
    params={
      "table_top_z": TABLE_TOP_Z,
      "asset_cfg": keypoints,
      "tolerance": HAND_TABLE_TERMINATION_TOLERANCE,
    },
  )
  # Truncation, not termination: the reference has no workspace bound at all and
  # always runs the full 70 steps, so cutting the return here would add a
  # negative bias it never had -- and this fires on ~35% of early episodes.
  # Bootstrapping instead lets the critic learn the state is low-value on its
  # own, while the reset still guards against unrecoverable off-table spin.
  cfg.terminations["object_out_of_workspace"] = TerminationTermCfg(
    func=mdp.object_out_of_workspace,
    params={"bounds": OBJECT_WORKSPACE_BOUNDS, "object_entity": "object"},
    time_out=True,
  )
  cfg.terminations["nan"] = TerminationTermCfg(func=mdp.nan_detection)
  arm_links = SceneEntityCfg(
    "robot", body_names=rc.ARM_LINK_BODIES, preserve_order=True
  )
  wrist = SceneEntityCfg("robot", body_names=("right_hand",))
  hand_center = SceneEntityCfg("robot", site_names=(rc.GRASP_CENTER_SITE,))
  actor_terms["joint_pos"].params["asset_cfg"] = joints
  actor_terms["pd_error"].params["asset_cfg"] = joints
  actor_terms["contacts"].params["pad_parent_indices"] = rc.PAD_PARENT_INDICES
  actor_terms["keypoint_heights"].params["asset_cfg"] = keypoints
  actor_terms["arm_link_heights"].params["asset_cfg"] = arm_links
  actor_terms["hand_center"].params["asset_cfg"] = hand_center
  actor_terms["wrist_orientation"].params["asset_cfg"] = wrist
  actor_terms["af_vec"].params["asset_cfg"] = keypoints
  actor_terms["af_vec"].params["object_names"] = selected_object_names

  action = cfg.actions["joint_pos"]
  assert isinstance(action, RelativeJointPositionActionCfg)
  action.scale = ACTION_SCALE

  # Replace the skeleton's default-pose resets with the sampled object pose +
  # analytic-IK pre-grasp (§C). Arena/base placement stays as is.
  del cfg.events["reset_robot_joints"]
  del cfg.events["reset_object"]
  cfg.events["reset_grasp_pose"] = EventTermCfg(
    func=mdp.ResetGraspPose,
    mode="reset",
    params={
      "object_names": selected_object_names,
      "table_top_z": TABLE_TOP_Z,
      "mount_z": ARM_MOUNT_Z,
      "object_clearance": 0.0 if play else 0.002,
      "non_uniform_sampling": non_uniform_sampling and not play,
    },
  )

  cfg.viewer.body_name = "base"

  if play:
    cfg.terminations = {}
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
