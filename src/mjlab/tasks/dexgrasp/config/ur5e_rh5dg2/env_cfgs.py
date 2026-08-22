"""UR5e + RH5-DG2 wiring for the DexGrasp env."""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.dexgrasp import mdp
from mjlab.tasks.dexgrasp.dexgrasp_env_cfg import (
  ARM_MOUNT_Z,
  TABLE_TOP_Z,
  make_dexgrasp_env_cfg,
)

# Actuator-name regexes -> delta-action scale (arm 0.005, finger 0.015 rad).
# Like RobustDexGrasp, scale is a residual gain, not a per-step cap: the delta is
# unbounded and only the absolute target is limited (arm actuator ctrlrange;
# fingers by joint limits). §D adds explicit soft-limit target clipping.
ACTION_SCALE = {
  r"(shoulder|elbow|wrist).*": rc.ACTION_SCALE_ARM,
  r"R_.*": rc.ACTION_SCALE_FINGER,
}

# §B skeleton: one fixed object on the table (multi-object VariantEntityCfg in §C+).
SKELETON_OBJECT = "potted_meat_can"
OBJECT_XY = (0.0, -0.6)  # Polar r~0.6, theta=-0.5pi (in the sampling region).


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


def get_skeleton_object_cfg(name: str) -> EntityCfg:
  obj = oc.PHASE1_OBJECTS[name]
  init_state = EntityCfg.InitialStateCfg(
    pos=(OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z - obj.lowest_point),
  )
  return EntityCfg(init_state=init_state, spec_fn=obj.spec_fn)


def dexgrasp_ur5e_rh5dg2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_dexgrasp_env_cfg()

  cfg.scene.entities["robot"] = get_dexgrasp_robot_cfg()
  cfg.scene.entities["object"] = get_skeleton_object_cfg(SKELETON_OBJECT)

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
      "object_name": SKELETON_OBJECT,
      "table_top_z": TABLE_TOP_Z,
      "mount_z": ARM_MOUNT_Z,
    },
  )

  cfg.viewer.body_name = "base"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
