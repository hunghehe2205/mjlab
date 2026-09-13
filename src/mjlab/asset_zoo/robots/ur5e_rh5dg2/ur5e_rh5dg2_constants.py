"""Combined UR5e + rh5dg2 right-hand robot."""

import math

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.rh5dg2_hand import rh5dg2_hand_constants as hand
from mjlab.asset_zoo.robots.universal_robots_ur5e import ur5e_constants as ur5e
from mjlab.asset_zoo.scenes.workstation import ARM_MOUNT_Z
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

HAND_PREFIX = ""

# Tool-attach offset (hand root relative to the wrist site); tuned in viewer.
HAND_MOUNT_POS = (0.0, 0.0, 0.0)
HAND_MOUNT_QUAT = (1.0, 0.0, 0.0, 0.0)

# Base orientation so the arm faces the table (+y); tuned in viewer.
BASE_ROT = (1.0, 0.0, 0.0, 0.0)

ARM_HOME_RAD = {
  "shoulder_pan_joint": -math.pi / 2,
  "shoulder_lift_joint": -math.pi / 2,
  "elbow_joint": math.pi / 2,
  "wrist_1_joint": -math.pi / 2,
  "wrist_2_joint": -math.pi / 2,
  "wrist_3_joint": 0.0,
}


def get_spec() -> mujoco.MjSpec:
  arm = ur5e.get_spec()
  hand_spec = hand.get_spec()
  hand_body = arm.site(ur5e.ATTACHMENT_SITE).attach_body(
    hand_spec.body(hand.HAND_ROOT_BODY), HAND_PREFIX, ""
  )
  hand_body.pos = np.array(HAND_MOUNT_POS)
  hand_body.quat = np.array(HAND_MOUNT_QUAT)
  return arm


ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*ur5e.ARM_ACTUATORS, hand.HAND_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, ARM_MOUNT_Z),
  rot=BASE_ROT,
  joint_pos={**ARM_HOME_RAD, hand.HAND_JOINT_EXPR: 0.0},
  joint_vel={".*": 0.0},
)


def get_ur5e_rh5dg2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
    collisions=(hand.HAND_COLLISION,),
  )


UR5E_RH5DG2_ACTION_SCALE = {
  **ur5e.UR5E_ACTION_SCALE,
  **hand.RH5DG2_HAND_ACTION_SCALE,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.asset_zoo.scenes.workstation import get_workstation_spec

  robot = get_spec()
  scene = get_workstation_spec()
  frame = scene.worldbody.add_frame(pos=(0.0, 0.0, ARM_MOUNT_Z), quat=BASE_ROT)
  frame.attach_body(robot.body("base"), "robot_", "")
  viewer.launch(scene.compile())
