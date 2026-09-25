"""Primitive, pre-grasp sampling and hand presets for the grasp-only teacher."""

import math

from mjlab.asset_zoo.scenes.workstation import TABLE_TOP_Z

BOX_SIZE = (0.03, 0.03, 0.06)
OBJECT_POS = (-0.10, 0.48, TABLE_TOP_Z + BOX_SIZE[2])
# Grasp-only episodes; the lift test (RobustDexGrasp evaluation) runs after them,
# ramping the arm while the policy keeps the fingers.
GRASP_TIME = 4.0
LIFT_TIME = 4.0
LIFT_RAMP = 3.0
LIFT_OFFSET = 0.20
LIFT_HEIGHT = 0.10
HOLD_TIME = 3.0
FORCE_THRESHOLD = 0.1
GRIP_FORCE = 5.0
# Object-table load above the box weight (0.78 N) is pressing; cost per 10 N.
PUSH_FORCE = 1.0
PUSH_SCALE = 10.0
LIFT_TEST_ENVS = 256
# Targets accumulate per 0.05 s step (reference: 0.005/0.015 rad per 0.2 s step,
# raised so the 6 cm approach fits a 4 s episode). The target may lead the joint by
# at most the offsets, which bound the grasp preload.
ARM_ACTION_SCALE = 0.01
HAND_ACTION_SCALE = 0.03
ARM_MAX_OFFSET = 0.10
HAND_MAX_OFFSET = 0.50
MIN_ACTION_STD = 0.2
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ROOTS = (
  "R_thumb_yaw",
  "R_index_yaw",
  "R_middle_yaw",
  "R_ring_mcp",
  "R_pinky_mcp",
)
HAND_BODIES = ("R_hand_palm",) + tuple(
  f"R_{finger}_{segment}"
  for finger in FINGERS
  for segment in (
    ("yaw", "mcp", "pip", "dip") if finger in FINGERS[:3] else ("mcp", "pip", "dip")
  )
)
ARM_BODIES = (
  "shoulder_link",
  "upper_arm_link",
  "forearm_link",
  "wrist_1_link",
  "wrist_2_link",
  "wrist_3_link",
)
# Pre-grasp sampling after RobustDexGrasp (cfg_reg.yaml, train.py), mirrored so
# the workspace lies in +Y of the arm base. Hand-local +X is the palm normal and
# +Z the finger axis of the right_hand (wrist) frame.
OBJECT_ANGLE = (0.3 * math.pi, 0.7 * math.pi)
OBJECT_DISTANCE = (0.45, 0.75)
OBJECT_MAX_ABS_X = 0.25
CAMERA_POS = (-0.035, 0.58, 1.531)
TOP_GRASP = True
# The paper starts 0.25 m away. Resets start at pre-close (box between thumb and
# fingers) raised by up to 3 cm, so the policy learns to close rather than to
# press on the box top; the scripted probe approaches from 6 cm.
PREGRASP_STANDOFF = (0.0, 0.03)
STANDOFF = 0.06
NUM_ROLLS = 10
GRASP_WIDTH_LIMIT = 0.18
LENGTH_SCORE_COEFF = 5.0
ANGLE_SCORE_COEFF = 1.0
PREGRASP_CLEARANCE = 0.005
POOL_SIZE = 1024
# Object center inside the open hand at pre-close, in the right_hand frame.
HAND_CENTER = (0.125, 0.005, 0.190)
# Collision-free palm-down IK branch, used to seed every pre-grasp solve.
ARM_IK_SEED = (
  -1.7897409333,
  -1.9260224786,
  2.5735678028,
  -0.6475453242,
  1.3518517203,
  -1.5707963268,
)
# Nearly extended fingers and a thumb swung 47 deg from the palm plane, inside the
# 0.9 soft joint limits that the action clamps to.
OPEN_HAND = (
  0.8,
  0.08,
  0.06,
  0.06,
  0.0,
  0.1,
  0.1,
  0.1,
  0.0,
  0.1,
  0.1,
  0.1,
  0.1,
  0.1,
  0.1,
  0.1,
  0.1,
  0.1,
)
CLOSED_HAND = (
  1.2,
  0.4,
  0.5,
  0.5,
  0.0,
  0.5,
  0.5,
  0.5,
  0.0,
  0.5,
  0.5,
  0.5,
  0.5,
  0.5,
  0.5,
  0.5,
  0.5,
  0.5,
)
