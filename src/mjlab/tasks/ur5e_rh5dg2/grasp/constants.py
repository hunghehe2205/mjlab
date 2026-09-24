"""Fixed primitive and calibrated joint waypoints for the teacher baseline."""

from mjlab.asset_zoo.scenes.workstation import TABLE_TOP_Z

BOX_SIZE = (0.03, 0.03, 0.06)
OBJECT_POS = (-0.10, 0.48, TABLE_TOP_Z + BOX_SIZE[2])
LIFT_HEIGHT = 0.10
HOLD_TIME = 3.0
FORCE_THRESHOLD = 0.1
ARM_ACTION_SCALE = 0.10
HAND_ACTION_SCALE = 0.50
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
TIP_BODIES = tuple(f"R_{finger}_force_sensor" for finger in FINGERS)
ARM_BODIES = (
  "shoulder_link",
  "upper_arm_link",
  "forearm_link",
  "wrist_1_link",
  "wrist_2_link",
  "wrist_3_link",
)
PREGRASP_ARM = (
  -1.5379673853,
  -1.5623198752,
  1.7366317066,
  -1.7451081582,
  -1.5707963268,
  0.0328289415,
)
GRASP_ARM = (
  -1.5379673853,
  -1.4268370118,
  2.0031156539,
  -2.1470749689,
  -1.5707963268,
  0.0328289415,
)
LIFT_ARM = (
  -1.5379673853,
  -1.5758645552,
  1.5452335959,
  -1.5401653675,
  -1.5707963268,
  0.0328289415,
)
OPEN_HAND = (
  1.2,
  0.4,
  0.4,
  0.4,
  0.0,
  0.4,
  0.4,
  0.4,
  0.0,
  0.4,
  0.4,
  0.4,
  0.4,
  0.4,
  0.4,
  0.4,
  0.4,
  0.4,
)
CLOSED_HAND = (
  1.2,
  0.4,
  0.7,
  0.7,
  0.0,
  0.9,
  0.9,
  0.9,
  0.0,
  0.9,
  0.9,
  0.9,
  0.9,
  0.9,
  0.9,
  0.9,
  0.9,
  0.9,
)
