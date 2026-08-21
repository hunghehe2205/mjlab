"""UR5e arm with RH5-DG2 right hand.

The UR5e arm comes from mujoco_menagerie and keeps its own well-tuned ``<position>``
actuators. The RH5-DG2 right hand is attached at the arm's ``attachment_site``; its 18
finger joints are driven by position actuators added here.

Finger gains are uniform rather than inertia-matched: the link inertias span ~150x
(2.5e-6..4e-4), so per-joint natural-frequency gains would leave the distal joints
floppy. This makes the small distal joints stiff, which is stable under mjlab's default
``implicitfast`` integrator (Euler jitters).
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

_XMLS: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "ur5e_rh5dg2" / "xmls"
UR5E_XML: Path = _XMLS / "ur5e.xml"
HAND_XML: Path = _XMLS / "right_hand.xml"
assert UR5E_XML.exists()
assert HAND_XML.exists()

# Attach prefix for hand elements; avoids name clashes with the arm.
HAND_PREFIX = "rh/"


def get_spec() -> mujoco.MjSpec:
  """Attach the right hand to the UR5e flange and return the merged spec."""
  arm = mujoco.MjSpec.from_file(str(UR5E_XML))
  hand = mujoco.MjSpec.from_file(str(HAND_XML))

  # Mount the hand root at the arm's attachment_site (identity offset).
  site = arm.site("attachment_site")
  frame = arm.body("wrist_3_link").add_frame(pos=list(site.pos), quat=list(site.quat))
  arm.attach(child=hand, prefix=HAND_PREFIX, frame=frame)

  # Drop the arm-only keyframe; the init state is set by InitialStateCfg.
  for key in list(arm.keys):
    arm.delete(key)
  return arm


##
# Actuator config.
##

# Only the fingers are configured here; the arm keeps its menagerie XML actuators.
FINGER_STIFFNESS = 1.0
FINGER_DAMPING = 0.1
FINGER_EFFORT_LIMIT = 1.0  # matches the hand XML actuatorfrcrange (+/-1 Nm).
FINGER_ARMATURE = 1e-4  # stabilizes the stiff PD on the tiny finger inertias.

# Match bare joint names: mjlab strips the attach prefix from entity-local names.
HAND_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("R_.*_joint",),
    stiffness=FINGER_STIFFNESS,
    damping=FINGER_DAMPING,
    effort_limit=FINGER_EFFORT_LIMIT,
    armature=FINGER_ARMATURE,
  ),
)

##
# Keyframe config.
##

# UR5e "home" pose from menagerie; fingers default to 0 (open hand).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  joint_pos={
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=HAND_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_ur5e_rh5dg2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ur5e_rh5dg2_robot_cfg())
  model = robot.spec.compile()
  # Match the integrator mjlab sims use; the compiled default is Euler.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  viewer.launch(model)
