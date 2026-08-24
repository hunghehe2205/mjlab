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
from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
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

  # Drop all keyframes; the init state is set by InitialStateCfg.
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

# Adopt the menagerie arm <position> actuators so mjlab actions can target the
# arm joints (keeps their tuned gains).
ARM_ACTUATORS = (XmlActuatorCfg(target_names_expr=(r"(shoulder|elbow|wrist).*",)),)

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
# Joint groups, action scale, pre-grasp pose.
##

ARM_JOINT_NAMES = (
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint",
)

FINGER_JOINT_NAMES = (
  "R_thumb_yaw_joint",
  "R_thumb_mcp_joint",
  "R_thumb_pip_joint",
  "R_thumb_dip_joint",
  "R_index_yaw_joint",
  "R_index_mcp_joint",
  "R_index_pip_joint",
  "R_index_dip_joint",
  "R_middle_yaw_joint",
  "R_middle_mcp_joint",
  "R_middle_pip_joint",
  "R_middle_dip_joint",
  "R_ring_mcp_joint",
  "R_ring_pip_joint",
  "R_ring_dip_joint",
  "R_pinky_mcp_joint",
  "R_pinky_pip_joint",
  "R_pinky_dip_joint",
)

# Palm-local grasp reference site (hand_center analog).
GRASP_CENTER_SITE = "grasp_center"

# Hand body frames for the teacher observation: wrist, then per finger the
# joint bodies followed by the fingertip pad. 24 total.
KEYPOINT_BODIES = (
  "right_hand",
  "R_thumb_yaw",
  "R_thumb_mcp",
  "R_thumb_pip",
  "R_thumb_dip",
  "R_thumb_force_sensor",
  "R_index_yaw",
  "R_index_mcp",
  "R_index_pip",
  "R_index_dip",
  "R_index_force_sensor",
  "R_middle_yaw",
  "R_middle_mcp",
  "R_middle_pip",
  "R_middle_dip",
  "R_middle_force_sensor",
  "R_ring_mcp",
  "R_ring_pip",
  "R_ring_dip",
  "R_ring_force_sensor",
  "R_pinky_mcp",
  "R_pinky_pip",
  "R_pinky_dip",
  "R_pinky_force_sensor",
)

# Contact bodies for the contact/impulse observation: palm plus the three
# distal links of each finger. The pad bodies are fixed children of the dip
# links (and of the palm), so body-subtree contact sensors see pad contacts.
CONTACT_BODIES = (
  "R_hand_palm",
  "R_thumb_mcp",
  "R_thumb_pip",
  "R_thumb_dip",
  "R_index_mcp",
  "R_index_pip",
  "R_index_dip",
  "R_middle_mcp",
  "R_middle_pip",
  "R_middle_dip",
  "R_ring_mcp",
  "R_ring_pip",
  "R_ring_dip",
  "R_pinky_mcp",
  "R_pinky_pip",
  "R_pinky_dip",
)

# Arm bodies whose frames give the 6 arm-link heights above the table.
ARM_LINK_BODIES = (
  "shoulder_link",
  "upper_arm_link",
  "forearm_link",
  "wrist_1_link",
  "wrist_2_link",
  "wrist_3_link",
)

# All actuated joints in action/observation order (arm then fingers).
ALL_JOINT_NAMES = ARM_JOINT_NAMES + FINGER_JOINT_NAMES

# Finger-layout indices for the §F reward weights, aligned with
# KEYPOINT_BODIES / CONTACT_BODIES order.
KEYPOINT_TIP_INDICES = (5, 10, 15, 19, 23)  # thumb, index, middle, ring, pinky pads
KEYPOINT_THUMB_TIP_INDEX = 5
CONTACT_TIP_INDICES = (3, 6, 9, 12, 15)  # per-finger dip link
CONTACT_THUMB_INDICES = (1, 2, 3)
CONTACT_THUMB_TIP_INDEX = 3

# Per-step delta-action scale (RobustDexGrasp: arm 0.005, finger 0.015 rad).
ACTION_SCALE_ARM = 0.005
ACTION_SCALE_FINGER = 0.015

# Pre-grasp finger pose (cupped, thumb opposed); first estimate, refine in viewer.
INIT_FINGER_POSE = {
  "R_thumb_yaw_joint": 1.2,
  "R_thumb_mcp_joint": 0.3,
  "R_thumb_pip_joint": 0.3,
  "R_thumb_dip_joint": 0.2,
  "R_index_yaw_joint": 0.0,
  "R_index_mcp_joint": 0.3,
  "R_index_pip_joint": 0.3,
  "R_index_dip_joint": 0.3,
  "R_middle_yaw_joint": 0.0,
  "R_middle_mcp_joint": 0.3,
  "R_middle_pip_joint": 0.3,
  "R_middle_dip_joint": 0.3,
  "R_ring_mcp_joint": 0.3,
  "R_ring_pip_joint": 0.3,
  "R_ring_dip_joint": 0.3,
  "R_pinky_mcp_joint": 0.3,
  "R_pinky_pip_joint": 0.3,
  "R_pinky_dip_joint": 0.3,
}

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS + HAND_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_ur5e_rh5dg2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


##
# Standalone viewing.
##


def add_scene(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Add a skybox, checker ground plane, and directional light in place."""
  spec.add_texture(
    name="skybox",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.0, 0.0, 0.0),
    width=512,
    height=3072,
  )
  spec.add_texture(
    name="groundplane",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.2, 0.3, 0.4),
    rgb2=(0.1, 0.15, 0.2),
    mark=mujoco.mjtMark.mjMARK_EDGE,
    markrgb=(0.8, 0.8, 0.8),
    width=300,
    height=300,
  )
  mat = spec.add_material(
    name="groundplane", texrepeat=(5, 5), texuniform=True, reflectance=0.2
  )
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
  spec.worldbody.add_light(
    pos=(0, 0, 3.0), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  )
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.05),
    material="groundplane",
  )
  return spec


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ur5e_rh5dg2_robot_cfg())
  spec = add_scene(robot.spec)
  model = spec.compile()
  # Match the integrator mjlab sims use; the compiled default is Euler.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  viewer.launch(model)
