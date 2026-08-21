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
  actuators=HAND_ACTUATORS,
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
