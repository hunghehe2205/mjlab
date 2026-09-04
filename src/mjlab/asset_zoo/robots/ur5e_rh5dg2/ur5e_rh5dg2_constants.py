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

ROBOT_FRICTION = 0.8

# Arm servo bandwidth. The menagerie gains give kd/kp = 0.2 s -- exactly the 5 Hz
# control period -- so a saturated action only realizes 1-exp(-1) = 63% of its
# delta. Retuning to a uniform natural frequency with zeta ~ 1 reproduces the
# reference's system-identified UR5 controller (kp 15775-16202, kd 281-577,
# i.e. wn 63.7, zeta 1.15) and is stable at any timestep since kp*dt^2/M
# reduces to (wn*dt)^2, independent of link inertia.
ARM_OMEGA_N = 64.0
ARM_ZETA = 1.15


def _retune_arm_gains(spec: mujoco.MjSpec) -> None:
  """Set arm position gains to (ARM_OMEGA_N, ARM_ZETA) using per-dof inertia."""
  model = spec.compile()
  for act in spec.actuators:
    joint = model.joint(act.target)
    inertia = model.dof_M0[joint.dofadr[0]]
    kp = inertia * ARM_OMEGA_N**2
    kd = 2.0 * ARM_ZETA * inertia * ARM_OMEGA_N
    act.gainprm[0] = kp
    act.biasprm[1] = -kp
    act.biasprm[2] = -kd


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

  _retune_arm_gains(arm)

  # Reference default material friction (RaiSim setDefaultMaterial(0.8)). Makes
  # hand-table and hand-object contacts 0.8 once the table drops its priority.
  for geom in arm.geoms:
    geom.friction[0] = ROBOT_FRICTION

  # Gravity compensation on every robot body. Models the UR payload-aware
  # controller; without it the relative-position action ratchets the arm down
  # (~0.3 m/episode) because each step re-anchors on the sagged qpos.
  for body in arm.bodies:
    if body.name:
      body.gravcomp = 1.0

  # Drop all keyframes; the init state is set by InitialStateCfg.
  for key in list(arm.keys):
    arm.delete(key)
  return arm


##
# Actuator config.
##

# Only the fingers are configured here; the arm keeps its menagerie XML actuators.
# Grip torque saturates at kp * ACTION_SCALE_FINGER, so kp=1 capped it at
# 0.015 Nm -- 47x below the reference Allegro's 0.7 Nm. Matching the reference's
# kp=600 needs armature to carry it: reflected rotor inertia dominates a geared
# finger, and without it kp*dt^2/M blows up (measured unstable above kp=67 at
# armature=1e-4). kd is set for zeta ~ 1 at that armature.
FINGER_STIFFNESS = 600.0
FINGER_ARMATURE = 3e-2
FINGER_DAMPING = 2.0 * (FINGER_STIFFNESS * FINGER_ARMATURE) ** 0.5
FINGER_EFFORT_LIMIT = 1.0  # matches the hand XML actuatorfrcrange (+/-1 Nm).

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

# Thumb opposition joint: ~0.1 lies alongside the index (no opposition), ~0.8
# opposes it. Logged at episode end as a grasp-type diagnostic.
THUMB_YAW_JOINT = "R_thumb_yaw_joint"

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
# links (and of the palm); body-mode sensors don't see child-body geoms, so
# the pads get their own sensor slots (PAD_BODIES) folded into these 16.
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

# Welded pad bodies holding the pad collision meshes, and the CONTACT_BODIES
# slot each folds into (the pad's parent: per-finger dip link, then palm).
PAD_BODIES = (
  "R_thumb_force_sensor",
  "R_index_force_sensor",
  "R_middle_force_sensor",
  "R_ring_force_sensor",
  "R_pinky_force_sensor",
  "R_palm_force_sensor",
)
PAD_PARENT_INDICES = (3, 6, 9, 12, 15, 0)

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
# Thumb matches the RaiSim RH5-DG2 variant. At yaw 1.2 the thumb tip sat 4 cm
# ahead of the other tips on the approach ray and touched first in every
# episode; model_50 took 80% of its contact reward from the thumb alone.
INIT_FINGER_POSE = {
  "R_thumb_yaw_joint": 0.3,
  "R_thumb_mcp_joint": 0.2,
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

# 1.0: the action target clamps to the full URDF range like the reference; 0.9
# silently cut ~10% of finger travel at each end.
ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS + HAND_ACTUATORS,
  soft_joint_pos_limit_factor=1.0,
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
