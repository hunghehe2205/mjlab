"""UR5e constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

UR5E_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "universal_robots_ur5e"
  / "xmls"
  / "ur5e.xml"
)
assert UR5E_XML.exists()

ATTACHMENT_SITE = "attachment_site"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(UR5E_XML))


SIZE3_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint")
SIZE1_JOINTS = ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# Gains taken from the menagerie position actuators (size3 / size1 groups).
ARM_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=SIZE3_JOINTS,
    stiffness=2000.0,
    damping=400.0,
    effort_limit=150.0,
    armature=0.1,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=SIZE1_JOINTS,
    stiffness=500.0,
    damping=100.0,
    effort_limit=28.0,
    armature=0.1,
  ),
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_ur5e_arm_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_spec, articulation=ARTICULATION)


UR5E_ACTION_SCALE: dict[str, float] = {}
for _a in ARM_ACTUATORS:
  assert _a.effort_limit is not None
  for _n in _a.target_names_expr:
    UR5E_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  viewer.launch(get_spec().compile())
