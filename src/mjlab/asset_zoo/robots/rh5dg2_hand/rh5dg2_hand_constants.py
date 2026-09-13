"""rh5dg2 right-hand constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

RH5DG2_HAND_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "rh5dg2_hand" / "xmls" / "right_hand.xml"
)
assert RH5DG2_HAND_XML.exists()

HAND_ROOT_BODY = "right_hand"
HAND_JOINT_EXPR = "R_.*_joint"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(RH5DG2_HAND_XML))


# Allegro-class position gains; the source caps joint torque at +/-1 Nm.
HAND_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(HAND_JOINT_EXPR,),
  stiffness=1.0,
  damping=0.05,
  effort_limit=1.0,
  armature=1e-3,
)

HAND_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=0,
  friction=(1.0,),
  disable_other_geoms=False,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HAND_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_rh5dg2_hand_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=ARTICULATION,
    collisions=(HAND_COLLISION,),
  )


RH5DG2_HAND_ACTION_SCALE: dict[str, float] = {HAND_JOINT_EXPR: 0.25 * 1.0 / 1.0}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  viewer.launch(get_spec().compile())
