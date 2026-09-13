import mujoco

from mjlab.asset_zoo.robots.rh5dg2_hand import rh5dg2_hand_constants as hand
from mjlab.entity.entity import Entity

HAND_JOINTS = {
  f"R_{finger}_{seg}_joint"
  for finger in ("thumb", "index", "middle")
  for seg in ("yaw", "mcp", "pip", "dip")
} | {
  f"R_{finger}_{seg}_joint"
  for finger in ("ring", "pinky")
  for seg in ("mcp", "pip", "dip")
}


def test_hand_spec_compiles_with_18_joints():
  model = hand.get_spec().compile()
  assert model.nq == 18
  assert model.njnt == 18
  joints = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  }
  assert joints == HAND_JOINTS


def test_hand_entity_has_18_position_actuators():
  model = Entity(hand.get_rh5dg2_hand_cfg()).spec.compile()
  assert model.nu == 18
