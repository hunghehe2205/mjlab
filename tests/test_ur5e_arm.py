import mujoco

from mjlab.asset_zoo.robots.universal_robots_ur5e import ur5e_constants as ur5e
from mjlab.entity.entity import Entity


def test_ur5e_spec_compiles_with_six_joints():
  model = ur5e.get_spec().compile()
  assert model.nq == 6
  assert model.njnt == 6
  site_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)
  }
  assert ur5e.ATTACHMENT_SITE in site_names


def test_ur5e_entity_has_six_position_actuators():
  model = Entity(ur5e.get_ur5e_arm_cfg()).spec.compile()
  assert model.nu == 6
  assert set(ur5e.UR5E_ACTION_SCALE) == {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
  }
