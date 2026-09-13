from mjlab.asset_zoo.robots import get_ur5e_rh5dg2_robot_cfg
from mjlab.entity.entity import Entity


def test_combined_robot_builds_with_24_joints_and_actuators():
  model = Entity(get_ur5e_rh5dg2_robot_cfg()).spec.compile()
  assert model.nq == 24  # 6 arm + 18 hand
  assert model.njnt == 24
  assert model.nu == 24
