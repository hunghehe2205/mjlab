import mujoco
import pytest

from mjlab.asset_zoo.robots import (
  get_rh5dg2_hand_cfg,
  get_ur5e_arm_cfg,
  get_ur5e_rh5dg2_robot_cfg,
)
from mjlab.entity import Entity


@pytest.mark.parametrize(
  "robot_name,robot_cfg_fn",
  [
    ("UR5E", get_ur5e_arm_cfg),
    ("RH5DG2_HAND", get_rh5dg2_hand_cfg),
    ("UR5E_RH5DG2", get_ur5e_rh5dg2_robot_cfg),
  ],
)
def test_robot_compiles_parametrized(robot_name: str, robot_cfg_fn) -> None:
  """Tests that all robots in the asset zoo compile without errors."""
  robot_cfg = robot_cfg_fn()
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)
