"""Tests for ur5e_rh5dg2_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as c
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr

ARM_JOINTS = (
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint",
)


@pytest.fixture(scope="module")
def entity() -> Entity:
  return Entity(c.get_ur5e_rh5dg2_robot_cfg())


@pytest.fixture(scope="module")
def model(entity: Entity) -> mujoco.MjModel:
  return entity.spec.compile()


def test_entity_creation(entity: Entity) -> None:
  # 6 arm joints + 18 right-hand joints, all actuated.
  assert entity.num_joints == 24
  assert entity.num_actuators == 24
  assert entity.is_actuated
  assert entity.is_fixed_base


def test_hand_attached(entity: Entity) -> None:
  hand_joints = [n for n in entity.joint_names if n.startswith("R_")]
  assert len(hand_joints) == 18


def test_arm_keeps_menagerie_actuators(model: mujoco.MjModel) -> None:
  names = {model.actuator(i).name for i in range(model.nu)}
  assert {
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
  } <= (names)


def test_hand_actuator_parameters(model: mujoco.MjModel) -> None:
  finger = re.compile(r"rh/R_.*_joint$")
  matched = 0
  for i in range(model.nu):
    act = model.actuator(i)
    if not finger.match(act.name):
      continue
    matched += 1
    assert act.gainprm[0] == c.FINGER_STIFFNESS
    assert act.biasprm[1] == -c.FINGER_STIFFNESS
    assert act.biasprm[2] == -c.FINGER_DAMPING
    assert act.forcerange[0] == -c.FINGER_EFFORT_LIMIT
    assert act.forcerange[1] == c.FINGER_EFFORT_LIMIT
  assert matched == 18


def test_keyframe_home_pose(entity: Entity, model: mujoco.MjModel) -> None:
  joint_pos = c.HOME_KEYFRAME.joint_pos
  assert joint_pos is not None
  key = model.key("init_state")
  expected = dict(
    zip(
      entity.joint_names,
      resolve_expr(joint_pos, entity.joint_names, 0.0),
      strict=True,
    )
  )
  # Model joint names carry the "rh/" attach prefix; entity names are stripped.
  for i in range(model.njnt):
    name = model.joint(i).name
    stripped = name.split("/")[-1]
    np.testing.assert_allclose(
      key.qpos[model.joint(i).qposadr[0]], expected[stripped], rtol=1e-5
    )
    if stripped.startswith("R_"):
      assert expected[stripped] == 0.0  # fingers open at home.


def test_home_pose_is_collision_free(model: mujoco.MjModel) -> None:
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  assert data.ncon == 0
