"""Tests for reset-time UR5e-to-RH5-DG2 collision rejection."""

import numpy as np

from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.tasks.dexgrasp.pregrasp.self_collision import ArmHandSelfCollisionProbe


def test_arm_hand_self_collision_probe() -> None:
  probe = ArmHandSelfCollisionProbe()
  home = np.array(
    [(rc.HOME_KEYFRAME.joint_pos or {})[name] for name in rc.ARM_JOINT_NAMES]
  )
  colliding = np.array([0.8601, -1.4457, -2.8827, -3.0362, 1.9673, 2.5921])

  assert not probe.collides(home)
  assert probe.collides(colliding)


def test_pregrasp_probe_rejects_folded_elbow_down_branch() -> None:
  probe = ArmHandSelfCollisionProbe()
  expected = np.array([-1.2321, -1.5034, -1.7856, -2.8954, -1.8436, -1.3316])
  folded = np.array([-1.4323, 2.8812, 1.9055, -1.6645, 2.0602, 1.6140])

  assert probe.is_valid_pregrasp(expected)
  assert not probe.is_valid_pregrasp(folded)
