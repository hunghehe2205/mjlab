"""Tests for scripted DexGrasp lift evaluation."""

from types import SimpleNamespace
from typing import cast

import numpy as np

from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics
from mjlab.tasks.dexgrasp.scripts.evaluate import lift_targets


def test_lift_targets_uses_angle_branch_nearest_to_current_pose() -> None:
  pose = np.eye(4)
  kin = SimpleNamespace(
    fk_grasp_center_env=lambda _: pose,
    arm_qpos_for_grasp_center=lambda *_args, **_kwargs: np.array(
      [2.0 * np.pi - 0.2, 0.1, -0.1, 0.2, -0.2, 0.3]
    ),
  )
  arm_qpos = np.zeros((1, 6))

  targets, reachable = lift_targets(
    arm_qpos, cast(ArmKinematics, kin), lift_height=0.15
  )

  assert reachable.tolist() == [True]
  np.testing.assert_allclose(targets[0], [-0.2, 0.1, -0.1, 0.2, -0.2, 0.3])
