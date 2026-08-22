"""ArmKinematics turns a grasp-center target into arm qpos and round-trips.

Validates the full env-frame -> flange -> base-body -> IK chain against MuJoCo:
placing the arm at the returned qpos must land ``rh/grasp_center`` back on the
requested pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics


def _sample_nonsingular(rng: np.random.Generator) -> np.ndarray:
  q = rng.uniform(-2.0, 2.0, 6)
  w2 = rng.uniform(0.4, 1.3)
  q[4] = w2 if rng.random() < 0.5 else -w2
  return q


@pytest.mark.parametrize("mount_z", [0.0, 0.771])
def test_grasp_center_round_trips(mount_z):
  kin = ArmKinematics(mount_pos=(0.0, 0.0, mount_z))
  rng = np.random.default_rng(0)
  for _ in range(30):
    q = _sample_nonsingular(rng)
    target = kin.fk_grasp_center_env(q)
    sol = kin.arm_qpos_for_grasp_center(target[:3, 3], target[:3, :3], seed=q)
    assert sol is not None, f"no IK for reachable grasp center from q={q}"
    got = kin.fk_grasp_center_env(sol)
    pos_err = float(np.linalg.norm(got[:3, 3] - target[:3, 3]))
    rot_err = float(np.linalg.norm(got[:3, :3] - target[:3, :3]))
    assert pos_err < 1.0e-3, f"pos err {pos_err * 1000:.3f} mm"
    assert rot_err < 1e-2, f"rot err {rot_err:.4f}"
