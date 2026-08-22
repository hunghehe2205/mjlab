"""Analytic UR5e IK round-trips against MuJoCo forward kinematics.

The IK must agree with the *simulated* arm, not textbook UR5e, so the oracle is
MuJoCo's own FK of the flange (``attachment_site``) in the base-body frame.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.tasks.dexgrasp.pregrasp.ik_ur5e import (
  InverseKinematicsUR5e,
  solve_arm_qpos,
)


@pytest.fixture(scope="module")
def fk():
  """MuJoCo FK of attachment_site in the base frame, as a 4x4, for arm qpos."""
  model = rc.get_spec().compile()
  data = mujoco.MjData(model)
  qadr = np.array(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
      for n in rc.ARM_JOINT_NAMES
    ]
  )
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

  def _fk(q: np.ndarray) -> np.ndarray:
    data.qpos[:] = 0.0
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    tb = np.eye(4)
    tb[:3, :3] = data.xmat[bid].reshape(3, 3)
    tb[:3, 3] = data.xpos[bid]
    ts = np.eye(4)
    ts[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    ts[:3, 3] = data.site_xpos[sid]
    return np.linalg.inv(tb) @ ts

  return _fk


def _pose_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
  return (
    float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
    float(np.linalg.norm(a[:3, :3] - b[:3, :3])),
  )


def _sample_nonsingular(rng: np.random.Generator) -> np.ndarray:
  """Random reachable qpos, keeping wrist_2 away from the theta5=0 singularity."""
  q = rng.uniform(-2.0, 2.0, 6)
  w2 = rng.uniform(0.4, 1.3)
  q[4] = w2 if rng.random() < 0.5 else -w2
  return q


def test_ik_roundtrips_against_mujoco_fk(fk):
  rng = np.random.default_rng(0)
  ik = InverseKinematicsUR5e()
  for _ in range(40):
    q = _sample_nonsingular(rng)
    target = fk(q)
    sol = solve_arm_qpos(target, seed=q, ik=ik)
    assert sol is not None, f"no IK solution for reachable pose from q={q}"
    pos_err, rot_err = _pose_err(fk(sol), target)
    # Bounded by the DH-fit residual (~0.5 mm ceiling measured over 500 poses).
    assert pos_err < 1.0e-3, f"pos err {pos_err * 1000:.3f} mm"
    assert rot_err < 1e-2, f"rot err {rot_err:.4f}"


def test_ik_seeded_solution_stays_near_seed(fk):
  """Seeding with the true qpos returns that branch (delta-zero is the min)."""
  rng = np.random.default_rng(1)
  ik = InverseKinematicsUR5e()
  for _ in range(20):
    q = _sample_nonsingular(rng)
    sol = solve_arm_qpos(fk(q), seed=q, ik=ik)
    assert sol is not None
    # Compare modulo 2*pi; a branch bug drifts O(1) rad vs ~0.05 rad residual.
    diff = np.abs((sol - q + np.pi) % (2 * np.pi) - np.pi)
    assert diff.max() < 6e-2, f"seeded branch drifted: {diff}"


def test_ik_returns_none_when_unreachable(fk):
  target = fk(np.zeros(6))
  target[0, 3] += 5.0  # 5 m outside the workspace
  assert solve_arm_qpos(target, seed=np.zeros(6)) is None
