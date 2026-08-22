"""Pre-grasp pose sampling: polar object poses + palm-roll rotation candidates."""

from __future__ import annotations

import numpy as np
import pytest

from mjlab.tasks.dexgrasp.pregrasp.pose_sampler import (
  sample_object_pose,
  sample_rot_mats,
)

_EPS = 1e-9


def test_sample_object_pose_respects_polar_constraints():
  rng = np.random.default_rng(0)
  for _ in range(500):
    p = sample_object_pose(rng, table_top_z=0.771, lowest_point=-0.03)
    x, y, z = p[:3]
    r = float(np.hypot(x, y))
    angle = float(np.arctan2(y, x))
    assert 0.45 - _EPS <= r <= 0.75 + _EPS
    assert -0.7 * np.pi - _EPS <= angle <= -0.3 * np.pi + _EPS
    assert abs(x) < 0.25
    assert z == pytest.approx(0.771 + 0.03)
    assert p[4] == pytest.approx(0.0) and p[5] == pytest.approx(0.0)
    assert float(np.linalg.norm(p[3:7])) == pytest.approx(1.0)


def test_sample_object_pose_is_deterministic_per_seed():
  a = sample_object_pose(np.random.default_rng(3), 0.771, -0.03)
  b = sample_object_pose(np.random.default_rng(3), 0.771, -0.03)
  assert np.allclose(a, b)


def test_sample_rot_mats_are_proper_rotations():
  rng = np.random.default_rng(0)
  approach = np.array([0.3, -0.5, 0.8])
  approach = approach / np.linalg.norm(approach)
  pts = rng.normal(size=(200, 3)) * 0.05 + np.array([0.1, -0.5, 0.8])
  rot_mats, proj_range = sample_rot_mats(approach, 10, pts)
  assert rot_mats.shape == (10, 3, 3)
  assert proj_range.shape == (10,)
  assert (proj_range >= 0).all()
  for m in rot_mats:
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(m)) == pytest.approx(1.0)
