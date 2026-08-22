"""Object pose + palm-roll sampling for the pre-grasp (§C).

Polar tabletop object poses and the `sample_rot_mats` palm-roll candidates,
ported from RobustDexGrasp's train.py / initial_pose_final.py. Pure numpy;
object-pose randomness is the caller's rng. Phase 1 samples uniformly; the
edge-biased Beta branch (``non_uniform=True``) is wired for Phase 2.

Poses are arm-base-centric (base at the origin, table in -y), the same layout
as the reference, so the polar ranges port directly; the reset event adds the
per-env origin when it writes the object root.
"""

from __future__ import annotations

import numpy as np

_ANGLE = (-0.7 * np.pi, -0.3 * np.pi)
_RADIUS = (0.45, 0.75)
_X_LIMIT = 0.25


def sample_object_pose(
  rng: np.random.Generator,
  table_top_z: float,
  lowest_point: float,
  non_uniform: bool = False,
) -> np.ndarray:
  """Polar tabletop pose as pos3 + wxyz quat (7,), resting on the table top."""

  def uniform_xy() -> tuple[float, float]:
    while True:
      angle = rng.uniform(*_ANGLE)
      dist = rng.uniform(*_RADIUS)
      x, y = dist * np.cos(angle), dist * np.sin(angle)
      if -_X_LIMIT < x < _X_LIMIT:
        return x, y

  def beta_xy() -> tuple[float, float]:
    while True:
      angle = _ANGLE[0] + rng.beta(0.5, 0.5) * (_ANGLE[1] - _ANGLE[0])
      dist = _RADIUS[0] + rng.beta(0.5, 0.5) * (_RADIUS[1] - _RADIUS[0])
      x, y = dist * np.cos(angle), dist * np.sin(angle)
      if -_X_LIMIT < x < _X_LIMIT:
        return x, y

  if non_uniform and rng.random() >= 0.5:
    x, y = beta_xy()
  else:
    x, y = uniform_xy()

  yaw = rng.uniform(-np.pi, np.pi)
  pose = np.zeros(7)
  pose[0], pose[1], pose[2] = x, y, table_top_z - lowest_point
  pose[3] = np.cos(yaw / 2.0)  # qw
  pose[6] = np.sin(yaw / 2.0)  # qz (rotation about world z)
  return pose


def sample_rot_mats(
  approach_dir_w: np.ndarray,
  num_samples: int,
  visible_points_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Palm-roll candidates about the approach axis + object projection widths.

  Returns ``num_samples`` proper rotation matrices (num_samples, 3, 3), palm x
  along ``-approach_dir_w``, and the object cloud's width (num_samples,) along
  each candidate's gripper axis (used to prefer narrow grasps).
  """
  ref = approach_dir_w.reshape(3)
  temp = np.array([1.0, 0.0, 0.0]) if abs(ref[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
  first = np.cross(ref, temp)
  first = first / np.linalg.norm(first)
  second = np.cross(ref, first)
  second = second / np.linalg.norm(second)

  thetas = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
  perp = np.zeros((num_samples, 3))
  for j, theta in enumerate(thetas):
    v = first * np.cos(theta) + second * np.sin(theta)
    v = v / np.linalg.norm(v)
    if v[1] < 0:
      v = -v  # canonical sign so mirrored rolls don't double-count
    perp[j] = v

  pts = visible_points_w.reshape(-1, 3)
  centered = pts - pts.mean(axis=0)
  proj = centered @ perp.T
  proj_range = proj.max(axis=0) - proj.min(axis=0)

  rot_mats = np.zeros((num_samples, 3, 3))
  for j in range(num_samples):
    y_dir = np.cross(ref, perp[j])
    y_dir = y_dir / np.linalg.norm(y_dir)
    rot_mats[j] = -np.stack((ref, y_dir, perp[j]), axis=-1)
  return rot_mats, proj_range
