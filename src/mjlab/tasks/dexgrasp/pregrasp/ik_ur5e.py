"""Closed-form inverse kinematics for the mjlab UR5e arm.

Standard 8-solution UR analytic IK, but the DH parameters are tuned to the
mjlab UR5e MJCF rather than textbook UR5e: FK residual vs MuJoCo is < ~0.5 mm
across the workspace (measured against ``attachment_site`` FK). The solver
works in the canonical UR DH frame; :func:`solve_arm_qpos` maps a flange target
given in the base-body frame into it with two robot-intrinsic offsets fitted to
MuJoCo FK:  ``T_base_flange = _B @ DH_FK(theta) @ _E``, where ``_B`` (base-body
-> DH0) is ``Rz(pi)`` (its own inverse) and ``_E`` (DH6 -> flange) is ~identity.

Borderline-reachable targets have their acos/asin arguments clipped to +/-1
within ``_ARG_TOL``, which must exceed the DH-fit residual so a pose reachable
in MuJoCo is not rejected when the ~0.5 mm model mismatch pushes an argument
just past 1 (mirrors the RaiSim solver's 1.01 flag tolerance).

Faithful reimplementation of RobustDexGrasp's analytic IK for a new embodiment
(UR5e, not UR5) -- same well-known geometry, re-derived DH/offsets.
"""

from __future__ import annotations

from math import acos, asin, atan2, cos, pi, sin, sqrt

import numpy as np

# DH tuned to the mjlab UR5e MJCF (textbook UR5e leaves ~1.4 mm; this ~0.5 mm).
_D = (0.163, 0.0, 0.0, 0.134, 0.0997, 0.0996)
_A = (0.0, -0.425, -0.392, 0.0, 0.0, 0.0)
_ALPHA = (pi / 2, 0.0, 0.0, pi / 2, -pi / 2, 0.0)

# Frame offsets fitted to MuJoCo FK; see module docstring.
_B = np.diag([-1.0, -1.0, 1.0, 1.0])
_E = np.array(
  [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -3.0e-4],
    [0.0, 0.0, 1.0, 4.0e-4],
    [0.0, 0.0, 0.0, 1.0],
  ]
)
_E_INV = np.linalg.inv(_E)

_ARG_TOL = 1e-2  # acos/asin over-range clip; see module docstring.


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
  ct, st, ca, sa = cos(theta), sin(theta), cos(alpha), sin(alpha)
  return np.array(
    [
      [ct, -st * ca, st * sa, a * ct],
      [st, ct * ca, -ct * sa, a * st],
      [0.0, sa, ca, d],
      [0.0, 0.0, 0.0, 1.0],
    ]
  )


def _inv_T(T: np.ndarray) -> np.ndarray:
  out = np.eye(4)
  out[:3, :3] = T[:3, :3].T
  out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
  return out


def _acos_safe(x: float) -> float | None:
  """acos with a small over-range tolerance; None if truly unreachable."""
  if abs(x) > 1.0 + _ARG_TOL:
    return None
  return acos(min(1.0, max(-1.0, x)))


def fk_dh(theta: np.ndarray) -> np.ndarray:
  """Forward kinematics in the DH frame (DH0 -> DH6)."""
  T = np.eye(4)
  for i in range(6):
    T = T @ _dh(_A[i], _D[i], _ALPHA[i], float(theta[i]))
  return T


class InverseKinematicsUR5e:
  """Analytic UR5e IK in the DH frame."""

  def __init__(
    self,
    joint_limits: tuple[float, float] = (-pi, pi),
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
  ) -> None:
    self._lo, self._hi = joint_limits
    self._w = np.asarray(weights, dtype=float)

  def _normalize(self, v: float) -> float:
    while v > self._hi:
      v -= 2 * pi
    while v < self._lo:
      v += 2 * pi
    return v

  def solve(self, gd: np.ndarray) -> np.ndarray | None:
    """All valid IK solutions (N, 6) in the DH frame for target ``gd``."""
    d, a, alpha = _D, _A, _ALPHA

    # theta1
    p05 = gd @ np.array([0.0, 0.0, -d[5], 1.0]) - np.array([0.0, 0.0, 0.0, 1.0])
    psi = atan2(p05[1], p05[0])
    L1 = sqrt(p05[0] ** 2 + p05[1] ** 2)
    phi_arg = d[3] / L1 if L1 > 0 else 2.0
    phi = _acos_safe(phi_arg)
    if phi is None:
      return None
    t1_candidates = (
      self._normalize(psi + phi + pi / 2),
      self._normalize(psi - phi + pi / 2),
    )

    sols: list[list[float]] = []
    for t1 in t1_candidates:
      # theta5
      p16z = gd[0, 3] * sin(t1) - gd[1, 3] * cos(t1)
      t5_abs = _acos_safe((p16z - d[3]) / d[5])
      if t5_abs is None:
        continue
      T1 = _dh(a[0], d[0], alpha[0], t1)
      T16 = _inv_T(T1) @ gd
      for t5 in (t5_abs, -t5_abs):
        # theta6
        T61 = _inv_T(T16)
        s5 = sin(t5)
        if abs(s5) < 1e-9:
          t6 = 0.0  # wrist singular: any t6; pick 0.
        else:
          t6 = atan2(-T61[1, 2] / s5, T61[0, 2] / s5)
        # theta2, theta3 (elbow up/down)
        T45 = _dh(a[4], d[4], alpha[4], t5)
        T56 = _dh(a[5], d[5], alpha[5], t6)
        T14 = T16 @ _inv_T(T45 @ T56)
        P13 = T14 @ np.array([0.0, -d[3], 0.0, 1.0]) - np.array([0.0, 0.0, 0.0, 1.0])
        nP13 = float(np.linalg.norm(P13[:3]))
        if nP13 < 1e-9:
          continue
        t3_abs = _acos_safe((nP13**2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
        if t3_abs is None:
          continue
        for t3 in (t3_abs, -t3_abs):
          t2 = -atan2(P13[1], -P13[0]) + asin(a[2] * sin(t3) / nP13)
          # theta4
          T13 = _dh(a[1], d[1], alpha[1], t2) @ _dh(a[2], d[2], alpha[2], t3)
          T34 = _inv_T(T13) @ T14
          t4 = atan2(T34[1, 0], T34[0, 0])
          sols.append([self._normalize(v) for v in (t1, t2, t3, t4, t5, t6)])

    if not sols:
      return None
    return np.array(sols)

  def closest(self, gd: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    """Weighted-nearest IK solution to ``seed`` (6,), or None if unreachable."""
    Q = self.solve(gd)
    if Q is None:
      return None
    delta = np.abs(Q - np.asarray(seed, dtype=float)) * self._w
    return Q[int(np.argmin(delta.sum(axis=1)))]


_DEFAULT_IK = InverseKinematicsUR5e()


def solve_arm_qpos(
  T_base_flange: np.ndarray,
  seed: np.ndarray,
  ik: InverseKinematicsUR5e | None = None,
) -> np.ndarray | None:
  """Arm qpos (6,) reaching a flange pose given in the base-body frame.

  ``T_base_flange`` is the desired ``attachment_site`` pose (4x4) expressed in
  the arm's ``base`` body frame; ``seed`` selects the nearest of the up-to-8
  analytic branches. Returns None when the pose is unreachable.
  """
  solver = ik if ik is not None else _DEFAULT_IK
  gd = _B @ T_base_flange @ _E_INV  # _B is its own inverse
  return solver.closest(gd, seed)
