"""Rotation helpers for the teacher observation.

The euler convention mirrors the reference environment's patched RaiSim
``RotmatToEuler`` (fixed-frame XYZ with the same sign choices and gimbal
branch), so the wrist euler and the init-relative wrist euler feed the policy
in the representation the method was trained with.
"""

from __future__ import annotations

import torch

_EPS = 8.881784197001252e-16


def euler_from_rotmat(r: torch.Tensor) -> torch.Tensor:
  """Fixed-frame XYZ euler of a rotation matrix (RaiSim convention).

  Args:
    r: (..., 3, 3) rotation matrices.

  Returns:
    (..., 3) euler angles.
  """
  cy = torch.sqrt(r[..., 2, 2] ** 2 + r[..., 1, 2] ** 2)
  e0 = -torch.atan2(r[..., 1, 2], r[..., 2, 2])
  e1 = -torch.atan2(-r[..., 0, 2], cy)
  e2 = torch.where(
    cy > _EPS,
    -torch.atan2(r[..., 0, 1], r[..., 0, 0]),
    -torch.atan2(-r[..., 1, 0], r[..., 1, 1]),
  )
  return torch.stack([e0, e1, e2], dim=-1)


def unwrap_euler(euler: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
  """Component-wise +/-2pi unwrap of ``euler`` against the previous output.

  Mirrors the reference state machine: no unwrap while the previous euler is
  (near) zero, so the first frame after a reset stays raw.
  """
  d = euler - prev
  d = torch.where(d > torch.pi, d - 2.0 * torch.pi, d)
  d = torch.where(d < -torch.pi, d + 2.0 * torch.pi, d)
  active = prev.norm(dim=-1, keepdim=True) > 0.01
  return torch.where(active, prev + d, euler)
