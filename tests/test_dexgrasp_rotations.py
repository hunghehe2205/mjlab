"""Unit tests for the DexGrasp rotation helpers (§E)."""

import math

import torch

from mjlab.tasks.dexgrasp.rotations import euler_from_rotmat, unwrap_euler


def _rot_z(angle: float) -> torch.Tensor:
  c, s = math.cos(angle), math.sin(angle)
  return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_euler_identity() -> None:
  assert torch.allclose(euler_from_rotmat(torch.eye(3)), torch.zeros(3), atol=1e-6)


def test_euler_z_rotation() -> None:
  euler = euler_from_rotmat(_rot_z(math.pi / 2))
  assert torch.allclose(euler, torch.tensor([0.0, 0.0, math.pi / 2]), atol=1e-6)


def test_euler_roundtrip() -> None:
  rng = torch.Generator().manual_seed(0)
  for _ in range(100):
    q = torch.randn(4, generator=rng)
    r, _ = torch.linalg.qr(torch.randn(3, 3, generator=rng))
    r = r * torch.det(r).sign()
    e = euler_from_rotmat(r)
    assert not torch.isnan(e).any()
    assert torch.all(torch.abs(e) <= math.pi + 1e-5)
    del q


def test_unwrap_returns_raw_when_prev_near_zero() -> None:
  prev = torch.zeros(3)
  euler = torch.tensor([2.0, 1.0, -2.0])
  assert torch.allclose(unwrap_euler(euler, prev), euler)


def test_unwrap_across_pi() -> None:
  prev = torch.tensor([0.0, 0.0, 3.0])
  euler = torch.tensor([0.0, 0.0, -3.0])
  out = unwrap_euler(euler, prev)
  assert torch.allclose(out, torch.tensor([0.0, 0.0, 3.0 + 2.0 * math.pi - 6.0]))


def test_unwrap_small_delta_unchanged() -> None:
  prev = torch.tensor([0.1, 0.2, 0.3])
  euler = torch.tensor([0.11, 0.21, 0.31])
  assert torch.allclose(unwrap_euler(euler, prev), euler)


def test_unwrap_batched() -> None:
  prev = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
  euler = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, -3.0]])
  out = unwrap_euler(euler, prev)
  assert torch.allclose(out[0], euler[0])  # prev near zero -> raw
  assert torch.allclose(
    out[1], torch.tensor([0.0, 0.0, 3.0 + 2.0 * math.pi - 6.0]), atol=1e-6
  )
