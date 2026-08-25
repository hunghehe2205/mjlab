from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.dexgrasp.mdp.metrics import (
  object_angular_speed,
  object_linear_speed,
)


def _env_with_object_velocity(linear: torch.Tensor, angular: torch.Tensor):
  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_w=linear,
      root_link_ang_vel_w=angular,
    )
  )
  scene = {"object": obj}
  return cast(ManagerBasedRlEnv, SimpleNamespace(scene=scene))


def test_object_speed_metrics_report_physical_units() -> None:
  env = _env_with_object_velocity(
    torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]),
    torch.tensor([[0.0, 0.0, 12.0], [1.0, 2.0, 2.0]]),
  )

  assert torch.equal(object_linear_speed(env), torch.tensor([5.0, 2.0]))
  assert torch.equal(object_angular_speed(env), torch.tensor([12.0, 3.0]))
