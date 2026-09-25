"""Teacher PPO with the reference std floor, and a runner with a periodic lift test."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import LIFT_TEST_ENVS, MIN_ACTION_STD
from mjlab.tasks.ur5e_rh5dg2.grasp.evaluate import lift_test_env, run_lift_test


class FlooredPPO(PPO):
  """Projects the action std onto its floor after each update, as the reference's
  ``enforce_minimum_std``; a clamp in the forward pass would stop its gradient."""

  def update(self) -> dict[str, float]:
    losses = super().update()
    std = getattr(self.get_policy().distribution, "std_param", None)
    assert isinstance(std, torch.nn.Parameter)
    with torch.no_grad():
      std.clamp_(min=MIN_ACTION_STD)
    return losses


class GraspTeacherRunner(MjlabOnPolicyRunner):
  """Runs the lift test on every saved checkpoint and logs it under Lift_Test/."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._lift_env: RslRlVecEnvWrapper | None = None

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    if self.logger.writer is None:
      return
    if self._lift_env is None:
      self._lift_env = lift_test_env(LIFT_TEST_ENVS, self.device)
    results = run_lift_test(self._lift_env, self.alg.get_policy())
    for name, value in results.items():
      self.logger.writer.add_scalar(
        f"Lift_Test/{name}", value, self.current_learning_iteration
      )
