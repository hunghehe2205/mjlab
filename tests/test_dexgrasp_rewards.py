"""Tests for the DexGrasp reward terms (Phase 1 §F)."""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import mjlab.tasks  # noqa: F401  (triggers task registration)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import (
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from mjlab.tasks.dexgrasp.mdp.rewards import (
  REWARD_COEFFS,
  ContactReward,
  affordance_weights,
  contact_weights,
)


def test_affordance_weights() -> None:
  w = affordance_weights(
    tip_indices=(5, 10, 15, 19, 23),
    thumb_tip_index=5,
    wrist_index=0,
    num=24,
  )
  assert w[0] == 0.0  # wrist zeroed
  assert w[5] == pytest.approx(w[10] * 2.0)  # thumb tip = 8x vs other tip 4x
  assert w.sum() == pytest.approx(16.0)
  assert w[10] > w[6]  # tip > joint


def test_contact_weights() -> None:
  w = contact_weights(
    tip_indices=(3, 6, 9, 12, 15),
    thumb_indices=(1, 2, 3),
    thumb_tip_index=3,
    palm_index=0,
    num=16,
  )
  assert w[0] == 0.0  # palm zeroed
  assert w[3] == pytest.approx(w[6] * 4.0)  # thumb tip = 3*2*2 vs index tip 3
  assert w[1] == pytest.approx(w[4] * 2.0)  # thumb link vs index link
  assert w.sum() == pytest.approx(16.0)


def test_reward_coeffs_complete() -> None:
  # Every reference coeff maps to a configured term weight.
  expected = {
    "affordance_distance": 0.5,
    "affordance_contact": 1.5,
    "affordance_impulse": 1.0,
    "table_logbarrier": -0.03,
    "table_contact": -1.0,
    "table_impulse": -0.5,
    "arm_height_logbarrier": -0.05,
    "arm_contact": -0.1,
    "arm_impulse": -0.1,
    "arm_collision": -1.0,
    "object_velocity": -15.0,
    "object_angular_velocity": -0.2,
    "object_displacement": -5.0,
    "wrist_velocity": -1.0,
    "wrist_angular_velocity": -0.1,
    "arm_joint_velocity": -1.0,
  }
  assert REWARD_COEFFS == expected


def test_contact_reward_combines_pair_impulses_before_norm() -> None:
  table = SimpleNamespace(
    data=SimpleNamespace(force_history=torch.tensor([[[[1.0, 0.0, 0.0]]]]))
  )
  obj = SimpleNamespace(
    data=SimpleNamespace(force_history=torch.tensor([[[[-1.0, 0.0, 0.0]]]]))
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"table": table, "object": obj},
      sim=SimpleNamespace(cfg=SimpleNamespace(mujoco=SimpleNamespace(timestep=1.0))),
      device="cpu",
    ),
  )
  cfg = SimpleNamespace(params={"sensor_names": ("table", "object"), "mode": "impulse"})

  reward = ContactReward(cfg, env)(env)

  torch.testing.assert_close(reward, torch.zeros(1))


@pytest.mark.slow
def test_object_displacement_uses_reset_pose() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_name="potted_meat_can")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      env.reset()
      obj = env.scene["object"]
      root_pose = obj.data.data.qpos[:, obj.indexing.free_joint_q_adr].clone()
      root_pose[:, 0] += 0.02
      obj.write_root_link_pose_to_sim(root_pose)
      env.sim.forward()
      term = env.reward_manager.get_term_cfg("object_displacement").func
      displacement = term(env)
      env.close()

  torch.testing.assert_close(displacement, torch.full((1,), 0.02), atol=1e-6, rtol=0)


@pytest.mark.slow
def test_rewards_finite_and_quiet_at_reset() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_name="potted_meat_can")
  cfg.scene.num_envs = 2
  cfg.seed = 0
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      env.reset()
      action = torch.zeros(env.num_envs, 24)
      per_step: dict[str, list[float]] = {}
      for _ in range(70):
        _, reward, _, _, _ = env.step(action)
        for name, vals in env.reward_manager.get_active_iterable_terms(0):
          per_step.setdefault(name, []).append(vals[0])
        assert bool(torch.isfinite(reward).all())
      env.close()

  # The pregrasp pose starts contact-free (hand ~0.25 m from the object).
  assert per_step["affordance_contact"][0] == 0.0
  assert per_step["affordance_impulse"][0] == 0.0
  # The arm never hits the table/object over the whole episode.
  assert all(v == 0.0 for v in per_step["arm_collision"])
  assert all(v == 0.0 for v in per_step["arm_contact"])
