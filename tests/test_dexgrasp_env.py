"""Tests for the DexGrasp teacher env skeleton (Phase 1 §B)."""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout

import pytest
import torch

import mjlab.tasks  # noqa: F401  (triggers task registration)
from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import (
  ARM_MOUNT_Z,
  SKELETON_OBJECT,
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from mjlab.tasks.dexgrasp.dexgrasp_env_cfg import TABLE_TOP_Z
from mjlab.tasks.registry import list_tasks

TASK_ID = "Mjlab-DexGrasp-UR5eRH5DG2"


def test_task_registered() -> None:
  assert TASK_ID in list_tasks()


@pytest.mark.slow
def test_skeleton_builds_and_steps() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg()
  cfg.scene.num_envs = 2
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      obs, _ = env.reset()
      robot = env.scene["robot"]
      jn = robot.joint_names
      # Live post-reset state (env 0), read before stepping so a broken reset is
      # caught: base on the pedestal, object on the table, fingers at pre-grasp.
      base_z = float(robot.data.root_link_pos_w[0, 2])
      obj_z = float(env.scene["object"].data.root_link_pos_w[0, 2])
      finger_q = {
        n: float(robot.data.joint_pos[0, jn.index(n)]) for n in rc.FINGER_JOINT_NAMES
      }
      # 6 arm + 18 finger joints, all controllable via the delta action.
      assert env.action_manager.total_action_dim == 24
      action = torch.zeros(env.num_envs, 24)
      for _ in range(10):
        obs, _, _, _, _ = env.step(action)
      actor_obs = obs["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      env.close()

  assert not torch.isnan(actor_obs).any()
  assert base_z == pytest.approx(ARM_MOUNT_Z, abs=1e-3)
  obj = oc.PHASE1_OBJECTS[SKELETON_OBJECT]
  # + 2 mm spawn clearance baked into the sampled pose.
  assert obj_z == pytest.approx(TABLE_TOP_Z - obj.lowest_point + 0.002, abs=1e-3)
  for name, want in rc.INIT_FINGER_POSE.items():
    assert finger_q[name] == pytest.approx(want, abs=1e-3)
