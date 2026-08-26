"""Tests for the DexGrasp teacher env skeleton (Phase 1 §B)."""

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout

import mujoco
import pytest
import torch

import mjlab.tasks  # noqa: F401  (triggers task registration)
from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.entity.variants import VariantEntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import (
  ARM_MOUNT_Z,
  HAND_TABLE_TERMINATION_TOLERANCE,
  PHASE1_OBJECT_NAMES,
  SKELETON_OBJECT,
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from mjlab.tasks.dexgrasp.dexgrasp_env_cfg import (
  TABLE_FRICTION,
  TABLE_TOP_Z,
  get_arena_spec,
)
from mjlab.tasks.registry import list_tasks

TASK_ID = "Mjlab-DexGrasp-UR5eRH5DG2"
SINGLE_TASK_ID = f"{TASK_ID}-Single"
PHASE1_TASK_ID = f"{TASK_ID}-Phase1"


def test_task_registered() -> None:
  registered = list_tasks()
  assert TASK_ID in registered
  assert SINGLE_TASK_ID in registered
  assert PHASE1_TASK_ID in registered


def test_phase1_object_names_are_coupled_across_config() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_names=PHASE1_OBJECT_NAMES)
  object_cfg = cfg.scene.entities["object"]

  assert isinstance(object_cfg, VariantEntityCfg)
  assert tuple(object_cfg.variants) == PHASE1_OBJECT_NAMES
  assert cfg.events["reset_grasp_pose"].params["object_names"] == PHASE1_OBJECT_NAMES
  assert (
    cfg.observations["actor"].terms["af_vec"].params["object_names"]
    == PHASE1_OBJECT_NAMES
  )
  assert (
    cfg.rewards["affordance_distance"].params["object_names"] == PHASE1_OBJECT_NAMES
  )


def test_table_contact_matches_reference_friction() -> None:
  model = get_arena_spec().compile()
  table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")

  assert model.geom_friction[table_id, 0] == pytest.approx(TABLE_FRICTION)
  assert TABLE_FRICTION == pytest.approx(0.2)
  assert model.geom_priority[table_id] == 1


def test_failure_semantics_are_explicit_in_training_config() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_name=SKELETON_OBJECT)
  params = cfg.terminations["hand_below_table"].params

  assert cfg.termination_reward == pytest.approx(-10.0)
  assert params["tolerance"] == pytest.approx(HAND_TABLE_TERMINATION_TOLERANCE)


@pytest.mark.slow
def test_skeleton_builds_and_steps() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_name=SKELETON_OBJECT)
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


@pytest.mark.slow
def test_play_object_stays_fixed() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(play=True, object_name=SKELETON_OBJECT)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      env.reset()
      obj = env.scene["object"]
      robot = env.scene["robot"]
      start = obj.data.root_link_pose_w.clone()
      arm_start = robot.data.joint_pos[0, :6].clone()
      for _ in range(10):
        env.step(torch.zeros(env.num_envs, 24))
      end = obj.data.root_link_pose_w
      env.close()

  assert torch.allclose(end, start, atol=1e-6)
  home = rc.HOME_KEYFRAME.joint_pos or {}
  assert not torch.allclose(
    arm_start, torch.tensor([home[name] for name in rc.ARM_JOINT_NAMES])
  )


@pytest.mark.slow
def test_baseline_cohort_steps_with_per_world_variants() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg()
  cfg.scene.num_envs = 8
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      obs, _ = env.reset()
      obs, reward, _, _, _ = env.step(torch.zeros(env.num_envs, 24))
      variant_ids = env.sim.world_to_variant["object"].clone()
      env.close()

  actor_obs = obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  assert actor_obs.shape == (8, 191)
  assert torch.isfinite(actor_obs).all()
  assert torch.isfinite(reward).all()
  assert variant_ids.unique().numel() > 1
