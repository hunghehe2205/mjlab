"""Reset event (§C) places the arm at the analytic-IK pre-grasp in-sim."""

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
from mjlab.tasks.dexgrasp.pregrasp.generator import CAMERA_POSITION
from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics


@pytest.mark.slow
def test_reset_places_arm_at_pregrasp():
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg()
  cfg.scene.num_envs = 4
  cfg.seed = 0
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      env.reset()
      robot = env.scene["robot"]
      obj = env.scene["object"]
      gc_idx = robot.site_names.index("grasp_center")
      arm_ids = [robot.joint_names.index(n) for n in rc.ARM_JOINT_NAMES]
      gc_w = robot.data.site_pos_w[:, gc_idx].clone()
      obj_w = obj.data.root_link_pos_w.clone()
      arm_q = robot.data.joint_pos[:, arm_ids].cpu().numpy()
      origins = env.scene.env_origins.clone()
      no_nan = not bool(torch.isnan(robot.data.joint_pos).any())
      env.close()

  assert no_nan
  # Object rests on the table (z unchanged by xy sampling).
  low = oc.PHASE1_OBJECTS[SKELETON_OBJECT].lowest_point
  assert torch.allclose(obj_w[:, 2], torch.full((4,), TABLE_TOP_Z - low), atol=1e-3)

  # The in-sim grasp-center site matches the IK prediction for the written arm
  # qpos (validates the joint-index + env-frame wiring end to end).
  kin = ArmKinematics(mount_pos=(0.0, 0.0, ARM_MOUNT_Z))
  cam = torch.tensor(CAMERA_POSITION, dtype=torch.float32)
  for e in range(4):
    predicted = kin.fk_grasp_center_env(arm_q[e])[:3, 3]
    predicted = torch.tensor(predicted, dtype=torch.float32) + origins[e]
    assert torch.allclose(predicted, gc_w[e], atol=2e-3)
    # Side grasp: grasp center sits between object and the camera.
    gc_local = gc_w[e] - origins[e]
    obj_local = obj_w[e] - origins[e]
    assert torch.norm(gc_local - cam) < torch.norm(obj_local - cam)
