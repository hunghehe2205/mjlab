"""Tests for the DexGrasp teacher observation terms (Phase 1 §E)."""

import io
import math
import warnings
from contextlib import redirect_stderr, redirect_stdout

import pytest
import torch

import mjlab.tasks  # noqa: F401  (triggers task registration)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import (
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from mjlab.tasks.dexgrasp.mdp.observations import (
  compute_af_vec,
  nearest_affordance_points,
)
from mjlab.utils.lab_api.math import matrix_from_quat

OBS_DIM = 24 + 24 + 32 + 24 + 6 + 3 + 6 + 72


def test_nearest_affordance_points() -> None:
  pcd = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
  keypoints = torch.tensor([[[0.9, 0.0, 0.0]], [[1.6, 0.0, 0.0]]])
  nearest, dist = nearest_affordance_points(keypoints, pcd)
  assert torch.allclose(nearest[0], torch.tensor([[1.0, 0.0, 0.0]]))
  assert torch.allclose(nearest[1], torch.tensor([[2.0, 0.0, 0.0]]))
  assert torch.allclose(dist, torch.tensor([[0.1], [0.4]]), atol=1e-6)


def test_af_vec_identity_pose() -> None:
  pcd = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
  keypoints_w = torch.tensor([[[1.9, 0.0, 0.0]]])
  obj_pos = torch.zeros(1, 3)
  obj_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  af = compute_af_vec(keypoints_w, obj_pos, obj_quat, pcd)
  assert torch.allclose(af, torch.tensor([[0.1, 0.0, 0.0]]), atol=1e-6)


def test_af_vec_rotates_to_world_frame() -> None:
  pcd = torch.tensor([[2.0, 0.0, 0.0]])
  keypoints_w = torch.tensor([[[0.0, 1.0, 0.0]]])  # (1, 0, 0) in object frame
  obj_pos = torch.zeros(1, 3)
  obj_quat = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
  af = compute_af_vec(keypoints_w, obj_pos, obj_quat, pcd)
  # af_vec_obj = (1, 0, 0), rotated by +90 deg about z -> (0, 1, 0).
  assert torch.allclose(af, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


@pytest.mark.slow
def test_teacher_obs_shape_and_contacts() -> None:
  cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_name="potted_meat_can")
  cfg.scene.num_envs = 2
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      obs, _ = env.reset()
      actor_obs = obs["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      contacts_reset = actor_obs[:, 48:64].clone()
      pd_error_reset = actor_obs[:, 24:48].clone()
      robot = env.scene["robot"]
      obj = env.scene["object"]
      palm_id = robot.body_names.index("R_hand_palm")
      palm_pos = robot.data.body_link_pos_w[:, palm_id].clone()
      palm_axis = matrix_from_quat(robot.data.body_link_quat_w[:, palm_id])[:, :, 0]
      # Sink the object into the palm along its approach axis: contact must fire
      # whatever the IK pose, as long as the palm is not blocked.
      pose = obj.data.root_link_pose_w.clone()
      pose[:, :3] = palm_pos - 0.01 * palm_axis
      obj.write_root_link_pose_to_sim(pose)
      action = torch.zeros(env.num_envs, 24)
      saw_palm_contact = False
      for _ in range(40):
        obs, _, _, _, _ = env.step(action)
        actor_obs = obs["actor"]
        assert isinstance(actor_obs, torch.Tensor)
        saw_palm_contact |= bool((actor_obs[:, 48] == 1.0).any())
      nan = bool(torch.isnan(actor_obs).any())
      shape = tuple(actor_obs.shape)
      env.close()

  assert shape == (2, OBS_DIM)
  assert not nan
  assert torch.all(contacts_reset == 0.0)  # nothing touches the object at reset
  assert torch.all(pd_error_reset.abs() < 1e-4)  # target anchored to qpos at reset
  assert saw_palm_contact  # object pressed into the palm
