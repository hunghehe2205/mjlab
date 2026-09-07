"""Tests for the DexGrasp teacher observation terms (native sensing)."""

import io
import math
import warnings
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

import mjlab.tasks  # noqa: F401  (triggers task registration)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import (
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from mjlab.tasks.dexgrasp.mdp.observations import (
  LinkContactObs,
  PadContactObs,
  compute_af_vec,
  hand_center_pos,
  nearest_affordance_points,
  wrist_rot,
)

OBS_DIM = 24 + 24 + 12 + 15 + 24 + 6 + 3 + 6 + 72  # 186


def test_hand_center_position_is_environment_local() -> None:
  positions = torch.tensor([[[0.1, -0.2, 0.9]], [[2.6, 2.3, 0.9]]])
  robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=positions))
  scene = MagicMock()
  scene.__getitem__.return_value = robot
  scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.5, 2.5, 0.0]])
  env = SimpleNamespace(scene=scene)
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]

  local = hand_center_pos(cast(ManagerBasedRlEnv, env), asset_cfg=asset_cfg)

  expected = torch.tensor([[0.1, -0.2, 0.9], [0.1, -0.2, 0.9]])
  assert torch.allclose(local, expected)


def test_wrist_rot_returns_frame_x_and_z_axes() -> None:
  quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # identity, shape (B, nbody, 4)
  robot = SimpleNamespace(data=SimpleNamespace(body_link_quat_w=quat))
  scene = MagicMock()
  scene.__getitem__.return_value = robot
  env = SimpleNamespace(scene=scene)
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.body_ids = [0]

  out = wrist_rot(cast(ManagerBasedRlEnv, env), asset_cfg=asset_cfg)

  # x-axis then z-axis of an identity frame.
  torch.testing.assert_close(out, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))


def test_pad_contact_obs_flags_then_magnitudes() -> None:
  force = torch.zeros(1, 6, 3)
  force[0, 0, 0] = 2.0  # thumb pad 2 N (flags)
  force[0, 1, 0] = 0.5  # index pad 0.5 N (below threshold)
  sensor = SimpleNamespace(data=SimpleNamespace(force=force))
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene={"pad": sensor}, device="cpu"))
  cfg = SimpleNamespace(params={"sensor_name": "pad"})

  out = PadContactObs(cfg, env)(env)

  flags = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
  mags = torch.tensor([[2.0, 0.5, 0.0, 0.0, 0.0, 0.0]])
  torch.testing.assert_close(out, torch.cat([flags, mags], dim=-1))


def test_link_contact_obs_is_boolean_found() -> None:
  found = torch.tensor([[0.0, 3.0, 0.0]])  # 3 = matched contact count
  sensor = SimpleNamespace(data=SimpleNamespace(found=found))
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene={"link": sensor}, device="cpu"))
  cfg = SimpleNamespace(params={"sensor_name": "link"})

  out = LinkContactObs(cfg, env)(env)

  torch.testing.assert_close(out, torch.tensor([[0.0, 1.0, 0.0]]))


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
      pad_contact_reset = actor_obs[:, 48:60].clone()
      pd_error_reset = actor_obs[:, 24:48].clone()
      robot = env.scene["robot"]
      obj = env.scene["object"]
      # Pin the object on the index fingertip pad each step so contact is
      # sustained (instantaneous force reads the last substep): its pad flag
      # (pad index 1 -> column 49) and dip-link flag (link index 5 -> column
      # 65) must fire. Pad-mode sensing only sees the welded pad geoms.
      tip_id = robot.body_names.index("R_index_force_sensor")
      action = torch.zeros(env.num_envs, 24)
      saw_pad_contact = False
      saw_link_contact = False
      for _ in range(40):
        pose = obj.data.root_link_pose_w.clone()
        pose[:, :3] = robot.data.body_link_pos_w[:, tip_id]
        obj.write_root_link_pose_to_sim(pose)
        obs, _, _, _, _ = env.step(action)
        actor_obs = obs["actor"]
        assert isinstance(actor_obs, torch.Tensor)
        saw_pad_contact |= bool((actor_obs[:, 49] == 1.0).any())  # index pad flag
        saw_link_contact |= bool((actor_obs[:, 65] == 1.0).any())  # index dip link
      nan = bool(torch.isnan(actor_obs).any())
      shape = tuple(actor_obs.shape)
      env.close()

  assert shape == (2, OBS_DIM)
  assert not nan
  assert torch.all(pad_contact_reset == 0.0)  # nothing touches the object at reset
  assert torch.all(pd_error_reset.abs() < 1e-4)  # target anchored to qpos at reset
  assert saw_pad_contact  # object pressed into the index fingertip pad
  assert saw_link_contact  # ... and the index dip link it is welded to
