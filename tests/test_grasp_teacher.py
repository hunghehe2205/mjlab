"""Teacher action semantics, reset isolation, contacts and success conditions."""

import math

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.ur5e_rh5dg2.ur5e_rh5dg2_constants import get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import LIFT_HEIGHT, OBJECT_POS
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp import rewards, signals, terminations
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspAction
from mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe import run_native_probe, run_warp_probe
from mjlab.tasks.ur5e_rh5dg2.grasp.robot import teacher_robot_spec
from mjlab.tasks.ur5e_rh5dg2.grasp.teacher_env_cfg import teacher_env_cfg


@pytest.fixture(scope="module")
def teacher():
  cfg = teacher_env_cfg(hold_time=0.15)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield env
  env.close()


@pytest.fixture
def env(teacher):
  teacher.reset()
  return teacher


def test_translated_env_observations_and_steps(env):
  obs = env.get_observations()
  assert env.action_manager.total_action_dim == 24
  assert obs["actor"].shape == (2, 259)
  torch.testing.assert_close(obs["actor"][0], obs["actor"][1], atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(obs["actor"], obs["critic"])
  for name in ("robot", "object", "props"):
    local = env.scene[name].data.root_link_pos_w - env.scene.env_origins
    torch.testing.assert_close(local[0], local[1])
  for _ in range(3):
    obs, reward, _, _, _ = env.step(torch.zeros((2, 24), device=env.device))
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
  assert (signals.normal_force(env, "object_table") > 0.1).all()
  assert not signals.finger_contacts(env).any()
  assert not env.termination_manager.get_term("success").any()


def test_action_target_held_clipped_and_partial_reset(env):
  action = env.action_manager.get_term("joint_pos")
  assert isinstance(action, GraspAction)
  robot = env.scene["robot"]
  q = robot.data.joint_pos[:, action.target_ids].clone()
  action.process_actions(torch.full_like(q, 10.0))
  limits = robot.data.joint_pos_limits[:, action.target_ids]
  expected = torch.clamp(q + action.scale, min=limits[..., 0], max=limits[..., 1])
  torch.testing.assert_close(action.target, expected)
  robot.write_joint_state_to_sim(q - 0.01, torch.zeros_like(q))
  for _ in range(3):
    action.apply_actions()
    torch.testing.assert_close(robot.data.joint_pos_target, expected)
  ids = torch.tensor([0], device=env.device)
  other_q = robot.data.joint_pos[1].clone()
  env.reset(env_ids=ids)
  torch.testing.assert_close(action.target[0], robot.data.default_joint_pos[0])
  torch.testing.assert_close(action.target[1], expected[1])
  torch.testing.assert_close(robot.data.joint_pos[1], other_q)
  assert (action.raw_action[0] == 0).all()


def test_box_surface_vectors_inside_outside_and_translation():
  points = torch.tensor([[[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]])
  pos = torch.zeros((1, 3))
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  actual = signals.box_surface_vectors(points, pos, quat)
  torch.testing.assert_close(
    actual, torch.tensor([[[-0.02, 0.0, 0.0], [0.03, 0.0, 0.0]]])
  )
  translated = signals.box_surface_vectors(points + 2, pos + 2, quat)
  torch.testing.assert_close(actual, translated, atol=1e-6, rtol=1e-5)
  angle = math.pi / 4
  rotation = torch.tensor([[math.cos(angle), 0, 0, math.sin(angle)]])
  rotated = signals.box_surface_vectors(torch.tensor([[[0, 0.05, 0.0]]]), pos, rotation)
  torch.testing.assert_close(
    rotated, torch.tensor([[[0, -0.02, 0.0]]]), atol=1e-6, rtol=1e-5
  )


def test_reward_signs_and_vertical_motion(env):
  heights = torch.tensor([-0.02, 0.0, 0.01, 0.02, 0.1])
  cost = rewards.clearance_cost(heights)
  assert torch.isfinite(cost).all()
  assert (cost[:-1] >= cost[1:]).all()
  assert cost[-1] == 0
  for name in (
    "clearance",
    "undesired_contact",
    "object_displacement_xy",
    "object_velocity_xy",
    "joint_velocity",
    "palm_velocity",
  ):
    cfg = env.cfg.rewards[name]
    assert cfg.weight < 0
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, 2] += 0.5
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros((2, 6), device=env.device))
  env.sim.forward()
  env.sim.sense()
  env.scene.update(0.0)
  torch.testing.assert_close(
    rewards.horizontal_displacement(env),
    torch.zeros(2, device=env.device),
    atol=1e-6,
    rtol=0,
  )
  assert not signals.stable_hold(env, LIFT_HEIGHT).any()
  assert not rewards.lift(env, LIFT_HEIGHT).any()


def test_contact_sensor_reads_live_warp_state(env):
  robot = env.scene["robot"]
  tip_ids, _ = robot.find_bodies("R_middle_force_sensor")
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = robot.data.body_link_pos_w[:, tip_ids[0]]
  obj.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  env.sim.sense()
  env.scene.update(0.0)
  assert (signals.normal_force(env, "hand_object").amax(dim=-1) > 0.1).all()
  assert signals.finger_contacts(env).any(dim=-1).all()
  assert env.sim.mj_data.time == 0


def test_hold_counter_continuity_idempotence_and_partial_reset(env, monkeypatch):
  term = env.termination_manager.get_term_cfg("success").func
  assert isinstance(term, terminations.LiftSuccess)
  valid = torch.ones(2, dtype=torch.bool, device=env.device)
  monkeypatch.setattr(terminations, "stable_hold", lambda *_: valid)
  for _ in range(2):
    env.episode_length_buf += 1
    assert not term(env, LIFT_HEIGHT, 0.15).any()
    before = term.count.clone()
    env.get_observations()
    term(env, LIFT_HEIGHT, 0.15)
    torch.testing.assert_close(term.count, before)
  valid[0] = False
  env.episode_length_buf += 1
  result = term(env, LIFT_HEIGHT, 0.15)
  assert result.tolist() == [False, True]
  env.reset(env_ids=torch.tensor([0], device=env.device))
  assert term.count.tolist() == [0, 3]


def test_partial_reset_restores_object_and_clears_contacts(env):
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, 0] += 0.1
  obj.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  env.reset(env_ids=torch.tensor([0], device=env.device))
  local = obj.data.root_link_pos_w - env.scene.env_origins
  torch.testing.assert_close(local[0], local.new_tensor(OBJECT_POS))
  torch.testing.assert_close(
    local[1], local.new_tensor(OBJECT_POS) + local.new_tensor([0.1, 0, 0])
  )
  assert not signals.finger_contacts(env)[0].any()


def test_scripted_native_lift_without_initial_penetration():
  result = run_native_probe()
  assert result.success, result
  assert result.initial_penetration_m < 1e-5
  assert result.final_lift_m >= 0.10
  assert result.hold_seconds >= 3.0


@pytest.mark.slow
def test_scripted_warp_lift_without_initial_penetration():
  result = run_warp_probe(get_test_device())
  assert result.success, result
  assert result.initial_penetration_m < 1e-5
  assert result.final_lift_m >= 0.10
  assert result.hold_seconds >= 3.0


def test_rounded_pads_keep_body_inertias_and_fit_original_bounds():
  original = get_spec().compile()
  rounded = teacher_robot_spec().compile()
  np.testing.assert_array_equal(original.body_mass, rounded.body_mass)
  np.testing.assert_array_equal(original.body_inertia, rounded.body_inertia)
  for i in range(original.ngeom):
    if original.geom_type[i] == rounded.geom_type[i]:
      continue
    assert "_dip_collision_" in original.geom(i).name
    assert original.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
    radius, half_length = rounded.geom_size[i, :2]
    assert radius <= original.geom_size[i, 0]
    assert radius + half_length <= original.geom_size[i, 1] + 1e-12
    np.testing.assert_array_equal(original.geom_pos[i], rounded.geom_pos[i])
    np.testing.assert_array_equal(original.geom_quat[i], rounded.geom_quat[i])
