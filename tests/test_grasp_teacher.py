"""Teacher action semantics, reset isolation, contacts, rewards and lift test."""

import math

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.ur5e_rh5dg2.ur5e_rh5dg2_constants import get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import (
  GRASP_TIME,
  HAND_CENTER,
  LIFT_OFFSET,
  LIFT_RAMP,
  OBJECT_ANGLE,
  OBJECT_DISTANCE,
  OBJECT_MAX_ABS_X,
  OBJECT_POS,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp import events, rewards, signals, terminations
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspAction
from mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe import (
  APPROACH_END,
  RolloutAudit,
  ScriptedGrasp,
  run_native_probe,
  run_warp_probe,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.pregrasp import PregraspSolver, build_pool
from mjlab.tasks.ur5e_rh5dg2.grasp.robot import teacher_robot_spec
from mjlab.tasks.ur5e_rh5dg2.grasp.teacher_env_cfg import teacher_env_cfg


@pytest.fixture(scope="module")
def teacher():
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield env
  env.close()


@pytest.fixture
def env(teacher):
  teacher.reset()
  return teacher


def place_same(env):
  """Put every env in the first pooled placement."""
  term = env.event_manager.get_term_cfg("reset_pregrasp").func
  assert isinstance(term, events.PregraspReset)
  n = env.num_envs
  term.write(
    env,
    torch.arange(n, device=env.device),
    term.object_pos[:1].expand(n, -1),
    term.object_quat[:1].expand(n, -1),
    term.arm_pos[:1].expand(n, -1),
    term.lift_pos[:1].expand(n, -1),
  )
  env.sim.forward()
  env.sim.sense()
  env.scene.update(0.0)
  env.action_manager.get_term("joint_pos").reset()
  env.observation_manager.reset(torch.arange(n, device=env.device))


def test_translated_env_observations_and_steps(env):
  place_same(env)
  obs = env.get_observations()
  assert env.action_manager.total_action_dim == 24
  assert obs["actor"].shape == (2, 258)
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
  assert not env.termination_manager.terminated.any()


def test_action_target_accumulates_held_delayed_and_partial_reset(env):
  action = env.action_manager.get_term("joint_pos")
  assert isinstance(action, GraspAction)
  robot = env.scene["robot"]
  q = robot.data.joint_pos[:, action.target_ids].clone()
  torch.testing.assert_close(action.target, q)
  limits = robot.data.joint_pos_limits[:, action.target_ids]
  expected = q
  for _ in range(2):
    action.process_actions(torch.full_like(q, 10.0))
    expected = torch.clamp(
      expected + action.scale, min=limits[..., 0], max=limits[..., 1]
    )
    torch.testing.assert_close(action.target, expected)
  previous = torch.clamp(q + action.scale, min=limits[..., 0], max=limits[..., 1])
  robot.write_joint_state_to_sim(q - 0.01, torch.zeros_like(q))
  action._delay = torch.tensor([True, False], device=env.device)
  action.apply_actions()
  torch.testing.assert_close(robot.data.joint_pos_target[0], previous[0])
  torch.testing.assert_close(robot.data.joint_pos_target[1], expected[1])
  for _ in range(2):
    action.apply_actions()
    torch.testing.assert_close(robot.data.joint_pos_target, expected)
  action.process_actions(torch.zeros_like(q))
  torch.testing.assert_close(action.target, expected)
  for _ in range(20):
    action.process_actions(torch.ones_like(q))
  torch.testing.assert_close(action.target[:, :6], q[:, :6] - 0.01 + 0.10)
  ids = torch.tensor([0], device=env.device)
  other_q = robot.data.joint_pos[1].clone()
  other_target = action.target[1].clone()
  env.reset(env_ids=ids)
  torch.testing.assert_close(
    action.target[0], robot.data.joint_pos[0, action.target_ids]
  )
  torch.testing.assert_close(action.target[1], other_target)
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


def test_reward_signs_weights_and_lift_check(env):
  heights = torch.tensor([-0.02, 0.0, 0.01, 0.02, 0.1])
  cost = rewards.clearance_cost(heights)
  assert torch.isfinite(cost).all()
  assert (cost[:-1] >= cost[1:]).all()
  assert cost[-1] == 0
  for name, cfg in env.cfg.rewards.items():
    assert (cfg.weight > 0) == (name in ("contact", "grip")), name
  for weights in (rewards.contact_weights(), rewards.distance_weights()):
    assert weights[0] == 0 and math.isclose(sum(weights), 1.0)
  assert "lift_height" not in env.cfg.metrics
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, 2] += 0.5
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros((2, 6), device=env.device))
  env.sim.forward()
  env.sim.sense()
  env.scene.update(0.0)
  torch.testing.assert_close(
    rewards.object_displacement(env), torch.full((2,), 0.5, device=env.device)
  )
  assert terminations.lifted(env).all()
  assert rewards.contact(env).eq(0).all() and rewards.grip(env).eq(0).all()


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


def test_lift_test_ramps_arm_after_grasp_phase():
  cfg = teacher_env_cfg(lift_test=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    env.reset()
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, GraspAction)
    grasp_steps = round(GRASP_TIME / env.step_dt)
    ramp_steps = math.ceil(LIFT_RAMP / env.step_dt)
    assert env.max_episode_length > grasp_steps + ramp_steps
    close = torch.ones((1, 24), device=env.device)
    close[:, :6] = 0.0
    scale = action.scale
    assert isinstance(scale, torch.Tensor)
    for _ in range(grasp_steps - 1):
      env.step(close)
    q = env.scene["robot"].data.joint_pos[:, action.target_ids[6:]].clone()
    previous = action.target[:, 6:].clone()
    env.step(close)
    torch.testing.assert_close(
      action.target[:, 6:],
      torch.clamp(
        torch.minimum(previous + scale[:, 6:], q + 0.5),
        max=env.scene["robot"].data.joint_pos_limits[:, action.target_ids[6:], 1],
      ),
    )
    start = action.target[:, :6].clone()
    delta = events.lift_delta(env)
    env.step(close)
    torch.testing.assert_close(action.target[:, :6], start + delta / ramp_steps)
    for _ in range(ramp_steps):
      env.step(close)
    torch.testing.assert_close(action.target[:, :6], start + delta)
    assert "success" in env.cfg.metrics
  finally:
    env.close()


def test_partial_reset_restores_object_and_clears_contacts(env):
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, 0] += 0.1
  obj.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  start = events.object_start(env).clone()
  env.reset(env_ids=torch.tensor([0], device=env.device))
  local = obj.data.root_link_pos_w - env.scene.env_origins
  torch.testing.assert_close(local[0], events.object_start(env)[0])
  torch.testing.assert_close(local[1], start[1] + local.new_tensor([0.1, 0, 0]))
  assert not signals.finger_contacts(env)[0].any()


def test_pregrasp_pool_is_in_region_palm_down_and_collision_free():
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, device="cpu").compile()
  pool = build_pool(model, 16, seed=0, edge_biased=True)
  xy = pool.object_pos[:, :2]
  angle = np.arctan2(xy[:, 1], xy[:, 0])
  distance = np.linalg.norm(xy, axis=-1)
  assert ((angle >= OBJECT_ANGLE[0]) & (angle <= OBJECT_ANGLE[1])).all()
  assert ((distance >= OBJECT_DISTANCE[0]) & (distance <= OBJECT_DISTANCE[1])).all()
  assert (np.abs(xy[:, 0]) < OBJECT_MAX_ABS_X).all()
  np.testing.assert_allclose(pool.object_pos[:, 2], OBJECT_POS[2])
  solver = PregraspSolver(model)
  lo, hi = solver.arm_limits.T
  for pos, quat, arm, lift in zip(
    pool.object_pos, pool.object_quat, pool.arm_pos, pool.lift_pos, strict=True
  ):
    assert ((arm >= lo) & (arm <= hi)).all()
    assert not solver.collides(arm, pos, quat)
    rotation = solver.data.xmat[solver.wrist].reshape(3, 3).copy()
    wrist = solver.data.xpos[solver.wrist].copy()
    np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-3)
    solver.set_state(lift, pos, quat)
    mujoco.mj_kinematics(model, solver.data)
    np.testing.assert_allclose(
      solver.data.xpos[solver.wrist] - wrist, [0, 0, LIFT_OFFSET], atol=1e-3
    )
    np.testing.assert_allclose(
      solver.data.xmat[solver.wrist].reshape(3, 3), rotation, atol=1e-3
    )


def test_reset_places_sampled_pregrasp_and_tracks_start(env):
  local = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  torch.testing.assert_close(local, events.object_start(env))
  torch.testing.assert_close(
    rewards.object_displacement(env), torch.zeros(2, device=env.device)
  )
  term = events.pregrasp_reset(env)
  arm = env.scene["robot"].data.joint_pos[:, term.joint_ids[:6]]
  torch.testing.assert_close(
    events.lift_delta(env),
    term.lift_pos[(term.arm_pos[None] - arm[:, None]).abs().sum(-1).argmin(-1)] - arm,
  )
  assert (signals.normal_force(env, "hand_object") == 0).all()


def test_scripted_approach_is_collision_free():
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, device="cpu").compile()
  script = ScriptedGrasp.solve(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
  assert joint_pos is not None
  joint_ids = np.array([model.joint(f"robot/{name}").id for name in joint_pos])
  qpos_ids = model.jnt_qposadr[joint_ids]
  palm = model.body("robot/right_hand").id
  for t in np.linspace(0.0, APPROACH_END, 61):
    q = script.target(float(t))
    assert (q >= model.jnt_range[joint_ids, 0]).all()
    assert (q <= model.jnt_range[joint_ids, 1]).all()
    data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, data)
    rotation = data.xmat[palm].reshape(3, 3)
    np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-3)
    assert (data.contact.dist >= -1e-5).all()
  rotation = data.xmat[palm].reshape(3, 3)
  local = rotation.T @ (np.array(OBJECT_POS) - data.xpos[palm])
  np.testing.assert_allclose(local, HAND_CENTER, atol=1e-3)

  # Another IK branch reaches the same palm pose but has unnamed arm collision
  # geoms interpenetrating. The audit must classify them by body ownership.
  data.qpos[qpos_ids[:6]] = (
    -1.7897409333,
    -0.9473604500,
    2.7322809275,
    1.3566721760,
    -1.3518517203,
    1.5707963268,
  )
  mujoco.mj_forward(model, data)
  audit = RolloutAudit()
  audit.update(
    model, data.contact.geom, data.contact.dist, np.array(OBJECT_POS), True, 0.0
  )
  assert audit.undesired_penetration_m > 0.005
  assert not audit.passed


def test_scripted_native_lift_with_clean_approach():
  result = run_native_probe()
  assert result.success, result
  assert result.initial_penetration_m < 1e-5
  assert result.final_lift_m >= 0.10
  assert result.hold_seconds >= 3.0
  assert result.rollout.passed


@pytest.mark.slow
def test_scripted_warp_lift_with_clean_approach():
  result = run_warp_probe(get_test_device())
  assert result.success, result
  assert result.initial_penetration_m < 1e-5
  assert result.final_lift_m >= 0.10
  assert result.hold_seconds >= 3.0
  assert result.rollout.passed


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
