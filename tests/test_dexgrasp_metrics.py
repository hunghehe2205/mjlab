from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.dexgrasp.mdp.metrics import (
  HandObjectGrip,
  LiftSuccess,
  ObjectLiftHeight,
  hand_keypoint_below_table_depth,
  joint_pos_mean,
  mean_arm_action_magnitude,
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


def test_hand_keypoint_depth_and_arm_action_metrics() -> None:
  class Scene(dict):
    env_origins: torch.Tensor

  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor(
        [[[0.0, 0.0, 0.8], [0.0, 0.0, 0.7]], [[0.0, 0.0, 0.9], [0.0, 0.0, 0.85]]]
      )
    )
  )
  scene = Scene(robot=robot)
  scene.env_origins = torch.zeros(2, 3)
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene=scene,
      action_manager=SimpleNamespace(
        action=torch.tensor([[1.0, -0.5, 0.0], [0.2, 0.4, 0.6]])
      ),
    ),
  )
  asset_cfg = cast(SceneEntityCfg, SimpleNamespace(name="robot", body_ids=[0, 1]))

  torch.testing.assert_close(
    hand_keypoint_below_table_depth(env, 0.75, asset_cfg), torch.tensor([0.05, 0.0])
  )
  assert torch.equal(mean_arm_action_magnitude(env, 2), torch.tensor([0.75, 0.3]))


def test_joint_pos_mean_and_grip_metrics() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(joint_pos=torch.tensor([[0.1, 0.5, 0.9]]))
  )
  # force_history (E, P, H, 3): two bodies plus one pad that folds into body 0.
  history = torch.tensor([[[[0.0, 0.0, -2.0]], [[1.5, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]])
  sensor = SimpleNamespace(data=SimpleNamespace(force_history=history))
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(scene={"robot": robot, "hand_object_contact": sensor}),
  )
  asset_cfg = cast(SceneEntityCfg, SimpleNamespace(name="robot", joint_ids=[1]))
  torch.testing.assert_close(joint_pos_mean(env, asset_cfg), torch.tensor([0.5]))

  def grip(quantity: str) -> torch.Tensor:
    cfg = SimpleNamespace(
      params={
        "sensor_name": "hand_object_contact",
        "pad_parent_indices": (0,),
        "quantity": quantity,
      }
    )
    return HandObjectGrip(cfg, env)(env)

  # Impulse = force x 0.01: body 0 (0.005, 0, -0.02), body 1 (0.015, 0, 0).
  torch.testing.assert_close(grip("bodies"), torch.tensor([2.0]))
  torch.testing.assert_close(grip("squeeze_xy"), torch.tensor([0.02]))
  torch.testing.assert_close(grip("net_z"), torch.tensor([-0.02]))


def test_lift_metrics_track_displacement_from_reset() -> None:
  # At reset, xpos (root_link_pos_w) is stale from the previous episode; the
  # metrics must snapshot the freshly written qpos instead.
  obj = SimpleNamespace(
    is_fixed_base=False,
    indexing=SimpleNamespace(free_joint_q_adr=[0, 1, 2, 3, 4, 5, 6]),
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 1.3]]),  # stale
      data=SimpleNamespace(qpos=torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]])),
    ),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(scene={"object": obj}, num_envs=1, device="cpu"),
  )
  cfg = SimpleNamespace(params={"object_entity": "object", "success_height": 0.10})
  lift_height = ObjectLiftHeight(cfg, env)
  lift_success = LiftSuccess(cfg, env)
  lift_height.reset()
  lift_success.reset()
  obj.data.root_link_pos_w[:, 2] = 0.91

  torch.testing.assert_close(lift_height(env), torch.tensor([0.11]))
  assert torch.equal(lift_success(env), torch.tensor([1.0]))
