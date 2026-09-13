import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.ur5e_rh5dg2.view_env_cfg import ur5e_rh5dg2_view_env_cfg


def test_view_env_builds_and_steps():
  cfg = ur5e_rh5dg2_view_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  adim = env.action_manager.total_action_dim
  assert adim == 24  # 6 arm + 18 hand
  for _ in range(2):
    env.step(torch.zeros((env.num_envs, adim), device=env.device))
  assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
