"""Reset to sampled pre-grasp placements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import OPEN_HAND
from mjlab.tasks.ur5e_rh5dg2.grasp.pregrasp import ARM_JOINTS, HAND_JOINTS, build_pool

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PregraspReset:
  """Draw object placements and IK pre-grasps from a pool solved at startup."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    pool = build_pool(
      env.sim.mj_model,
      cfg.params["pool_size"],
      int(np.random.randint(2**31)),
      cfg.params["edge_biased"],
    )

    def as_tensor(x) -> torch.Tensor:
      return torch.tensor(x, device=env.device, dtype=torch.float32)

    self.object_pos = as_tensor(pool.object_pos)
    self.object_quat = as_tensor(pool.object_quat)
    self.arm_pos = as_tensor(pool.arm_pos)
    self.open_hand = as_tensor(OPEN_HAND)
    ids, _ = env.scene["robot"].find_joints(
      ARM_JOINTS + HAND_JOINTS, preserve_order=True
    )
    self.joint_ids = torch.tensor(ids, device=env.device)
    self.object_start = torch.zeros(env.num_envs, 3, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pool_size: int,
    edge_biased: bool,
  ) -> None:
    del pool_size, edge_biased
    env_ids = resolve_env_ids(env, env_ids)
    idx = torch.randint(len(self.arm_pos), (len(env_ids),), device=env.device)
    self.write(
      env, env_ids, self.object_pos[idx], self.object_quat[idx], self.arm_pos[idx]
    )

  def write(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    arm_pos: torch.Tensor,
  ) -> None:
    """Place the object and the open hand; positions are env-local."""
    root = torch.zeros(len(env_ids), 13, device=env.device)
    root[:, :3] = object_pos + env.scene.env_origins[env_ids]
    root[:, 3:7] = object_quat
    env.scene["object"].write_root_state_to_sim(root, env_ids=env_ids)
    joints = torch.cat([arm_pos, self.open_hand.expand(len(env_ids), -1)], dim=-1)
    env.scene["robot"].write_joint_state_to_sim(
      joints, torch.zeros_like(joints), joint_ids=self.joint_ids, env_ids=env_ids
    )
    self.object_start[env_ids] = object_pos


def object_start(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Env-local object position at the last reset."""
  term = env.event_manager.get_term_cfg("reset_pregrasp").func
  assert isinstance(term, PregraspReset)
  return term.object_start
