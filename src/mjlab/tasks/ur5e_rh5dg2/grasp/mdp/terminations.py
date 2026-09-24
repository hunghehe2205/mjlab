"""Grasp episode failures and the lift-test success check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.scenes.workstation import TABLE_CENTER, TABLE_SIZE, TABLE_TOP_Z
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import LIFT_HEIGHT
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import lift_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def dropped(env: ManagerBasedRlEnv) -> torch.Tensor:
  pos = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  return (
    (pos[:, 2] < TABLE_TOP_Z - 0.05)
    | ((pos[:, 0] - TABLE_CENTER[0]).abs() > TABLE_SIZE[0])
    | ((pos[:, 1] - TABLE_CENTER[1]).abs() > TABLE_SIZE[1])
  )


def hand_below_table(env: ManagerBasedRlEnv, links: SceneEntityCfg) -> torch.Tensor:
  """Any hand link origin under the tabletop, as in the reference."""
  z = env.scene["robot"].data.body_link_pos_w[:, links.body_ids, 2]
  return (z - env.scene.env_origins[:, None, 2] < TABLE_TOP_Z).any(dim=-1)


def lifted(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reference lift-test success: the object rose more than LIFT_HEIGHT."""
  return (lift_height(env) > LIFT_HEIGHT).float()
