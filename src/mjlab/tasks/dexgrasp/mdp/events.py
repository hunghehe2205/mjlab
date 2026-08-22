"""DexGrasp reset event: sampled object pose + analytic-IK pre-grasp (§C).

``ResetGraspPose`` samples a tabletop object pose per env, solves the analytic
UR5e IK pre-grasp for it (RaiSim-style CPU work at reset frequency), and writes
the object root + arm joints. Fingers keep the robot's default pose
(INIT_FINGER_POSE); only the six arm joints are overwritten. IK failures fall
back to a canonical facing pose. Single-object for now; per-world variants land
with the scene wiring (see documents/problems/variant-object-all-mesh.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.tasks.dexgrasp.pregrasp.generator import fallback_arm_qpos, generate_pregrasp
from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics
from mjlab.tasks.dexgrasp.pregrasp.pose_sampler import sample_object_pose

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ResetGraspPose:
  """Reset event: sampled object pose + IK pre-grasp arm pose."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    p = cfg.params
    self._table_top_z = float(p["table_top_z"])
    obj = oc.PHASE1_OBJECTS[p.get("object_name", "potted_meat_can")]
    self._mesh = obj.load_affordance_mesh()
    self._pcd = obj.load_surface_points().astype(np.float64)
    self._lowest = obj.lowest_point
    self._kin = ArmKinematics(mount_pos=(0.0, 0.0, float(p["mount_z"])))
    home = rc.HOME_KEYFRAME.joint_pos or {}
    self._seed = np.array([home[n] for n in rc.ARM_JOINT_NAMES])
    self._rng = np.random.default_rng(env.cfg.seed)
    robot = env.scene["robot"]
    self._arm_ids = [robot.joint_names.index(n) for n in rc.ARM_JOINT_NAMES]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    object_name: str = "potted_meat_can",
    table_top_z: float = 0.771,
    mount_z: float = 0.771,
  ) -> None:
    del object_name, table_top_z, mount_z  # consumed at init
    ids = resolve_env_ids(env, env_ids)
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["object"]

    n = len(ids)
    joint_pos = robot.data.default_joint_pos[ids].clone()
    root_pose = torch.zeros((n, 7), device=device)
    for row, i in enumerate(ids.tolist()):
      del i
      pose = sample_object_pose(self._rng, self._table_top_z, self._lowest)
      arm = generate_pregrasp(
        pose[:3], pose[3:7], self._mesh, self._pcd, self._kin, self._seed
      )
      if arm is None:
        arm = fallback_arm_qpos(pose[:3])
      joint_pos[row, self._arm_ids] = torch.as_tensor(
        arm, dtype=torch.float, device=device
      )
      root_pose[row, :3] = torch.as_tensor(pose[:3], dtype=torch.float, device=device)
      root_pose[row, 3:7] = torch.as_tensor(pose[3:7], dtype=torch.float, device=device)

    root_pose[:, :3] += env.scene.env_origins[ids]
    obj.write_root_link_pose_to_sim(root_pose, env_ids=ids)
    obj.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=ids)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)
