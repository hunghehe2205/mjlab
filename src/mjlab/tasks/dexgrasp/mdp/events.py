"""DexGrasp reset event: sampled object pose + analytic-IK pre-grasp."""

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
from mjlab.tasks.dexgrasp.pregrasp.self_collision import ArmHandSelfCollisionProbe

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ResetGraspPose"]

MAX_PREGRASP_ATTEMPTS = 8


class ResetGraspPose:
  """Reset event: sampled object pose + IK pre-grasp arm pose."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    p = cfg.params
    self._table_top_z = float(p["table_top_z"])
    names = tuple(p.get("object_names", (p.get("object_name", "potted_meat_can"),)))
    objects = [oc.PHASE1_OBJECTS[name] for name in names]
    self._meshes = [obj.load_affordance_mesh() for obj in objects]
    self._pcds = [obj.load_surface_points().astype(np.float64) for obj in objects]
    self._lowest = np.asarray([obj.lowest_point for obj in objects])
    self._clearance = float(p.get("object_clearance", 0.002))
    self._kin = ArmKinematics(mount_pos=(0.0, 0.0, float(p["mount_z"])))
    home = rc.HOME_KEYFRAME.joint_pos or {}
    self._seed = np.array([home[n] for n in rc.ARM_JOINT_NAMES])
    self._max_attempts = int(p.get("max_attempts", MAX_PREGRASP_ATTEMPTS))
    self._probe = ArmHandSelfCollisionProbe()
    if not self._probe.is_valid_pregrasp(self._seed):
      raise ValueError("UR5e RH5-DG2 home arm pose is not a valid pre-grasp.")
    # Seed from mjlab's global numpy RNG (seed_rng'd from the resolved env seed)
    # rather than raw cfg.seed, which is None -> OS entropy when unset.
    self._rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
    robot = env.scene["robot"]
    self._arm_ids = [robot.joint_names.index(n) for n in rc.ARM_JOINT_NAMES]
    self._variant_ids = env.sim.world_to_variant.get("object")
    if self._variant_ids is None:
      self._variant_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    object_names: tuple[str, ...] | None = None,
    table_top_z: float = 0.771,
    mount_z: float = 0.771,
    object_clearance: float = 0.002,
  ) -> None:
    del object_names, table_top_z, mount_z, object_clearance
    ids = resolve_env_ids(env, env_ids)
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["object"]
    variant_ids = self._variant_ids
    assert variant_ids is not None

    n = len(ids)
    # Accumulate per-env numpy, then transfer to the device once (not per env).
    arms = np.empty((n, 6))
    poses = np.empty((n, 7))
    for row in range(n):
      variant = int(variant_ids[ids[row]].item())
      pose, arm = self._sample_collision_free_pregrasp(variant)
      arms[row] = arm
      poses[row] = pose

    joint_pos = robot.data.default_joint_pos[ids]  # advanced-index copy
    joint_pos[:, self._arm_ids] = torch.as_tensor(
      arms, dtype=torch.float, device=device
    )
    root_pose = torch.as_tensor(poses, dtype=torch.float, device=device)
    root_pose[:, :3] += env.scene.env_origins[ids]

    if obj.is_fixed_base:
      obj.write_mocap_pose_to_sim(root_pose, env_ids=ids)
    else:
      obj.write_root_link_pose_to_sim(root_pose, env_ids=ids)
      obj.write_root_link_velocity_to_sim(
        torch.zeros((n, 6), device=device), env_ids=ids
      )
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)

  def _sample_collision_free_pregrasp(
    self, variant: int
  ) -> tuple[np.ndarray, np.ndarray]:
    pose = self._sample_object_pose(variant)
    for _ in range(self._max_attempts):
      arm = generate_pregrasp(
        pose[:3],
        pose[3:7],
        self._meshes[variant],
        self._pcds[variant],
        self._kin,
        self._seed,
      )
      if arm is not None and self._probe.is_valid_pregrasp(arm):
        return pose, arm
      pose = self._sample_object_pose(variant)
    fallback = fallback_arm_qpos(pose[:3])
    return pose, fallback if self._probe.is_valid_pregrasp(fallback) else self._seed

  def _sample_object_pose(self, variant: int) -> np.ndarray:
    return sample_object_pose(
      self._rng,
      self._table_top_z,
      self._lowest[variant],
      clearance=self._clearance,
    )
