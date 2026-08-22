"""End-to-end pre-grasp generation on a real Phase 1 object.

Exercises visibility -> palm-roll sampling -> IK -> scoring together and checks
the chosen grasp-center sits ~0.25 m in front of the object toward the camera.
"""

from __future__ import annotations

import numpy as np
import trimesh

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.tasks.dexgrasp.pregrasp.generator import CAMERA_POSITION, generate_pregrasp
from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics
from mjlab.tasks.dexgrasp.pregrasp.pose_sampler import sample_object_pose

TABLE_TOP_Z = 0.771


def _load_mesh(name: str) -> trimesh.Trimesh:
  mesh = trimesh.load_mesh(str(oc.ASSETS_DIR / name / "collision.obj"))
  assert isinstance(mesh, trimesh.Trimesh)
  return mesh


def test_generate_pregrasp_places_grasp_center_toward_camera():
  name = "potted_meat_can"
  obj = oc.PHASE1_OBJECTS[name]
  mesh = _load_mesh(name)
  pcd = obj.load_surface_points()
  kin = ArmKinematics(mount_pos=(0.0, 0.0, TABLE_TOP_Z))
  home = rc.HOME_KEYFRAME.joint_pos or {}
  seed = np.array([home[n] for n in rc.ARM_JOINT_NAMES])

  rng = np.random.default_rng(0)
  ok = 0
  for _ in range(20):
    pose = sample_object_pose(rng, TABLE_TOP_Z, obj.lowest_point)
    q = generate_pregrasp(pose[:3], pose[3:7], mesh, pcd, kin, seed_qpos=seed)
    if q is None:
      continue
    ok += 1
    gc = kin.fk_grasp_center_env(q)[:3, 3]
    obj_to_cam = np.linalg.norm(pose[:3] - CAMERA_POSITION)
    assert np.linalg.norm(gc - CAMERA_POSITION) < obj_to_cam
    assert 0.1 < np.linalg.norm(gc - pose[:3]) < 0.45
  assert ok >= 10
