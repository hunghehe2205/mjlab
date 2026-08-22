"""Single-view visibility raycast for the pre-grasp.

Casts one ray from the fixed camera toward each affordance-cloud point and
keeps the first surface hit -- the object surface the camera actually sees.
Ported from RobustDexGrasp's train.py visibility block; the cloud and mesh are
this port's convex hull (see objects.precompute), so every ray from the outside
hits the near face.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _quat2mat(quat: np.ndarray) -> np.ndarray:
  """wxyz quaternion to rotation matrix."""
  w, x, y, z = quat
  n = w * w + x * x + y * y + z * z
  s = 2.0 / n
  xs, ys, zs = x * s, y * s, z * s
  return np.array(
    [
      [1.0 - (y * ys + z * zs), x * ys - w * zs, x * zs + w * ys],
      [x * ys + w * zs, 1.0 - (x * xs + z * zs), y * zs - w * xs],
      [x * zs - w * ys, y * zs + w * xs, 1.0 - (x * xs + y * ys)],
    ]
  )


def visible_points(
  mesh: trimesh.Trimesh,
  aff_pcd: np.ndarray,
  obj_pos: np.ndarray,
  obj_quat: np.ndarray,
  camera_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Camera-visible surface points (N, 3) in the env frame and their center.

  ``aff_pcd`` is the (N, 3) affordance cloud in the object frame; ``mesh`` is the
  matching surface mesh (object frame). Rays that miss fall back to their cloud
  point (they cannot for a convex mesh viewed from outside).
  """
  rot = _quat2mat(np.asarray(obj_quat, dtype=float))
  view_obj = rot.T @ (np.asarray(camera_pos, dtype=float) - obj_pos)

  n = aff_pcd.shape[0]
  origins = np.tile(view_obj, (n, 1))
  directions = aff_pcd - view_obj
  directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

  locs, ray_idx, _ = mesh.ray.intersects_location(
    ray_origins=origins, ray_directions=directions, multiple_hits=False
  )
  hit_obj = aff_pcd.astype(float).copy()
  if len(ray_idx) > 0:
    hit_obj[ray_idx] = locs

  visible_env = (rot @ hit_obj.T).T + obj_pos
  return visible_env, visible_env.mean(axis=0)
