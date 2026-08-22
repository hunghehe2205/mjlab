"""Single-view visibility raycast returns the camera-facing object surface."""

from __future__ import annotations

import numpy as np
import trimesh

from mjlab.tasks.dexgrasp.pregrasp.visibility import visible_points


def test_visible_points_lie_on_the_camera_facing_side():
  box = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
  pts = trimesh.sample.sample_surface(box, 200)[0]
  obj_pos = np.array([0.0, -0.6, 0.8])
  obj_quat = np.array([1.0, 0.0, 0.0, 0.0])
  camera = np.array([0.0, -0.6, 2.0])  # straight above the object

  vis, center = visible_points(box, np.asarray(pts), obj_pos, obj_quat, camera)

  assert vis.shape == (200, 3)
  # Camera is above, so first hits sit on the top half (z >= object center z).
  assert (vis[:, 2] >= obj_pos[2] - 1e-3).mean() > 0.95
  assert center[2] > obj_pos[2]
  # Every visible point stays within the box footprint (on its surface).
  local = vis - obj_pos
  assert np.all(np.abs(local) <= 0.03 + 1e-6)


def test_visible_points_respect_object_rotation():
  box = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
  pts = trimesh.sample.sample_surface(box, 200)[0]
  obj_pos = np.array([0.1, -0.5, 0.8])
  yaw = 0.9
  obj_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
  camera = np.array([0.1, -0.5, 2.0])

  vis, _ = visible_points(box, np.asarray(pts), obj_pos, obj_quat, camera)
  # A yaw about z keeps the top face flush; visible points stay near the top.
  assert (vis[:, 2] >= obj_pos[2] - 1e-3).mean() > 0.95
