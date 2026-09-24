"""Teacher-local fingertip collision approximation for Warp contact dynamics."""

import mujoco

from mjlab.asset_zoo.robots.ur5e_rh5dg2.ur5e_rh5dg2_constants import get_spec


def teacher_robot_spec() -> mujoco.MjSpec:
  """Round fingertip cylinders, retaining their poses and original bounds.

  Warp supports only one contact for cylinder-box pairs. Rounded pads avoid
  the flat cylinder edges that made the scripted grasp backend-sensitive.
  Each replacement fits inside the original cylinder. Visuals, explicit body
  inertias, joint limits and the shared robot asset remain unchanged.
  """
  spec = get_spec()
  for geom in spec.geoms:
    if (
      "_dip_collision_" not in geom.name or geom.type != mujoco.mjtGeom.mjGEOM_CYLINDER
    ):
      continue
    radius = min(geom.size[0], geom.size[1])
    half_length = geom.size[1] - radius
    geom.type = (
      mujoco.mjtGeom.mjGEOM_CAPSULE if half_length > 0 else mujoco.mjtGeom.mjGEOM_SPHERE
    )
    geom.size = (radius, half_length, 0.0)
  return spec
