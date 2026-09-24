"""Primitive rigid bodies with explicitly configured contact physics."""

from functools import partial

import mujoco

from mjlab.entity import EntityCfg


def box_spec(size: tuple[float, float, float], mass: float) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="free_joint")
  body.add_geom(
    name="collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=size,
    mass=mass,
    friction=(1.0, 0.005, 0.0001),
    condim=4,
    rgba=(0.9, 0.35, 0.1, 1.0),
  )
  return spec


def get_box_cfg(
  size: tuple[float, float, float] = (0.03, 0.03, 0.06), mass: float = 0.08
) -> EntityCfg:
  """Create a floating box; size denotes half extents in meters."""
  if any(s <= 0 for s in size) or mass <= 0:
    raise ValueError("Box half extents and mass must be positive.")
  return EntityCfg(spec_fn=partial(box_spec, size, mass))
