"""Workstation scene: a table and an arm-mount pedestal, both from the floor."""

import mujoco

TABLE_SIZE = (0.60, 0.55, 0.3855)
PEDESTAL_SIZE = (0.12, 0.12, 0.3655)

TABLE_TOP_Z = 2 * TABLE_SIZE[2]  # 0.771
PEDESTAL_TOP_Z = 2 * PEDESTAL_SIZE[2]  # 0.731
ARM_MOUNT_Z = TABLE_TOP_Z - 0.04  # 0.731

Y_GAP = 0.08
PEDESTAL_CENTER = (0.0, 0.0, PEDESTAL_SIZE[2])
TABLE_CENTER = (0.0, PEDESTAL_SIZE[1] + Y_GAP + TABLE_SIZE[1], TABLE_SIZE[2])


def get_workstation_spec() -> mujoco.MjSpec:
  """Two solid static boxes (table + pedestal) resting on the floor."""
  spec = mujoco.MjSpec()
  table = spec.worldbody.add_body(name="table", pos=TABLE_CENTER)
  table.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_SIZE,
    rgba=(0.6, 0.45, 0.3, 1.0),
  )
  pedestal = spec.worldbody.add_body(name="pedestal", pos=PEDESTAL_CENTER)
  pedestal.add_geom(
    name="pedestal_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=PEDESTAL_SIZE,
    rgba=(0.3, 0.3, 0.33, 1.0),
  )
  return spec
