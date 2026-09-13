import math

from mjlab.asset_zoo.scenes import workstation as ws


def test_dimensions_and_layout():
  assert math.isclose(ws.TABLE_TOP_Z, 0.771)
  assert math.isclose(ws.PEDESTAL_TOP_Z, 0.731)
  assert math.isclose(ws.ARM_MOUNT_Z, ws.TABLE_TOP_Z - 0.04)
  # Pedestal fully within table width in x.
  assert ws.PEDESTAL_SIZE[0] <= ws.TABLE_SIZE[0]
  # 8 cm gap along y between pedestal front and table near face.
  pedestal_front = ws.PEDESTAL_CENTER[1] + ws.PEDESTAL_SIZE[1]
  table_near = ws.TABLE_CENTER[1] - ws.TABLE_SIZE[1]
  assert math.isclose(table_near - pedestal_front, ws.Y_GAP)


def test_scene_compiles_with_two_static_boxes():
  model = ws.get_workstation_spec().compile()
  assert model.ngeom == 2  # table + pedestal
  assert model.nq == 0  # static, no freejoints
