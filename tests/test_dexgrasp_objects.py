"""Tests for DexGrasp Phase 1 object assets."""

from typing import cast

import mujoco
import numpy as np
import pytest
import trimesh

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.objects.dexgrasp.precompute import get_object_trimesh
from mjlab.entity import Entity

OBJECT_NAMES = tuple(oc.PHASE1_OBJECTS)


@pytest.mark.parametrize("name", ("potted_meat_can", "sugar_box", "box"))
def test_load_affordance_mesh_matches_collision_hull(name: str) -> None:
  """Visibility raycast must hit the same hull MuJoCo collides with, and where
  the affordance cloud was sampled -- else camera rays miss on raw concavities."""
  mesh = oc.PHASE1_OBJECTS[name].load_affordance_mesh()
  assert mesh.is_convex


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_object_spec_compiles(name: str) -> None:
  obj = oc.PHASE1_OBJECTS[name]
  entity = Entity(obj.get_entity_cfg())
  model = entity.spec.compile()
  # Single free body with a collision geom.
  assert not entity.is_fixed_base
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision") >= 0
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
  assert model.body_mass[bid] == pytest.approx(oc.OBJECT_MASS)


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_npz_integrity(name: str) -> None:
  obj = oc.PHASE1_OBJECTS[name]
  assert obj.npz_path.exists(), f"missing {obj.npz_path}; run precompute.py"
  data = np.load(obj.npz_path)
  assert data["points"].shape == (oc.NUM_SURFACE_POINTS, 3)
  assert data["normals"].shape == (oc.NUM_SURFACE_POINTS, 3)
  assert data["centroid"].shape == (3,)
  assert float(data["lowest_point"]) == pytest.approx(obj.lowest_point, abs=1e-5)


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_surface_points_above_lowest(name: str) -> None:
  # Every surface sample sits at or above the mesh lowest point.
  obj = oc.PHASE1_OBJECTS[name]
  points = obj.load_surface_points()
  assert points[:, 2].min() >= obj.lowest_point - 1e-4


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_surface_points_on_collision_hull(name: str) -> None:
  # Shipped cloud must lie on the current convex-hull collision surface; catches
  # a stale npz after a mesh edit without re-running precompute.
  hull = get_object_trimesh(name).convex_hull
  points = oc.PHASE1_OBJECTS[name].load_surface_points()
  dist = trimesh.proximity.signed_distance(hull, points)
  assert np.abs(dist).max() < 1e-3


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_trimesh_matches_lowest_point(name: str) -> None:
  # The precompute trimesh and the registry agree on the mesh bottom.
  mesh = get_object_trimesh(name)
  assert float(mesh.vertices[:, 2].min()) == pytest.approx(
    oc.PHASE1_OBJECTS[name].lowest_point, abs=1e-3
  )


def test_phase1_variant_cfg_builds() -> None:
  # All objects are mesh geoms, so one VariantEntityCfg can vary the object
  # per world (only mesh geoms may differ across variants).
  cfg = oc.get_phase1_variant_cfg()
  entity = cfg.build()
  model = entity.spec.compile()
  assert entity.variant_metadata is not None
  assert model.nmesh == len(OBJECT_NAMES)


def test_robustdexgrasp_training_cohort_matches_baseline() -> None:
  names = oc.ROBUST_DEXGRASP_TRAIN_OBJECTS
  cfg = oc.get_robustdexgrasp_variant_cfg()
  assert len(names) == 35
  assert oc.ROBUST_DEXGRASP_BASELINE_NUM_ENVS == 88
  assert set(names) == set(oc.ROBUST_DEXGRASP_SOURCES)
  assert isinstance(cfg.assignment, dict)
  assignment = cast(dict[str, float], cfg.assignment)
  assert sum(assignment.values()) == 44.0
  assert assignment["scissors"] == 3.0
  assert assignment["off_water_body"] == 3.0
  assert assignment["banana"] == 2.0
