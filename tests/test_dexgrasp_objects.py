"""Tests for the RobustDexGrasp converted object assets.

Every object is one free body built from its upstream URDF by ``convert.py``:
``assets/<name>/object.xml`` with URDF inertia, a non-colliding visual mesh and
CoACD convex collision parts. These tests verify the generated assets, not the
conversion itself (which is run once and committed).
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.objects.dexgrasp.convert import simulate_rest
from mjlab.entity import Entity, EntityCfg
from mjlab.entity.variants import build_variant_model

OBJECT_NAMES = tuple(oc.PHASE1_OBJECTS)


def _urdf_inertial(obj: oc.DexGraspObject) -> tuple[float, np.ndarray, np.ndarray]:
  """Mass, COM and inertia eigenvalues parsed straight from the URDF ``top``."""
  root = ET.parse(obj.source_urdf_path).getroot()
  top = next(link for link in root.findall("link") if link.get("name") == "top")
  inr = top.find("inertial")
  assert inr is not None
  origin, mass, inertia = inr.find("origin"), inr.find("mass"), inr.find("inertia")
  assert origin is not None and mass is not None and inertia is not None
  com = np.array([float(v) for v in cast(str, origin.get("xyz")).split()])
  m = float(cast(str, mass.get("value")))
  ixx, ixy, ixz, iyy, iyz, izz = (
    float(cast(str, inertia.get(k))) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
  )
  tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
  return m, com, np.linalg.eigvalsh(tensor)


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_object_spec_compiles(name: str) -> None:
  """One free body ``object`` with a non-colliding visual and manifest hulls."""
  obj = oc.PHASE1_OBJECTS[name]
  entity = Entity(EntityCfg(spec_fn=obj.spec_fn))
  model = entity.spec.compile()
  assert not entity.is_fixed_base
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision") >= 0
  np.testing.assert_allclose(model.dof_damping, oc.OBJECT_FREE_JOINT_DAMPING)
  visual = model.geom("object_visual").id
  assert model.geom_contype[visual] == model.geom_conaffinity[visual] == 0
  assert model.ngeom == 1 + len(obj.load_manifest()["hulls"])


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_mass_and_inertia_match_urdf(name: str) -> None:
  """Compiled inertia equals the URDF ``<inertial>`` parsed independently."""
  obj = oc.PHASE1_OBJECTS[name]
  mass, com, eig = _urdf_inertial(obj)
  model = obj.spec_fn().compile()
  bid = model.body("object").id
  # Manifest keeps full precision; object.xml serializes to ~6 sig figs, so the
  # compiled (as-shipped) values carry that rounding relative to the raw URDF.
  assert obj.load_manifest()["mass"] == pytest.approx(mass)
  assert model.body_mass[bid] == pytest.approx(mass, rel=1e-4)
  np.testing.assert_allclose(model.body_ipos[bid], com, atol=1e-5)
  # Principal moments equal the full-tensor eigenvalues -> rpy/off-diagonals kept.
  np.testing.assert_allclose(
    np.sort(model.body_inertia[bid]), eig, rtol=1e-4, atol=1e-8
  )


def test_stale_or_incomplete_assets_fail(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A missing hull or an edited source mesh must fail loudly, not silently."""
  real_mug = oc.ASSETS_DIR / "mug"
  shutil.copytree(real_mug, tmp_path / "mug")
  monkeypatch.setattr(oc, "ASSETS_DIR", tmp_path)
  obj = oc.DexGraspObject("mug")

  hull = obj.load_manifest()["hulls"][0]
  (tmp_path / "mug" / "collision" / hull).unlink()
  with pytest.raises(FileNotFoundError, match="Incomplete collision"):
    obj.load_manifest()

  shutil.copy(real_mug / "collision" / hull, tmp_path / "mug" / "collision")
  obj.source_mesh_path.write_text("edited source mesh")
  with pytest.raises(ValueError, match="Stale assets"):
    obj.load_manifest()


@pytest.mark.parametrize(
  ("name", "point"),
  (
    ("mug", (-0.00152, 0.001083, 0.038786)),
    ("extra_large_clamp", (0.088205, 0.009987, -0.000276)),
    ("scissors", (-0.019177, 0.036183, -0.003912)),
    ("small_tape", (-0.008582, -0.008459, -0.019241)),
  ),
)
def test_concavities_do_not_create_phantom_contacts(
  name: str, point: tuple[float, float, float]
) -> None:
  """A probe in an empty recess misses the convex parts but hits the old hull."""
  spec = oc.get_mesh_object_spec(name)
  spec.worldbody.add_geom(
    name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.003, 0, 0), pos=point
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 0

  # Positive control: the old single-hull representation would collide here.
  for geom in list(spec.geoms):
    if geom.name.startswith("object_collision"):
      spec.delete(geom)
  visual = spec.geom("object_visual")
  assert visual is not None
  visual.contype = visual.conaffinity = 1
  visual.mass = oc.DexGraspObject(name).load_manifest()["mass"]
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0


def test_phase1_variant_cfg_builds() -> None:
  # All objects are mesh geoms, so one VariantEntityCfg can vary the object
  # per world (only mesh geoms may differ across variants).
  cfg = oc.get_phase1_variant_cfg()
  entity = cfg.build()
  model = entity.spec.compile()
  assert entity.variant_metadata is not None
  assert model.nmesh == sum(
    1 + len(oc.PHASE1_OBJECTS[n].load_manifest()["hulls"]) for n in OBJECT_NAMES
  )


def test_object_variants_keep_each_worlds_inertia_and_parts() -> None:
  names = ("mug", "scissors", "banana")
  entity = oc.get_phase1_variant_cfg(names).build()
  metadata = entity.variant_metadata
  assert metadata is not None
  spec = mujoco.MjSpec()
  spec.attach(entity.spec, prefix="object/", frame=spec.worldbody.add_frame())
  result = build_variant_model(spec, 3, [("object/", metadata)])
  bid = result.mj_model.body("object/object").id
  dataids = result.wp_model.geom_dataid.numpy()
  for world, variant in enumerate(result.world_to_variant["object/"]):
    reference = oc.get_mesh_object_spec(names[variant]).compile()
    for field in ("body_mass", "body_ipos", "body_inertia"):
      np.testing.assert_allclose(
        getattr(result.wp_model, field).numpy()[world, bid],
        getattr(reference, field)[reference.body("object").id],
        rtol=1e-5,
        atol=1e-8,
      )
    assert np.count_nonzero(dataids[world] >= 0) == reference.ngeom


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


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_surface_npz_integrity(name: str) -> None:
  obj = oc.PHASE1_OBJECTS[name]
  data = np.load(obj.npz_path)
  assert data["points"].shape == (oc.NUM_SURFACE_POINTS, 3)
  assert data["normals"].shape == (oc.NUM_SURFACE_POINTS, 3)
  assert data["centroid"].shape == (3,)
  assert float(data["lowest_point"]) == pytest.approx(obj.lowest_point, abs=1e-5)
  points = obj.load_surface_points()
  assert points[:, 2].min() >= obj.lowest_point - 1e-4


@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_surface_points_lie_on_source_mesh(name: str) -> None:
  import trimesh

  mesh = oc.PHASE1_OBJECTS[name].load_affordance_mesh()
  points = oc.PHASE1_OBJECTS[name].load_surface_points()
  _, dist, _ = trimesh.proximity.closest_point(mesh, points)
  assert np.abs(dist).max() < 1e-3


def test_hammer_free_spin_remains_finite_in_mjwarp() -> None:
  """Regress the high-spin path that produced NaNs during teacher training."""
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

  cfg = SimulationCfg(
    mujoco=MujocoCfg(timestep=0.01, gravity=(0.0, 0.0, 0.0), disableflags=("contact",))
  )
  sim = Simulation(
    num_envs=1, cfg=cfg, spec=oc.get_mesh_object_spec("hammer"), device="cpu"
  )
  sim.data.qvel[0, 3:6] = (torch.ones(3) / np.sqrt(3.0)) * 100.0
  for _ in range(50):
    sim.step()
  assert torch.isfinite(sim.data.qpos).all()
  assert torch.isfinite(sim.data.qvel).all()
  assert torch.linalg.vector_norm(sim.data.qvel[0, 3:6]) < 100.0


@pytest.mark.slow
@pytest.mark.parametrize("name", OBJECT_NAMES)
def test_object_settles_stably_on_table(name: str) -> None:
  """Placed on the table under the training solver, an object comes to rest.

  Objects keep their URDF-native orientation, which is not always table-stable,
  so several settle into a tilted pose or creep a little. We do not require a
  particular orientation -- only that the object stays on the table and becomes
  quasi-static (no explosion, fly-off, sink-through or perpetual sliding).
  """
  m = simulate_rest(name)
  assert m["finite"], f"{name}: non-finite state"
  assert m["xy_drift"] < 0.02, f"{name}: slid {m['xy_drift'] * 1e3:.1f}mm"
  assert -0.01 < m["z_sink"] < 0.02, f"{name}: z_sink {m['z_sink'] * 1e3:.1f}mm"
  assert m["settle_pos"] < 0.005, f"{name}: still moving {m['settle_pos'] * 1e3:.2f}mm"
  assert m["settle_rot"] < 1.0, f"{name}: still rotating {m['settle_rot']:.2f}deg"
