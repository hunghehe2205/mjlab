"""DexGrasp Phase 1 object assets.

Near-convex objects only. MuJoCo mesh collision uses the convex hull, so shape
fidelity holds for boxes and cans; concave objects plus coacd decomposition are
deferred to Phase 2 (see documents/problems/coacd-convex-decomposition.md).

Every object is a single free body with one mesh geom (box/cylinder are meshed
primitives too) so a ``VariantEntityCfg`` can vary the object per world -- the
variant system only lets mesh geoms differ across variants (see
documents/problems/variant-object-all-mesh.md). A precomputed ``.npz`` next to
each mesh holds the 200-point affordance cloud used by the af_vec observation
and the pre-grasp visible-point raycast; regenerate with ``precompute.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import trimesh

from mjlab.entity import EntityCfg
from mjlab.entity.variants import VariantEntityCfg

OBJECTS_DIR = Path(__file__).parent
ASSETS_DIR = OBJECTS_DIR / "assets"

# Object mass used by the original RobustDexGrasp URDFs (uniform across objects).
OBJECT_MASS = 0.24875
OBJECT_RGBA = (0.85, 0.55, 0.25, 1.0)
NUM_SURFACE_POINTS = 200

# Meshed-primitive dimensions (graspable sizes); precompute.py exports the .obj.
BOX_HALF_EXTENTS = (0.03, 0.03, 0.03)
CYLINDER_RADIUS = 0.03
CYLINDER_HALF_HEIGHT = 0.05

# All Phase 1 objects live under assets/<name>/collision.obj (+ lowest_point.txt).
OBJECT_NAMES = (
  "box",
  "cylinder",
  "potted_meat_can",
  "tomato_soup_can",
  "sugar_box",
)


def get_mesh_object_spec(name: str) -> mujoco.MjSpec:
  """One free body with a single convex-hull mesh geom."""
  spec = mujoco.MjSpec()
  spec.add_material(name="object", rgba=OBJECT_RGBA)
  spec.meshdir = str(ASSETS_DIR / name)
  spec.add_mesh(name="object_mesh", file="collision.obj")
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_joint")
  geom = body.add_geom()
  geom.name = "object_collision"
  geom.type = mujoco.mjtGeom.mjGEOM_MESH
  geom.meshname = "object_mesh"
  geom.mass = OBJECT_MASS
  geom.material = "object"
  return spec


# Primitives know their lowest point analytically; meshes read the YCB txt.
_PRIMITIVE_LOWEST = {"box": -BOX_HALF_EXTENTS[2], "cylinder": -CYLINDER_HALF_HEIGHT}


def _read_lowest_point(name: str) -> float:
  if name in _PRIMITIVE_LOWEST:
    return _PRIMITIVE_LOWEST[name]
  return float((ASSETS_DIR / name / "lowest_point.txt").read_text().strip())


@dataclass(frozen=True)
class DexGraspObject:
  """A Phase 1 grasp object plus its table-placement offset."""

  name: str
  spec_fn: Callable[[], mujoco.MjSpec]

  @property
  def lowest_point(self) -> float:
    """Min-z in object frame; place at table_z - lowest_point.

    Read lazily (not at registry build) so importing this module needs no asset;
    building an env cfg that places the object still reads it.
    """
    return _read_lowest_point(self.name)

  @property
  def npz_path(self) -> Path:
    return ASSETS_DIR / f"{self.name}.npz"

  def load_surface_points(self) -> np.ndarray:
    """(200, 3) affordance cloud in object frame."""
    return np.load(self.npz_path)["points"]

  def load_affordance_mesh(self) -> trimesh.Trimesh:
    """Convex-hull mesh (object frame) for the pre-grasp visibility raycast.

    Returns the hull, not the raw mesh: MuJoCo collides with the hull and the
    affordance cloud is sampled on it, so camera rays must hit the hull too --
    on the raw mesh, concavities make ~6% of rays miss.
    """
    mesh = trimesh.load_mesh(str(ASSETS_DIR / self.name / "collision.obj"))
    if isinstance(mesh, trimesh.Scene):
      mesh = mesh.dump(concatenate=True)
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh.convex_hull

  def get_entity_cfg(self) -> EntityCfg:
    return EntityCfg(spec_fn=self.spec_fn)


def _build_registry() -> dict[str, DexGraspObject]:
  return {
    name: DexGraspObject(name, lambda n=name: get_mesh_object_spec(n))
    for name in OBJECT_NAMES
  }


PHASE1_OBJECTS: dict[str, DexGraspObject] = _build_registry()


def get_phase1_variant_cfg(
  names: tuple[str, ...] | None = None,
  assignment: dict[str, float] | None = None,
) -> VariantEntityCfg:
  """Per-world object variant (evenly weighted by default)."""
  names = names or OBJECT_NAMES
  variants = {n: PHASE1_OBJECTS[n].spec_fn for n in names}
  assignment = assignment or {n: 1.0 for n in names}
  return VariantEntityCfg(variants=variants, assignment=assignment)
