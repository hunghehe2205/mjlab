"""DexGrasp object registry and the RobustDexGrasp 35-object training cohort.

All objects use one free body and one mesh collision geom so
``VariantEntityCfg`` can select a geometry per parallel world. Each mesh has a
200-point affordance cloud used by the teacher observations and pre-grasp
reset. The cohort names are readable local names; ``ROBUST_DEXGRASP_SOURCES``
records their exact names in the released RobustDexGrasp ``new_training_set``.
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

# Object mass used by the original RobustDexGrasp URDFs. Uniform except for five
# objects whose URDFs carry a lighter identified mass; matching them matters
# because scissors/small_block are ~3x lighter and are oversampled in training.
OBJECT_MASS = 0.24875
OBJECT_MASS_OVERRIDES = {
  "scissors": 0.08626186684547055,
  "small_block": 0.08340712944163264,
  "large_clamp": 0.1695701014449372,
  "off_water_body": 0.2049091757100137,
  "extra_large_clamp": 0.2291035804273685,
}
# A tiny free-joint damping prevents MJWarp's explicit free-body angular dynamics
# from gaining energy at high spin (notably for hammer/scissors at a 10 ms step).
# At normal grasp speeds its time constant is long enough to be negligible.
OBJECT_FREE_JOINT_DAMPING = 1e-5
# Matches the table so their elementwise-max pair friction is the reference's
# 0.2; against the 0.8 hand geoms the max still yields the reference's 0.8.
OBJECT_FRICTION = 0.2
OBJECT_RGBA = (0.85, 0.55, 0.25, 1.0)
NUM_SURFACE_POINTS = 200

# Meshed-primitive dimensions (graspable sizes); precompute.py exports the .obj.
BOX_HALF_EXTENTS = (0.03, 0.03, 0.03)
CYLINDER_RADIUS = 0.03
CYLINDER_HALF_HEIGHT = 0.05

# Kept for fast scene/debugging outside the baseline training cohort.
DEBUG_OBJECT_NAMES = ("box", "cylinder")
ROBUST_DEXGRASP_BASELINE_NUM_ENVS = 88

# Exact RobustDexGrasp ``new_training_set`` cohort. The original teacher
# oversamples a few difficult objects; see ``baseline_assignment`` below.
ROBUST_DEXGRASP_TRAIN_OBJECTS = (
  "master_chef_can",
  "cracker_box",
  "sugar_box",
  "tomato_soup_can",
  "mustard_bottle",
  "tuna_fish_can",
  "pudding_box",
  "gelatin_box",
  "box",
  "potted_meat_can",
  "banana",
  "pitcher_base",
  "bleach_cleanser",
  "mug",
  "power_drill",
  "wood_block",
  "scissors",
  "large_clamp",
  "extra_large_clamp",
  "foam_brick",
  "big_tape",
  "blue_pitcher",
  "brush_functional",
  "car_down",
  "cracker_box_oriented",
  "fan_small_head",
  "gun_functional",
  "hammer",
  "loopy_head_side",
  "mouse",
  "off_water_body",
  "small_block",
  "small_tape",
  "solder_iron_head",
  "sugar_box_oriented",
  "wood_block_oriented",
)

# ``box`` is a local meshed primitive, not an object from the source cohort.
ROBUST_DEXGRASP_TRAIN_OBJECTS = tuple(
  name for name in ROBUST_DEXGRASP_TRAIN_OBJECTS if name != "box"
)

ROBUST_DEXGRASP_SOURCES = {
  "master_chef_can": "002_master_chef_can",
  "cracker_box": "003_cracker_box",
  "sugar_box": "004_sugar_box",
  "tomato_soup_can": "005_tomato_soup_can",
  "mustard_bottle": "006_mustard_bottle",
  "tuna_fish_can": "007_tuna_fish_can",
  "pudding_box": "008_pudding_box",
  "gelatin_box": "009_gelatin_box",
  "potted_meat_can": "010_potted_meat_can",
  "banana": "011_banana",
  "pitcher_base": "019_pitcher_base",
  "bleach_cleanser": "021_bleach_cleanser",
  "mug": "025_mug",
  "power_drill": "035_power_drill",
  "wood_block": "036_wood_block",
  "scissors": "037_scissors",
  "large_clamp": "051_large_clamp",
  "extra_large_clamp": "052_extra_large_clamp",
  "foam_brick": "061_foam_brick",
  "big_tape": "big_tape",
  "blue_pitcher": "blue_pitcher",
  "brush_functional": "brush_functional",
  "car_down": "car_down",
  "cracker_box_oriented": "cracker_box_oriented",
  "fan_small_head": "fan_small_head",
  "gun_functional": "gun_functional",
  "hammer": "hammer",
  "loopy_head_side": "loopy_head_side",
  "mouse": "mouse",
  "off_water_body": "off_water_body",
  "small_block": "small_block",
  "small_tape": "small_tape",
  "solder_iron_head": "solder_iron_head",
  "sugar_box_oriented": "suger_box_oriented",
  "wood_block_oriented": "wood_block_oriented",
}

OBJECT_NAMES = DEBUG_OBJECT_NAMES + ROBUST_DEXGRASP_TRAIN_OBJECTS


def get_mesh_object_spec(name: str, fixed: bool = False) -> mujoco.MjSpec:
  """One mesh collision body."""
  spec = mujoco.MjSpec()
  spec.add_material(name="object", rgba=OBJECT_RGBA)
  spec.meshdir = str(ASSETS_DIR / name)
  spec.add_mesh(name="object_mesh", file="collision.obj")
  body = spec.worldbody.add_body(name="object")
  if not fixed:
    joint = body.add_freejoint(name="object_joint")
    joint.damping = np.full(3, OBJECT_FREE_JOINT_DAMPING)
  geom = body.add_geom()
  geom.name = "object_collision"
  geom.type = mujoco.mjtGeom.mjGEOM_MESH
  geom.meshname = "object_mesh"
  geom.mass = OBJECT_MASS_OVERRIDES.get(name, OBJECT_MASS)
  geom.friction[0] = OBJECT_FRICTION
  geom.material = "object"
  return spec


# Primitives know their lowest point analytically; meshes read the YCB txt.
_PRIMITIVE_LOWEST = {"box": -BOX_HALF_EXTENTS[2], "cylinder": -CYLINDER_HALF_HEIGHT}


def _read_lowest_point(name: str) -> float:
  if name in _PRIMITIVE_LOWEST:
    return _PRIMITIVE_LOWEST[name]
  path = ASSETS_DIR / name / "lowest_point.txt"
  if path.exists():
    return float(path.read_text().strip())
  mesh = trimesh.load_mesh(str(ASSETS_DIR / name / "collision.obj"))
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
  assert isinstance(mesh, trimesh.Trimesh)
  return float(mesh.vertices[:, 2].min())


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


def get_robustdexgrasp_variant_cfg() -> VariantEntityCfg:
  """Build the 35-object teacher cohort with source-baseline oversampling."""
  assignment = {name: 1.0 for name in ROBUST_DEXGRASP_TRAIN_OBJECTS}
  for name in (
    "scissors",
    "off_water_body",
    "pitcher_base",
    "banana",
    "mouse",
    "hammer",
    "small_block",
  ):
    assignment[name] += 1.0
  for name in ("scissors", "off_water_body"):
    assignment[name] += 1.0
  return get_phase1_variant_cfg(ROBUST_DEXGRASP_TRAIN_OBJECTS, assignment)
