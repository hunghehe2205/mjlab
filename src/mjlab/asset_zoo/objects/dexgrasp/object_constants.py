"""DexGrasp object registry: the RobustDexGrasp 35-object training cohort.

Every object is one free body generated from its upstream URDF by
``convert.py``: ``assets/<name>/object.xml`` with URDF inertia, a non-colliding
visual mesh and convex collision parts. ``ROBUST_DEXGRASP_SOURCES`` maps the
readable local names to the upstream ``new_training_set`` directories.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from mjlab.entity.variants import VariantEntityCfg

OBJECTS_DIR = Path(__file__).parent
ASSETS_DIR = OBJECTS_DIR / "assets"
SOURCE_MESH_FILE = "top_watertight_tiny.obj"

# Tiny damping avoids MJWarp energy gain at high spin (hammer/scissors).
OBJECT_FREE_JOINT_DAMPING = 1e-5
# Slide matches the table; torsion/roll at MuJoCo defaults.
OBJECT_FRICTION = (0.2, 0.005, 0.0001)
OBJECT_RGBA = (0.85, 0.55, 0.25, 1.0)
NUM_SURFACE_POINTS = 200

ROBUST_DEXGRASP_BASELINE_NUM_ENVS = 88

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

ROBUST_DEXGRASP_TRAIN_OBJECTS = tuple(ROBUST_DEXGRASP_SOURCES)
OBJECT_NAMES = ROBUST_DEXGRASP_TRAIN_OBJECTS


@dataclass(frozen=True)
class DexGraspObject:
  """One converted grasp object and its generated files."""

  name: str

  @property
  def asset_dir(self) -> Path:
    return ASSETS_DIR / self.name

  @property
  def source_urdf_path(self) -> Path:
    return self.asset_dir / "source" / f"{ROBUST_DEXGRASP_SOURCES[self.name]}.urdf"

  @property
  def source_mesh_path(self) -> Path:
    return self.asset_dir / "source" / SOURCE_MESH_FILE

  @property
  def xml_path(self) -> Path:
    return self.asset_dir / "object.xml"

  @property
  def manifest_path(self) -> Path:
    return self.asset_dir / "manifest.json"

  @property
  def npz_path(self) -> Path:
    return self.asset_dir / "surface.npz"

  def load_manifest(self) -> dict:
    """Manifest, validated against the source files it was built from.

    Checked on every load: replacing a source mesh while keeping old hulls
    silently corrupts contact geometry, which is far harder to diagnose than a
    load failure.
    """
    manifest = json.loads(self.manifest_path.read_text())
    digests = manifest["source_sha256"]
    for key, path in (("urdf", self.source_urdf_path), ("mesh", self.source_mesh_path)):
      if digests[key] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"Stale assets for {self.name}: {key} changed; rerun convert")
    missing = [
      f for f in manifest["hulls"] if not (self.asset_dir / "collision" / f).is_file()
    ]
    if not manifest["hulls"] or missing:
      raise FileNotFoundError(f"Incomplete collision assets for {self.name}: {missing}")
    return manifest

  @property
  def lowest_point(self) -> float:
    """Min z of the source mesh in the object frame."""
    return float(self.load_manifest()["lowest_point"])

  @property
  def placement_lowest_point(self) -> float:
    """Min z of the collision hulls; place at ``table_z - placement_lowest_point``."""
    return float(self.load_manifest()["placement_lowest_point"])

  def spec_fn(self) -> mujoco.MjSpec:
    """Load ``object.xml`` with absolute mesh paths.

    Absolute because ``VariantEntityCfg`` copies every variant's meshes into one
    template spec, which resolves relative paths against a single directory.
    """
    self.load_manifest()
    spec = mujoco.MjSpec.from_file(str(self.xml_path))
    for mesh in spec.meshes:
      mesh.file = str(self.asset_dir / mesh.file)
    return spec

  def load_surface_points(self) -> np.ndarray:
    """(200, 3) affordance cloud in the object frame."""
    return np.load(self.npz_path)["points"]

  def load_affordance_mesh(self) -> trimesh.Trimesh:
    """The upstream watertight surface (visibility and affordance queries)."""
    mesh = trimesh.load_mesh(str(self.source_mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
      mesh = mesh.dump(concatenate=True)
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


def get_mesh_object_spec(name: str) -> mujoco.MjSpec:
  return DexGraspObject(name).spec_fn()


PHASE1_OBJECTS: dict[str, DexGraspObject] = {
  name: DexGraspObject(name) for name in OBJECT_NAMES
}


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
