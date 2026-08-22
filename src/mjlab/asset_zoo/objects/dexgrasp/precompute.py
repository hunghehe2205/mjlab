"""Precompute DexGrasp Phase 1 object assets.

Exports the meshed primitives (box, cylinder) then samples every object's
affordance cloud: ``trimesh.sample.sample_surface`` of the convex hull (200
points) plus face normals and centroid, mirroring the original RobustDexGrasp
recipe. Writes ``<name>/collision.obj`` (+ lowest_point.txt) and ``<name>.npz``.

Run: ``uv run python -m mjlab.asset_zoo.objects.dexgrasp.precompute``
"""

from __future__ import annotations

import numpy as np
import trimesh

from mjlab.asset_zoo.objects.dexgrasp.object_constants import (
  ASSETS_DIR,
  BOX_HALF_EXTENTS,
  CYLINDER_HALF_HEIGHT,
  CYLINDER_RADIUS,
  NUM_SURFACE_POINTS,
  PHASE1_OBJECTS,
)

SEED = 0


def _primitive_trimeshes() -> dict[str, trimesh.Trimesh]:
  return {
    "box": trimesh.creation.box(extents=[2.0 * h for h in BOX_HALF_EXTENTS]),
    "cylinder": trimesh.creation.cylinder(
      radius=CYLINDER_RADIUS, height=2.0 * CYLINDER_HALF_HEIGHT, sections=32
    ),
  }


def export_primitive_meshes() -> None:
  """Write box/cylinder collision.obj + lowest_point.txt from trimesh."""
  for name, mesh in _primitive_trimeshes().items():
    out = ASSETS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out / "collision.obj"))
    (out / "lowest_point.txt").write_text(f"{float(mesh.vertices[:, 2].min()):.6f}")


def get_object_trimesh(name: str) -> trimesh.Trimesh:
  return trimesh.load_mesh(str(ASSETS_DIR / name / "collision.obj"))


def sample_object(mesh: trimesh.Trimesh, num_points: int = NUM_SURFACE_POINTS) -> dict:
  sampled = trimesh.sample.sample_surface(mesh, num_points, seed=SEED)
  points, face_id = sampled[0], sampled[1]
  normals = np.asarray(mesh.face_normals)[face_id]
  return {
    "points": np.asarray(points, dtype=np.float32),
    "normals": np.asarray(normals, dtype=np.float32),
    "centroid": np.asarray(mesh.centroid, dtype=np.float32),
  }


def precompute_object(name: str) -> None:
  obj = PHASE1_OBJECTS[name]
  # Sample the convex hull: MuJoCo collides against the hull, so the affordance
  # cloud must lie on the surface the fingers actually contact.
  data = sample_object(get_object_trimesh(name).convex_hull)
  np.savez(obj.npz_path, lowest_point=np.float32(obj.lowest_point), **data)


def main() -> None:
  export_primitive_meshes()
  for name in PHASE1_OBJECTS:
    precompute_object(name)
    data = np.load(PHASE1_OBJECTS[name].npz_path)
    print(
      f"{name:18s} points={data['points'].shape} "
      f"centroid={np.round(data['centroid'], 4)} "
      f"lowest={float(data['lowest_point']):.4f}"
    )


if __name__ == "__main__":
  main()
