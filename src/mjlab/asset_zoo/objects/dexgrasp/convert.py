"""Build DexGrasp object assets from the RobustDexGrasp ``new_training_set``.

Each upstream object is a URDF with a dummy ``bottom`` link and a ``top`` link
holding the real mesh, joined by a near-locked hinge (a RaiSim floating-base
trick). We attach ``top`` into a fresh mjSpec as one free body, keep the URDF
inertia, make the source mesh a non-colliding visual, add CoACD convex parts
for contact, and write ``object.xml`` next to the hulls. Output is committed;
this runs once per upstream change::

  uv run --group assets python -m mjlab.asset_zoo.objects.dexgrasp.convert \
      --rdg-dir /path/to/RobustDexGrasp/rsc/new_training_set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from mjlab.asset_zoo.objects.dexgrasp.object_constants import (
  ASSETS_DIR,
  NUM_SURFACE_POINTS,
  OBJECT_FREE_JOINT_DAMPING,
  OBJECT_FRICTION,
  OBJECT_NAMES,
  OBJECT_RGBA,
  ROBUST_DEXGRASP_SOURCES,
  SOURCE_MESH_FILE,
  DexGraspObject,
)

MANIFEST_VERSION = 2
SOURCE_LOWEST_FILE = "lowest_point_new.txt"
SAMPLE_SEED = 0
# Near-convex meshes (cans, boxes) keep one hull: decomposing them only yields
# a lumpy base that rocks on the table.
CONVEX_RATIO = 0.90
COACD_PARAMS = {
  "threshold": 0.05,
  "max_convex_hull": 16,
  "resolution": 2000,
  "max_ch_vertex": 128,
  "seed": 0,
}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_trimesh(path: Path) -> trimesh.Trimesh:
  mesh = trimesh.load_mesh(str(path), process=False)
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
  if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(f"Expected one mesh in {path}, got {type(mesh).__name__}")
  return mesh


def upstream_commit(rdg_dir: Path) -> str | None:
  try:
    out = subprocess.run(
      ["git", "-C", str(rdg_dir), "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return out.stdout.strip()


def sync_source(name: str, rdg_dir: Path) -> Path:
  """Copy the upstream URDF, mesh, and lowest point verbatim into ``source/``."""
  src = ROBUST_DEXGRASP_SOURCES[name]
  src_dir = rdg_dir / src
  out = ASSETS_DIR / name / "source"
  out.mkdir(parents=True, exist_ok=True)
  for file in (f"{src}.urdf", SOURCE_MESH_FILE, SOURCE_LOWEST_FILE):
    shutil.copyfile(src_dir / file, out / file)
  return out


def decompose(mesh: trimesh.Trimesh) -> tuple[list[trimesh.Trimesh], float]:
  """Return convex parts and the volume ratio that picked the strategy."""
  hull = mesh.convex_hull
  ratio = float(mesh.volume / hull.volume) if mesh.is_watertight else 0.0
  if ratio >= CONVEX_RATIO:
    return [hull], ratio
  import coacd  # Only needed at build time (``uv run --group assets``).

  # Explicit kwargs (not ``**COACD_PARAMS``): coacd's stub types are strict and
  # dict unpacking loses the per-key int/float distinction.
  result = coacd.run_coacd(
    coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces),
    threshold=float(COACD_PARAMS["threshold"]),
    max_convex_hull=int(COACD_PARAMS["max_convex_hull"]),
    resolution=int(COACD_PARAMS["resolution"]),
    max_ch_vertex=int(COACD_PARAMS["max_ch_vertex"]),
    seed=int(COACD_PARAMS["seed"]),
  )
  parts = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in result]
  if not parts:
    raise ValueError("CoACD returned no convex parts")
  return parts, ratio


def build_spec(name: str, hull_files: list[str]) -> mujoco.MjSpec:
  """One free body from the URDF ``top`` link plus mesh collision parts."""
  obj = DexGraspObject(name)
  urdf = mujoco.MjSpec.from_file(str(obj.source_urdf_path))
  spec = mujoco.MjSpec()
  spec.modelname = name
  spec.add_material(name="object", rgba=OBJECT_RGBA)

  body = spec.worldbody.add_frame().attach_body(urdf.body("top"), "", "")
  body.name = "object"
  for joint in list(body.joints):
    spec.delete(joint)
  joint = body.add_freejoint(name="object_joint")
  joint.damping = np.full(3, OBJECT_FREE_JOINT_DAMPING)

  (visual,) = body.geoms
  visual.name = "object_visual"
  visual.contype = visual.conaffinity = 0
  visual.mass = 0.0
  visual.material = "object"
  mesh = spec.mesh(visual.meshname)
  assert mesh is not None
  mesh.name = visual.meshname = "object_visual_mesh"
  # Absolute so compile/to_xml can read the files; build_object rewrites the
  # emitted XML back to asset-relative paths.
  mesh.file = str(obj.asset_dir / "source" / SOURCE_MESH_FILE)

  for index, file in enumerate(hull_files):
    mesh_name = f"object_collision_mesh_{index:03d}"
    spec.add_mesh(name=mesh_name, file=str(obj.asset_dir / "collision" / file))
    geom = body.add_geom(
      name="object_collision" if index == 0 else f"object_collision_{index:03d}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      condim=6,
      friction=OBJECT_FRICTION,
      mass=0.0,
      material="object",
    )
    geom.contype = geom.conaffinity = 1
  return spec


def build_object(name: str, commit: str | None) -> dict:
  obj = DexGraspObject(name)
  mesh = _as_trimesh(obj.source_mesh_path)
  parts, ratio = decompose(mesh)

  collision_dir = obj.asset_dir / "collision"
  if collision_dir.exists():
    shutil.rmtree(collision_dir)
  collision_dir.mkdir()
  hull_files = [f"hull_{i:03d}.obj" for i in range(len(parts))]
  for part, file in zip(parts, hull_files, strict=True):
    part.export(collision_dir / file)
  hull_lowest = min(float(part.vertices[:, 2].min()) for part in parts)

  spec = build_spec(name, hull_files)
  model = spec.compile()
  xml = spec.to_xml().replace(f"{obj.asset_dir}/", "")
  # to_xml round-trips the URDF's main default as an invalid empty nested
  # <default/>; strip it so object.xml reloads.
  xml = re.sub(r"\s*<default>\s*<default\s*/>\s*</default>", "", xml)
  obj.xml_path.write_text(xml)

  sampled = trimesh.sample.sample_surface(mesh, NUM_SURFACE_POINTS, seed=SAMPLE_SEED)
  points, face_id = sampled[0], sampled[1]
  np.savez(
    obj.npz_path,
    points=np.asarray(points, dtype=np.float32),
    normals=np.asarray(mesh.face_normals[face_id], dtype=np.float32),
    centroid=np.asarray(mesh.centroid, dtype=np.float32),
    lowest_point=np.float32(mesh.vertices[:, 2].min()),
    placement_lowest_point=np.float32(hull_lowest),
  )

  manifest = {
    "version": MANIFEST_VERSION,
    "upstream": {
      "repo": "https://github.com/zdchan/RobustDexGrasp",
      "path": f"rsc/new_training_set/{ROBUST_DEXGRASP_SOURCES[name]}",
      "commit": commit,
    },
    "source_sha256": {
      "urdf": _sha256(obj.source_urdf_path),
      "mesh": _sha256(obj.source_mesh_path),
    },
    "coacd": None if len(parts) == 1 and ratio >= CONVEX_RATIO else COACD_PARAMS,
    "convex_ratio": ratio,
    "hulls": hull_files,
    "mass": float(model.body_mass[model.body("object").id]),
    "lowest_point": float(mesh.vertices[:, 2].min()),
    "upstream_lowest_point": float(
      (obj.asset_dir / "source" / SOURCE_LOWEST_FILE).read_text()
    ),
    "placement_lowest_point": hull_lowest,
  }
  obj.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


# Mirrors the DexGrasp training ``SimulationCfg`` and table (dexgrasp_env_cfg).
TABLE_TOP_Z = 0.771
TABLE_HALF = (0.60, 0.55, TABLE_TOP_Z / 2)
TABLE_CENTER = (0.0, -0.75, TABLE_TOP_Z / 2)
TABLE_FRICTION = (0.2, 0.005, 0.0001)
PLACEMENT_CLEARANCE = 0.002


def simulate_rest(
  name: str, seconds: float = 4.0, window: float = 0.5, device: str = "cpu"
) -> dict[str, float]:
  """Drop the object on a table under the training solver; report drift.

  Objects keep their URDF-native orientation, which is not always the
  table-stable pose, so several settle into a tilted rest or creep slowly.
  ``settle_pos``/``settle_rot`` measure motion over a trailing ``window`` after
  ``seconds`` of settling -- the honest "has it come to rest" signal.
  """
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

  obj = DexGraspObject(name)
  spec = obj.spec_fn()
  spec.add_material(name="table", rgba=(0.55, 0.40, 0.28, 1.0))
  spec.worldbody.add_body(name="arena").add_geom(
    name="table",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_HALF,
    pos=TABLE_CENTER,
    friction=TABLE_FRICTION,
    material="table",
  )
  cfg = SimulationCfg(
    nconmax=150,
    njmax=1500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      impratio=1.0,
      cone="pyramidal",
    ),
  )
  sim = Simulation(num_envs=1, cfg=cfg, spec=spec, device=device)
  start = np.array(
    [0.0, -0.6, TABLE_TOP_Z - obj.placement_lowest_point + PLACEMENT_CLEARANCE]
  )
  sim.data.qpos[0, :3] = sim.data.qpos.new_tensor(start)
  sim.data.qpos[0, 3:7] = sim.data.qpos.new_tensor([1.0, 0.0, 0.0, 0.0])
  sim.data.qvel[0, :] = 0.0
  for _ in range(int(round(seconds / cfg.mujoco.timestep))):
    sim.step()
  settled = sim.data.qpos[0].cpu().numpy().copy()
  for _ in range(int(round(window / cfg.mujoco.timestep))):
    sim.step()

  qpos = sim.data.qpos[0].cpu().numpy()
  qvel = sim.data.qvel[0].cpu().numpy()
  x, y = float(qpos[4]), float(qpos[5])
  # Angle between the body z-axis and world z, from the rotation matrix.
  cos_tilt = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
  # Motion over the trailing window: quasi-static drift, orientation-agnostic.
  qa = settled[3:7] / np.linalg.norm(settled[3:7])
  qb = qpos[3:7] / np.linalg.norm(qpos[3:7])
  settle_rot = float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(qa @ qb))))))
  return {
    "xy_drift": float(np.linalg.norm(qpos[:2] - start[:2])),
    "z_sink": float(start[2] - qpos[2]),
    "tilt_deg": float(np.degrees(np.arccos(cos_tilt))),
    "speed": float(np.linalg.norm(qvel[:3])),
    "ang_speed": float(np.linalg.norm(qvel[3:6])),
    "settle_pos": float(np.linalg.norm(qpos[:3] - settled[:3])),
    "settle_rot": settle_rot,
    "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--rdg-dir", type=Path, help="new_training_set checkout")
  parser.add_argument("names", nargs="*", default=list(OBJECT_NAMES))
  parser.add_argument("--check", action="store_true", help="Rest report only")
  args = parser.parse_args()
  if args.check:
    for name in args.names:
      m = simulate_rest(name)
      print(
        f"{name:22s} xy={m['xy_drift'] * 1e3:6.2f}mm sink={m['z_sink'] * 1e3:6.2f}mm "
        f"tilt={m['tilt_deg']:5.2f}deg dpos={m['settle_pos'] * 1e3:6.3f}mm "
        f"drot={m['settle_rot']:5.2f}deg"
      )
    return
  if args.rdg_dir is None:
    parser.error("--rdg-dir is required to build")
  commit = upstream_commit(args.rdg_dir)
  for name in args.names:
    sync_source(name, args.rdg_dir)
    manifest = build_object(name, commit)
    print(
      f"{name:22s} hulls={len(manifest['hulls']):2d} "
      f"convex_ratio={manifest['convex_ratio']:.2f} mass={manifest['mass']:.4f}"
    )


if __name__ == "__main__":
  main()
