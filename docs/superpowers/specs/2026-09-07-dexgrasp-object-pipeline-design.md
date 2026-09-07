# DexGrasp object pipeline (URDF → mjSpec → MJCF)

Date: 2026-09-07. Status: approved in chat.

## Goal

Rebuild the 35 RobustDexGrasp training objects as free MuJoCo bodies with
convex-part collision, generated from the upstream URDFs through mjSpec, then
verify each object rests still on the table under the training solver settings.

## Source

`zdchan/RobustDexGrasp`, `rsc/new_training_set`, commit `b00a6cd`. Each object
ships `<src>.urdf` (dummy `bottom` link + real `top` link joined by a
near-locked hinge), `top_watertight_tiny.obj` and `lowest_point_new.txt`.
The `bottom` link is a 1 cm cube far from the origin; it is discarded.

## Asset layout

```
src/mjlab/asset_zoo/objects/dexgrasp/assets/<name>/
  source/<src>.urdf                 # verbatim upstream
  source/top_watertight_tiny.obj    # verbatim upstream
  source/lowest_point_new.txt       # verbatim upstream
  object.xml                        # generated MJCF (one free body)
  collision/hull_NNN.obj            # CoACD convex parts
  manifest.json                     # provenance + build params + digests
  surface.npz                       # 200-pt affordance cloud, normals, centroid
```

`box` and `cylinder` primitives are dropped; every object comes from a URDF.

## Conversion (`convert.py`, run once, committed output)

1. `MjSpec.from_file(urdf)`; attach `top` into a fresh spec; delete the hinge;
   add freejoint `object_joint` (damping 1e-5); rename body to `object`.
2. The URDF collision mesh becomes `object_visual` (non-colliding, mass 0).
   Inertia stays as parsed from the URDF `<inertial>` (explicit, full tensor).
3. CoACD on the source mesh (threshold 0.05, max 16 hulls, seed 0). A mesh
   whose volume fills ≥ 90 % of its convex hull keeps the single hull so a flat
   base stays flat. Parts become `object_collision[_NNN]` with condim 6 and
   friction (0.2, 0.005, 1e-4), mass 0.
4. Write `object.xml` with mesh paths relative to the asset dir.
5. `surface.npz`: 200 surface samples on the source mesh (seed 0),
   `lowest_point` = mesh min z, `placement_lowest_point` = hull min z.
6. `manifest.json`: upstream commit, sha256 of urdf and obj, CoACD params,
   hull files, mass, both lowest points.

## Runtime (`object_constants.py`)

`get_mesh_object_spec(name)` loads `object.xml` and rewrites mesh paths to
absolute (VariantEntityCfg merges many specs into one template). The manifest
digests are checked against `source/` on every load so stale hulls fail loudly.
Public names kept: `PHASE1_OBJECTS`, `DexGraspObject`, `OBJECT_NAMES`,
`ROBUST_DEXGRASP_TRAIN_OBJECTS`, `ROBUST_DEXGRASP_SOURCES`,
`get_phase1_variant_cfg`, `get_robustdexgrasp_variant_cfg`. Hand-copied mass
and inertia constants are removed.

## Verification

- Unit tests: spec compiles to one free body named `object`; mass and COM equal
  the URDF values parsed independently; visual geom non-colliding; hull count
  equals manifest; stale/incomplete manifest raises; variant cfg builds;
  recess probes produce no phantom contact.
- Rest test, all objects, training sim cfg (dt 0.005, iterations 10, pyramidal
  cone, impratio 1, njmax 1500): place at table_z − placement_lowest_point
  + 2 mm, simulate 2 s, require xy drift < 2 mm, tilt < 1°, final speed
  < 1e-3 m/s.
