# Design: UR5e + rh5dg2 right hand + workstation scene

Date: 2026-09-13
Branch: DexGrasp
Status: approved (design), pending implementation

## Goal

Add a UR5e arm (from `mujoco_menagerie`) and the `rh5dg2` dexterous right
hand (from the local `rh5dg2_hand_description`, origin
`bitbucket.org/vinrobotics/rh5dg2_hand_description`) to mjlab, organized in
the style of `omarrayyann/mjlab_manipulation`: a **bare arm** asset and a
**separate hand** asset, combined by attaching the hand to the arm's tool
frame. Also add the surrounding **workstation scene** (a table and an arm
mount pedestal). No RL task yet — a task will be defined later. The
deliverable is a loadable, viewable combined robot standing on the pedestal
in front of the table, plus reusable assets a future task can consume.

## Non-goals

- No RL task, reward/observation/command design, or training config.
- No left hand on the arm (the left hand meshes/XML are not vendored now).
- No tendon coupling / underactuation modeling: the hand is fully actuated
  (18 independent joints), matching the source MJCF.
- No hardware-accurate hand gains (no datasheet exists); gains are a
  research-backed baseline, tunable later.

## Source facts (verified)

### UR5e (`mujoco_menagerie/universal_robots_ur5e`)
- MJCF `ur5e.xml` + 20 `.obj` meshes in `assets/`, BSD-3-Clause `LICENSE`.
- 6 joints: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`,
  `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`.
- Position `general` actuators via `<default>`:
  - size3 (`shoulder_pan/lift`, `elbow`): `gainprm=2000`, `biasprm=0 -2000 -400`,
    `forcerange=±150`, `armature=0.1`. (`elbow` uses `size3_limited`:
    `ctrlrange/range = ±3.1415`.)
  - size1 (`wrist_1/2/3`): `gainprm=500`, `biasprm=0 -500 -100`,
    `forcerange=±28`, `armature=0.1`.
- `wrist_3_link` contains `<site name="attachment_site" pos="0 0.1 0"
  quat="-1 1 0 0"/>` — the tool mount frame.
- Home keyframe: `qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]`.
- Base body has `quat="0 0 0 -1"`.

### rh5dg2 right hand (`rh5dg2_hand_description/mjcf/origin/right_hand.xml`)
- Root body `right_hand` — **no joint** (attachable). `meshdir="../../meshes/"`.
- 18 joints, all `axis`-driven hinges, each `actuatorfrcrange="-1 1"` (±1 Nm):
  - thumb: `R_thumb_{yaw,mcp,pip,dip}_joint`
  - index: `R_index_{yaw,mcp,pip,dip}_joint`
  - middle: `R_middle_{yaw,mcp,pip,dip}_joint`
  - ring: `R_ring_{mcp,pip,dip}_joint`
  - pinky: `R_pinky_{mcp,pip,dip}_joint`
  - (thumb/index/middle have yaw = 4 DoF; ring/pinky = 3 DoF;
    3 fingers × 4 + 2 fingers × 3 = 18.)
- Flexion joint ranges are `[0, ~1.5]` → **0 = open/extended**; yaw ranges are
  small around 0.
- Collision geoms use class `rh5dg2_hand_collision`, named `*_collision`
  (group 3). Visual class `rh5dg2_hand_visual` (group 2). No name clashes with
  the UR5e defaults (`ur5e`/`visual`/`collision`).
- 27 right-hand `.stl` meshes (of 54 total in the shared `meshes/` dir).
- The source repo's `actuators.xml`/`sensors.xml` are separate combined
  (L+R) files and are **not** referenced by `right_hand.xml`; we do not
  vendor them. Actuators are defined in Python.

### Environment
- `mujoco 3.11.0`. `MjSpec.attach` and `MjsSite.attach_body` exist.
- Verified: `site.attach_body(hand.body("right_hand"), prefix, "")` then
  `arm.compile()` succeeds → combined model `nq=24, njnt=24` (6 arm + 18 hand).

## Architecture

Mirror the existing mjlab asset_zoo pattern (one folder per source robot,
`<name>_constants.py` with `get_spec()` + actuator/collision/home configs +
`get_*_robot_cfg()` + `*_ACTION_SCALE`, exported via `robots/__init__.py`),
plus a new `scenes/` subpackage for task-agnostic furniture.

### File layout (new)

```
src/mjlab/asset_zoo/
  robots/
    universal_robots_ur5e/
      __init__.py
      ur5e_constants.py
      xmls/ur5e.xml               # vendored, edited (see below)
      xmls/assets/*.obj (20)      # + LICENSE (BSD-3)
    rh5dg2_hand/
      __init__.py
      rh5dg2_hand_constants.py
      xmls/right_hand.xml         # vendored, meshdir -> assets
      xmls/assets/*.stl (27)      # right-hand meshes only
    ur5e_rh5dg2/
      __init__.py
      ur5e_rh5dg2_constants.py    # attach + combined EntityCfg
  scenes/
    __init__.py
    workstation.py                # table + pedestal spec + dims
```

`robots/__init__.py` re-exports:
`get_ur5e_arm_cfg`, `UR5E_ACTION_SCALE`, `get_rh5dg2_hand_cfg`,
`RH5DG2_HAND_ACTION_SCALE`, `get_ur5e_rh5dg2_robot_cfg`,
`UR5E_RH5DG2_ACTION_SCALE`.

### Vendored-XML edits
- `ur5e.xml`: keep `<compiler ... meshdir="assets">`; **remove** `<option>`,
  `<actuator>`, and `<keyframe>` (mjlab owns integrator/actuators/home);
  remove the tracking `<light>` (scene provides lighting). Keep
  `attachment_site`.
- `right_hand.xml`: change `meshdir="../../meshes/"` → `meshdir="assets"`;
  no other edits.

### UR5e arm (`ur5e_constants.py`)
- `UR5E_XML = MJLAB_SRC_PATH / "asset_zoo/robots/universal_robots_ur5e/xmls/ur5e.xml"`,
  `get_spec()` loads it.
- `ATTACHMENT_SITE = "attachment_site"` (exported for the combiner).
- Actuators as `BuiltinPositionActuatorCfg` using menagerie-tuned gains:
  - size3 group (`shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`):
    `stiffness=2000, damping=400, effort_limit=150, armature=0.1`.
  - size1 group (`wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`):
    `stiffness=500, damping=100, effort_limit=28, armature=0.1`.
- `ARTICULATION = EntityArticulationInfoCfg(actuators=(...),
  soft_joint_pos_limit_factor=0.9)`.
- `get_ur5e_arm_cfg() -> EntityCfg` (bare arm; base placement left to caller).
- `UR5E_ACTION_SCALE`: per-joint `0.25 * effort / stiffness` (yam convention).
- `__main__`: launch viewer on the bare arm (sanity).

### rh5dg2 right hand (`rh5dg2_hand_constants.py`)
- `RH5DG2_HAND_XML = .../rh5dg2_hand/xmls/right_hand.xml`, `get_spec()`.
- 18 `BuiltinPositionActuatorCfg`, one per joint (regex `("<joint>",)` each, or
  one cfg with all 18 names), **research-backed Allegro-class gains**:
  `stiffness=1.0, damping=0.05, effort_limit=1.0, armature=1e-3`, uniform.
  Rationale: the source caps joint torque at ±1 Nm; the Allegro hand
  (anthropomorphic, ~4-DoF fingers, comparable torque) uses kp≈1.0, kd≈0.05
  as the standard MuJoCo-RL position-control baseline. Uniform now, tunable
  per finger later. `effort_limit=1.0` matches `actuatorfrcrange`.
- `COLLISION = CollisionCfg(geom_names_expr=(".*_collision",), contype=1,
  conaffinity=1, condim=3, friction=(1.0,))` — enable palm+fingertip contact
  for future grasping; keep it simple/tunable. (Self-collision left permissive
  for now; can be restricted when the task is defined.)
- `get_rh5dg2_hand_cfg() -> EntityCfg` (bare hand, for standalone viewing).
- `RH5DG2_HAND_ACTION_SCALE` (yam convention).
- `__main__`: viewer on the bare hand.

### Combined robot (`ur5e_rh5dg2_constants.py`)
- `HAND_MOUNT_POS`, `HAND_MOUNT_QUAT` — the **tool-attach offset** applied when
  attaching, so the palm points along the wrist approach axis. Start with
  identity offset; final values dialed in via the viewer.
- `HAND_PREFIX = ""` (hand joint names stay `R_*_joint`; no clashes). If
  MuJoCo requires a non-empty prefix at attach time, fall back to `"rh5_"`
  and update all downstream regexes/keys accordingly.
- `get_spec()`:
  1. `arm = ur5e.get_spec()`, `hand = rh5dg2_hand.get_spec()`.
  2. `site = arm.site(ATTACHMENT_SITE)`.
  3. `frame = site.attach_body(hand.body("right_hand"), HAND_PREFIX, "")`.
  4. apply `HAND_MOUNT_POS/QUAT` to the attach frame (offset).
  5. return `arm`.
- `ARM_MOUNT_Z = TABLE_TOP_Z - 0.04 = 0.731` (from `scenes.workstation`).
- `HOME = EntityCfg.InitialStateCfg(pos=(0,0,ARM_MOUNT_Z), rot=<face +y>,
  joint_pos={<arm menagerie home in rad>, "R_.*_joint": 0.0},
  joint_vel={".*": 0.0})`. `rot` orients the arm toward the table (+y);
  starting quaternion tuned in the viewer.
- `ARTICULATION`: concatenation of the arm actuators and the 18 hand actuators.
- `COLLISIONS`: arm + hand collision cfgs.
- `get_ur5e_rh5dg2_robot_cfg() -> EntityCfg(init_state=HOME,
  spec_fn=get_spec, articulation=ARTICULATION, collisions=COLLISIONS)`.
- `UR5E_RH5DG2_ACTION_SCALE = {**UR5E_ACTION_SCALE, **RH5DG2_HAND_ACTION_SCALE}`
  (hand keys prefixed if `HAND_PREFIX` is non-empty).
- `__main__`: launch the MuJoCo viewer with the combined robot on the pedestal
  and the workstation scene, for visual verification and offset/facing tuning.

### Workstation scene (`scenes/workstation.py`)
Half-extents are MuJoCo box `size` (half the real dimension).

| Piece    | `size` (half)          | Real D×W×H (m)      | Center (x,y,z)     | Top z |
|----------|------------------------|---------------------|--------------------|-------|
| Table    | (0.60, 0.55, 0.3855)   | 1.20 × 1.10 × 0.771 | (0, 0.75, 0.3855)  | 0.771 |
| Pedestal | (0.12, 0.12, 0.3655)   | 0.24 × 0.24 × 0.731 | (0, 0.00, 0.3655)  | 0.731 |

Constants: `TABLE_SIZE`, `PEDESTAL_SIZE`, `TABLE_TOP_Z=0.771`,
`PEDESTAL_TOP_Z=0.731`, `ARM_MOUNT_Z=TABLE_TOP_Z-0.04=0.731`, `Y_GAP=0.08`,
`TABLE_CENTER`, `PEDESTAL_CENTER`.

Layout invariants (match the request):
- Both boxes are solid, static (no freejoint), resting on the floor (z=0).
- Pedestal top is exactly 4 cm below the table top.
- 8 cm gap along y between pedestal and table (pedestal front face y=+0.12,
  table near face y=+0.20); pedestal x-extent ⊂ table x-extent (no x gap).

`get_workstation_spec() -> mujoco.MjSpec` builds a spec with two static box
geoms (table, pedestal) so a future task can attach/merge it into its
`SceneCfg`. (Furniture as static world geometry, not an articulated entity.)

## Testing / verification

- `tests/test_ur5e_rh5dg2.py` (flat, functions + fixtures, targeted):
  - combined `EntityCfg` builds; spec compiles; `nq == 24`, `njnt == 24`.
  - exactly 24 position actuators; actuator names cover all 6 arm + 18 hand
    joints.
  - `get_workstation_spec()` compiles; table/pedestal top-z equal 0.771/0.731.
- `make check` (ruff format, ruff check --fix, ty) clean; keep 88-col limit,
  2-space indent, no unnecessary local imports, and minimal comments
  (docstring or a single one-line comment only).
- No changelog edit: `docs/source/**` must not be touched (user rule), which
  overrides the CLAUDE.md changelog convention.
- Visual check via the combined `__main__` viewer to finalize
  `HAND_MOUNT_POS/QUAT` and the base facing quaternion.

## Open items dialed in during implementation (not blockers)
- Exact `HAND_MOUNT_POS/QUAT` (palm alignment to the wrist approach axis).
- Exact base facing quaternion (arm oriented toward the table).
- Whether `attach` tolerates an empty prefix (fallback `"rh5_"`).

## References
- UR5e MJCF: mujoco_menagerie `universal_robots_ur5e`.
- Reference org style: `github.com/omarrayyann/mjlab_manipulation`
  (bare arm under `robots/<name>/xmls/`; gripper attached at task time;
  `<name>_constants.py` holds home pose + tool-attach offset).
- Hand gains: Allegro (kp≈1.0, kd≈0.05) / TriFinger (kp=10, kd=0.05) as the
  standard MuJoCo-RL position-control baselines for dexterous hands.
