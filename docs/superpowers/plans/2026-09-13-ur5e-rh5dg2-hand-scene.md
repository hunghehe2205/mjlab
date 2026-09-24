# UR5e + rh5dg2 hand + workstation scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bare UR5e arm asset, a bare rh5dg2 right-hand asset, a combined
UR5e+hand robot (hand attached at the wrist), and a workstation scene (table +
pedestal) to mjlab's asset_zoo, ready for a task to be defined later.

**Architecture:** Follow mjlab's asset_zoo pattern — one folder per source
robot with a vendored MJCF, `xmls/assets/` meshes, and a `<name>_constants.py`
holding `get_spec()`, position actuators, home/collision configs, a
`get_*_robot_cfg()` returning `EntityCfg`, and an `*_ACTION_SCALE`. The arm and
hand stay separate and reusable; a combiner attaches the hand to the UR5e
`attachment_site` via `MjSpec` and produces one combined `EntityCfg`. Table and
pedestal are static world geometry in a new `asset_zoo/scenes/` subpackage.

**Tech Stack:** Python 3.13, `uv`, MuJoCo 3.11 (`mujoco.MjSpec`), mjlab entity
/ actuator configs, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-13-ur5e-rh5dg2-hand-scene-design.md`

## Global Constraints

- Use `uv run` for everything (never bare `python`).
- Style: 88-col limit, 2-space indent, no unnecessary local imports; **minimal
  comments** — docstring or a single one-line comment only.
- **Never edit `docs/source/**`** (no changelog entry).
- Commits/PRs are authored solely by the user — **no `Co-Authored-By: Claude`
  or `Claude-Session` trailers**. Auto commit/push is allowed.
- Run `make check` (ruff format, ruff check --fix, ty) before each commit; it
  must be clean.
- UR5e menagerie MJCF source (already fetched) lives at
  `/tmp/_ur5e_probe/universal_robots_ur5e/`. Hand source lives at
  `/Users/hunghehe2205/Downloads/rh5dg2_hand_description/`.
- Hand joint names after attach keep the `R_*_joint` form (empty attach
  prefix). If MuJoCo's attach rejects an empty prefix, use prefix `"rh5_"` and
  update every hand regex/key (`R_.*` → `rh5_R_.*`) accordingly.

---

### Task 1: Vendor UR5e arm + constants

**Files:**
- Create: `src/mjlab/asset_zoo/robots/universal_robots_ur5e/xmls/ur5e.xml`
- Create: `src/mjlab/asset_zoo/robots/universal_robots_ur5e/xmls/assets/` (20 `.obj` + `LICENSE`)
- Create: `src/mjlab/asset_zoo/robots/universal_robots_ur5e/ur5e_constants.py`
- Create: `src/mjlab/asset_zoo/robots/universal_robots_ur5e/__init__.py` (empty)
- Test: `tests/test_ur5e_arm.py`

**Interfaces:**
- Produces: `get_spec() -> mujoco.MjSpec`, `get_ur5e_arm_cfg() -> EntityCfg`,
  `ATTACHMENT_SITE: str = "attachment_site"`, `ARM_ACTUATORS`,
  `ARTICULATION: EntityArticulationInfoCfg`, `UR5E_ACTION_SCALE: dict[str,float]`.

- [ ] **Step 1: Vendor the assets**

```bash
cd /Users/hunghehe2205/Work/mjlab
DST=src/mjlab/asset_zoo/robots/universal_robots_ur5e
mkdir -p "$DST/xmls/assets"
cp /tmp/_ur5e_probe/universal_robots_ur5e/ur5e.xml "$DST/xmls/ur5e.xml"
cp /tmp/_ur5e_probe/universal_robots_ur5e/assets/*.obj "$DST/xmls/assets/"
cp /tmp/_ur5e_probe/universal_robots_ur5e/LICENSE "$DST/xmls/assets/LICENSE"
touch "$DST/__init__.py"
ls "$DST/xmls/assets" | wc -l   # expect 21 (20 obj + LICENSE)
```

- [ ] **Step 2: Edit `ur5e.xml`** — keep `<compiler ... meshdir="assets">` and
  the `attachment_site`. Remove these blocks (mjlab owns them):
  - the `<option integrator="implicitfast"/>` line,
  - the entire `<actuator>...</actuator>` block,
  - the entire `<keyframe>...</keyframe>` block,
  - the `<light name="spotlight" .../>` line inside `<worldbody>`.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_ur5e_arm.py
import mujoco

from mjlab.asset_zoo.robots.universal_robots_ur5e import ur5e_constants as ur5e
from mjlab.entity.entity import Entity


def test_ur5e_spec_compiles_with_six_joints():
  model = ur5e.get_spec().compile()
  assert model.nq == 6
  assert model.njnt == 6
  site_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)
  }
  assert ur5e.ATTACHMENT_SITE in site_names


def test_ur5e_entity_has_six_position_actuators():
  model = Entity(ur5e.get_ur5e_arm_cfg()).spec.compile()
  assert model.nu == 6
  assert set(ur5e.UR5E_ACTION_SCALE) == {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
  }
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_ur5e_arm.py -v`
Expected: FAIL (module `ur5e_constants` not found).

- [ ] **Step 5: Write `ur5e_constants.py`**

```python
"""UR5e constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

UR5E_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "universal_robots_ur5e"
  / "xmls"
  / "ur5e.xml"
)
assert UR5E_XML.exists()

ATTACHMENT_SITE = "attachment_site"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(UR5E_XML))


SIZE3_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint")
SIZE1_JOINTS = ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

ARM_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=SIZE3_JOINTS,
    stiffness=2000.0,
    damping=400.0,
    effort_limit=150.0,
    armature=0.1,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=SIZE1_JOINTS,
    stiffness=500.0,
    damping=100.0,
    effort_limit=28.0,
    armature=0.1,
  ),
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_ur5e_arm_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_spec, articulation=ARTICULATION)


UR5E_ACTION_SCALE: dict[str, float] = {}
for _a in ARM_ACTUATORS:
  assert _a.effort_limit is not None
  for _n in _a.target_names_expr:
    UR5E_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  viewer.launch(get_spec().compile())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_ur5e_arm.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: `make check`, then commit**

```bash
make check
git add src/mjlab/asset_zoo/robots/universal_robots_ur5e tests/test_ur5e_arm.py
git commit -m "Add bare UR5e arm asset to asset_zoo"
```

---

### Task 2: Vendor rh5dg2 right hand + constants

**Files:**
- Create: `src/mjlab/asset_zoo/robots/rh5dg2_hand/xmls/right_hand.xml`
- Create: `src/mjlab/asset_zoo/robots/rh5dg2_hand/xmls/assets/` (27 `.stl`)
- Create: `src/mjlab/asset_zoo/robots/rh5dg2_hand/rh5dg2_hand_constants.py`
- Create: `src/mjlab/asset_zoo/robots/rh5dg2_hand/__init__.py` (empty)
- Test: `tests/test_rh5dg2_hand.py`

**Interfaces:**
- Produces: `get_spec() -> mujoco.MjSpec`, `HAND_ROOT_BODY: str = "right_hand"`,
  `get_rh5dg2_hand_cfg() -> EntityCfg`, `HAND_ACTUATOR`, `HAND_COLLISION`,
  `RH5DG2_HAND_ACTION_SCALE: dict[str,float]`, `HAND_JOINT_EXPR: str = "R_.*_joint"`.

- [ ] **Step 1: Vendor the assets**

```bash
cd /Users/hunghehe2205/Work/mjlab
SRC=/Users/hunghehe2205/Downloads/rh5dg2_hand_description
DST=src/mjlab/asset_zoo/robots/rh5dg2_hand
mkdir -p "$DST/xmls/assets"
cp "$SRC/mjcf/origin/right_hand.xml" "$DST/xmls/right_hand.xml"
cp "$SRC"/meshes/right_*.stl "$DST/xmls/assets/"
touch "$DST/__init__.py"
ls "$DST/xmls/assets"/*.stl | wc -l   # expect 27
```

- [ ] **Step 2: Edit `right_hand.xml`** — change the compiler line
  `meshdir="../../meshes/"` to `meshdir="assets"`. No other changes.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_rh5dg2_hand.py
import mujoco

from mjlab.asset_zoo.robots.rh5dg2_hand import rh5dg2_hand_constants as hand
from mjlab.entity.entity import Entity

HAND_JOINTS = {
  f"R_{finger}_{seg}_joint"
  for finger in ("thumb", "index", "middle")
  for seg in ("yaw", "mcp", "pip", "dip")
} | {
  f"R_{finger}_{seg}_joint"
  for finger in ("ring", "pinky")
  for seg in ("mcp", "pip", "dip")
}


def test_hand_spec_compiles_with_18_joints():
  model = hand.get_spec().compile()
  assert model.nq == 18
  assert model.njnt == 18
  joints = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
  }
  assert joints == HAND_JOINTS


def test_hand_entity_has_18_position_actuators():
  model = Entity(hand.get_rh5dg2_hand_cfg()).spec.compile()
  assert model.nu == 18
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_rh5dg2_hand.py -v`
Expected: FAIL (module not found).

- [ ] **Step 5: Write `rh5dg2_hand_constants.py`**

```python
"""rh5dg2 right-hand constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

RH5DG2_HAND_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "rh5dg2_hand" / "xmls" / "right_hand.xml"
)
assert RH5DG2_HAND_XML.exists()

HAND_ROOT_BODY = "right_hand"
HAND_JOINT_EXPR = "R_.*_joint"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(RH5DG2_HAND_XML))


# Allegro-class position gains (source caps joint torque at +/-1 Nm).
HAND_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(HAND_JOINT_EXPR,),
  stiffness=1.0,
  damping=0.05,
  effort_limit=1.0,
  armature=1e-3,
)

HAND_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim=3,
  friction={".*_collision": (1.0,)},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HAND_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_rh5dg2_hand_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=ARTICULATION,
    collisions=(HAND_COLLISION,),
  )


RH5DG2_HAND_ACTION_SCALE: dict[str, float] = {HAND_JOINT_EXPR: 0.25 * 1.0 / 1.0}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  viewer.launch(get_spec().compile())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_rh5dg2_hand.py -v`
Expected: PASS. If `CollisionCfg` friction/condim keys differ from the mjlab
signature, mirror the exact field shapes used in
`src/mjlab/asset_zoo/robots/i2rt_yam/yam_constants.py` (`FULL_COLLISION`).

- [ ] **Step 7: `make check`, then commit**

```bash
make check
git add src/mjlab/asset_zoo/robots/rh5dg2_hand tests/test_rh5dg2_hand.py
git commit -m "Add bare rh5dg2 right-hand asset to asset_zoo"
```

---

### Task 3: Workstation scene (table + pedestal)

**Files:**
- Create: `src/mjlab/asset_zoo/scenes/workstation.py`
- Create: `src/mjlab/asset_zoo/scenes/__init__.py` (empty)
- Test: `tests/test_workstation_scene.py`

**Interfaces:**
- Produces: `TABLE_SIZE`, `PEDESTAL_SIZE`, `TABLE_TOP_Z=0.771`,
  `PEDESTAL_TOP_Z=0.731`, `ARM_MOUNT_Z=0.731`, `Y_GAP=0.08`, `TABLE_CENTER`,
  `PEDESTAL_CENTER`, `get_workstation_spec() -> mujoco.MjSpec`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workstation_scene.py
from mjlab.asset_zoo.scenes import workstation as ws


def test_dimensions_and_layout():
  assert ws.TABLE_TOP_Z == 0.771
  assert ws.PEDESTAL_TOP_Z == 0.731
  assert ws.ARM_MOUNT_Z == ws.TABLE_TOP_Z - 0.04
  # Pedestal fully within table width in x.
  assert ws.PEDESTAL_SIZE[0] <= ws.TABLE_SIZE[0]
  # 8 cm gap along y between pedestal front and table near face.
  pedestal_front = ws.PEDESTAL_CENTER[1] + ws.PEDESTAL_SIZE[1]
  table_near = ws.TABLE_CENTER[1] - ws.TABLE_SIZE[1]
  assert abs((table_near - pedestal_front) - ws.Y_GAP) < 1e-9


def test_scene_compiles_with_two_static_boxes():
  model = ws.get_workstation_spec().compile()
  assert model.ngeom == 2  # table + pedestal
  assert model.nq == 0  # static, no freejoints
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workstation_scene.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `workstation.py`**

```python
"""Workstation scene: a table and an arm-mount pedestal, both from the floor."""

import mujoco

TABLE_SIZE = (0.60, 0.55, 0.3855)
PEDESTAL_SIZE = (0.12, 0.12, 0.3655)

TABLE_TOP_Z = 2 * TABLE_SIZE[2]  # 0.771
PEDESTAL_TOP_Z = 2 * PEDESTAL_SIZE[2]  # 0.731
ARM_MOUNT_Z = TABLE_TOP_Z - 0.04  # 0.731

Y_GAP = 0.08
PEDESTAL_CENTER = (0.0, 0.0, PEDESTAL_SIZE[2])
TABLE_CENTER = (
  0.0,
  PEDESTAL_SIZE[1] + Y_GAP + TABLE_SIZE[1],
  TABLE_SIZE[2],
)


def get_workstation_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  table = spec.worldbody.add_body(name="table", pos=TABLE_CENTER)
  table.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_SIZE,
    rgba=(0.6, 0.45, 0.3, 1.0),
  )
  pedestal = spec.worldbody.add_body(name="pedestal", pos=PEDESTAL_CENTER)
  pedestal.add_geom(
    name="pedestal_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=PEDESTAL_SIZE,
    rgba=(0.3, 0.3, 0.33, 1.0),
  )
  return spec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workstation_scene.py -v`
Expected: PASS. (`TABLE_TOP_Z`/`PEDESTAL_TOP_Z` equal `0.771`/`0.731` exactly
since `2*0.3855` and `2*0.3655` are exact in float.)

- [ ] **Step 5: `make check`, then commit**

```bash
make check
git add src/mjlab/asset_zoo/scenes tests/test_workstation_scene.py
git commit -m "Add workstation table+pedestal scene to asset_zoo"
```

---

### Task 4: Combined UR5e + hand robot, exports, integration test

**Files:**
- Create: `src/mjlab/asset_zoo/robots/ur5e_rh5dg2/ur5e_rh5dg2_constants.py`
- Create: `src/mjlab/asset_zoo/robots/ur5e_rh5dg2/__init__.py` (empty)
- Modify: `src/mjlab/asset_zoo/robots/__init__.py` (add re-exports)
- Test: `tests/test_ur5e_rh5dg2.py`

**Interfaces:**
- Consumes: Task 1 `ur5e_constants` (`get_spec`, `ATTACHMENT_SITE`,
  `ARM_ACTUATORS`, `UR5E_ACTION_SCALE`), Task 2 `rh5dg2_hand_constants`
  (`get_spec`, `HAND_ROOT_BODY`, `HAND_ACTUATOR`, `HAND_COLLISION`,
  `RH5DG2_HAND_ACTION_SCALE`), Task 3 `workstation` (`ARM_MOUNT_Z`).
- Produces: `get_ur5e_rh5dg2_robot_cfg() -> EntityCfg`,
  `UR5E_RH5DG2_ACTION_SCALE: dict[str,float]`, `HAND_MOUNT_POS`,
  `HAND_MOUNT_QUAT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ur5e_rh5dg2.py
from mjlab.asset_zoo.robots import get_ur5e_rh5dg2_robot_cfg
from mjlab.entity.entity import Entity


def test_combined_robot_builds_with_24_joints_and_actuators():
  model = Entity(get_ur5e_rh5dg2_robot_cfg()).spec.compile()
  assert model.nq == 24  # 6 arm + 18 hand
  assert model.njnt == 24
  assert model.nu == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ur5e_rh5dg2.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Write `ur5e_rh5dg2_constants.py`**

```python
"""Combined UR5e + rh5dg2 right-hand robot."""

import math

import mujoco

from mjlab.asset_zoo.robots.rh5dg2_hand import rh5dg2_hand_constants as hand
from mjlab.asset_zoo.robots.universal_robots_ur5e import ur5e_constants as ur5e
from mjlab.asset_zoo.scenes.workstation import ARM_MOUNT_Z
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

HAND_PREFIX = ""
HAND_MOUNT_POS = (0.0, 0.0, 0.0)
HAND_MOUNT_QUAT = (1.0, 0.0, 0.0, 0.0)

# Face the arm toward the table (+y); refined in the viewer.
BASE_ROT = (1.0, 0.0, 0.0, 0.0)

ARM_HOME_RAD = {
  "shoulder_pan_joint": -math.pi / 2,
  "shoulder_lift_joint": -math.pi / 2,
  "elbow_joint": math.pi / 2,
  "wrist_1_joint": -math.pi / 2,
  "wrist_2_joint": -math.pi / 2,
  "wrist_3_joint": 0.0,
}


def get_spec() -> mujoco.MjSpec:
  arm = ur5e.get_spec()
  hand_spec = hand.get_spec()
  site = arm.site(ur5e.ATTACHMENT_SITE)
  frame = site.attach_body(hand_spec.body(hand.HAND_ROOT_BODY), HAND_PREFIX, "")
  frame.pos = HAND_MOUNT_POS
  frame.quat = HAND_MOUNT_QUAT
  return arm


ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*ur5e.ARM_ACTUATORS, hand.HAND_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, ARM_MOUNT_Z),
  rot=BASE_ROT,
  joint_pos={**ARM_HOME_RAD, hand.HAND_JOINT_EXPR: 0.0},
  joint_vel={".*": 0.0},
)


def get_ur5e_rh5dg2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
    collisions=(hand.HAND_COLLISION,),
  )


UR5E_RH5DG2_ACTION_SCALE = {
  **ur5e.UR5E_ACTION_SCALE,
  **hand.RH5DG2_HAND_ACTION_SCALE,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.asset_zoo.scenes.workstation import get_workstation_spec

  robot = get_spec()
  scene = get_workstation_spec()
  robot.attach(scene, prefix="scene_")
  viewer.launch(robot.compile())
```

Note on `attach_body`/frame: if `frame.pos`/`frame.quat` are not writable on
the returned object in mujoco 3.11, set the offset by adding an intermediate
frame — replace the attach call with:

```python
  frame = site.add_frame(pos=HAND_MOUNT_POS, quat=HAND_MOUNT_QUAT)
  frame.attach_body(hand_spec.body(hand.HAND_ROOT_BODY), HAND_PREFIX, "")
```

and drop the two `frame.pos/quat` lines. Verify whichever form compiles.

- [ ] **Step 4: Update `robots/__init__.py` re-exports**

Append to `src/mjlab/asset_zoo/robots/__init__.py`:

```python
from mjlab.asset_zoo.robots.universal_robots_ur5e.ur5e_constants import (
  UR5E_ACTION_SCALE as UR5E_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.universal_robots_ur5e.ur5e_constants import (
  get_ur5e_arm_cfg as get_ur5e_arm_cfg,
)
from mjlab.asset_zoo.robots.rh5dg2_hand.rh5dg2_hand_constants import (
  RH5DG2_HAND_ACTION_SCALE as RH5DG2_HAND_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.rh5dg2_hand.rh5dg2_hand_constants import (
  get_rh5dg2_hand_cfg as get_rh5dg2_hand_cfg,
)
from mjlab.asset_zoo.robots.ur5e_rh5dg2.ur5e_rh5dg2_constants import (
  UR5E_RH5DG2_ACTION_SCALE as UR5E_RH5DG2_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.ur5e_rh5dg2.ur5e_rh5dg2_constants import (
  get_ur5e_rh5dg2_robot_cfg as get_ur5e_rh5dg2_robot_cfg,
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ur5e_rh5dg2.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite for the new assets + `make check`**

Run: `uv run pytest tests/test_ur5e_arm.py tests/test_rh5dg2_hand.py tests/test_workstation_scene.py tests/test_ur5e_rh5dg2.py -v`
Then: `make check`
Expected: all PASS, checks clean.

- [ ] **Step 7: Visual tuning (manual)**

Run: `uv run python -m mjlab.asset_zoo.robots.ur5e_rh5dg2.ur5e_rh5dg2_constants`
Adjust `HAND_MOUNT_POS/QUAT` so the palm points along the wrist approach axis,
and `BASE_ROT` so the arm faces the table. Re-run the suite after edits.

- [ ] **Step 8: Commit**

```bash
git add src/mjlab/asset_zoo/robots/ur5e_rh5dg2 \
        src/mjlab/asset_zoo/robots/__init__.py tests/test_ur5e_rh5dg2.py
git commit -m "Add combined UR5e + rh5dg2 hand robot and exports"
```

---

## Self-Review

**Spec coverage:**
- Bare UR5e arm asset + constants + gains → Task 1. ✓
- Bare rh5dg2 right hand + constants + Allegro gains + collision → Task 2. ✓
- Workstation table+pedestal scene + layout invariants → Task 3. ✓
- Combined robot via attach, home pose, base at pedestal top, combined action
  scale, viewer entry point, exports → Task 4. ✓
- XML edits (strip option/actuator/keyframe/light; hand meshdir) → Tasks 1, 2. ✓
- Tests (nq/nu counts, layout), `make check`, no docs/source edit, no Claude
  co-author → all tasks. ✓

**Placeholder scan:** No TBD/TODO; every code step has full content. The
`HAND_MOUNT_POS/QUAT` and `BASE_ROT` values are intentionally identity
placeholders tuned in Task 4 Step 7 (a documented manual step), not code gaps.

**Type consistency:** `get_spec`, `ATTACHMENT_SITE`, `HAND_ROOT_BODY`,
`HAND_ACTUATOR`, `HAND_COLLISION`, `HAND_JOINT_EXPR`, `ARM_ACTUATORS`,
`*_ACTION_SCALE`, `get_ur5e_rh5dg2_robot_cfg` names match across tasks.
