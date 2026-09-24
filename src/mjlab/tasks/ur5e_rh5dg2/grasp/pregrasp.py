"""RobustDexGrasp-style pre-grasp sampling: object pose, approach, roll and IK.

Ported from the reset loop of ``allegro_teacher/train.py`` and
``helper/initial_pose_final.py``. The pool is solved once on the native model and
the reset event draws from it, replacing the per-iteration CPU resampling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from mjlab.tasks.ur5e_rh5dg2.grasp.constants import (
  ANGLE_SCORE_COEFF,
  ARM_IK_SEED,
  BOX_SIZE,
  CAMERA_POS,
  GRASP_WIDTH_LIMIT,
  HAND_CENTER,
  LENGTH_SCORE_COEFF,
  LIFT_OFFSET,
  NUM_ROLLS,
  OBJECT_ANGLE,
  OBJECT_DISTANCE,
  OBJECT_MAX_ABS_X,
  OBJECT_POS,
  OPEN_HAND,
  PREGRASP_CLEARANCE,
  STANDOFF,
  TOP_GRASP,
)

ARM_JOINTS = (
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint",
)
HAND_JOINTS = (
  "R_thumb_yaw_joint",
  "R_thumb_mcp_joint",
  "R_thumb_pip_joint",
  "R_thumb_dip_joint",
  "R_index_yaw_joint",
  "R_index_mcp_joint",
  "R_index_pip_joint",
  "R_index_dip_joint",
  "R_middle_yaw_joint",
  "R_middle_mcp_joint",
  "R_middle_pip_joint",
  "R_middle_dip_joint",
  "R_ring_mcp_joint",
  "R_ring_pip_joint",
  "R_ring_dip_joint",
  "R_pinky_mcp_joint",
  "R_pinky_pip_joint",
  "R_pinky_dip_joint",
)
WRIST_BODY = "robot/right_hand"


class Pregrasp(NamedTuple):
  arm: np.ndarray  # [6]
  rotation: np.ndarray  # [3, 3] wrist frame in the world
  center: np.ndarray  # [3] affordance center of the visible points
  lift: np.ndarray  # [6] arm with the wrist raised by LIFT_OFFSET


@dataclass
class PregraspPool:
  object_pos: np.ndarray  # [N, 3] env-local
  object_quat: np.ndarray  # [N, 4] wxyz
  arm_pos: np.ndarray  # [N, 6]
  lift_pos: np.ndarray  # [N, 6]


def sample_object_xy(rng: np.random.Generator, edge_biased: bool) -> np.ndarray:
  """Polar sample in front of the base; Beta(0.5, 0.5) favours the region edges."""
  (a0, a1), (d0, d1) = OBJECT_ANGLE, OBJECT_DISTANCE
  while True:
    u, v = rng.beta(0.5, 0.5, 2) if edge_biased else rng.uniform(0.0, 1.0, 2)
    angle, distance = a0 + u * (a1 - a0), d0 + v * (d1 - d0)
    xy = distance * np.array([np.cos(angle), np.sin(angle)])
    if abs(xy[0]) < OBJECT_MAX_ABS_X:
      return xy


def yaw_quat(yaw: float) -> np.ndarray:
  return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def box_surface_points(rng: np.random.Generator, count: int = 200) -> np.ndarray:
  """Area-weighted points on the box surface with their outward normals."""
  h = np.array(BOX_SIZE)
  areas = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]]).repeat(2)
  face = rng.choice(6, size=count, p=areas / areas.sum())
  axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
  points = rng.uniform(-h, h, size=(count, 3))
  points[np.arange(count), axis] = sign * h[axis]
  normals = np.zeros((count, 3))
  normals[np.arange(count), axis] = sign
  return np.concatenate([points, normals], axis=-1)


def visible_points(
  surface: np.ndarray, pos: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
  """Single-view visibility; exact for a convex primitive."""
  points = surface[:, :3] @ rotation.T + pos
  normals = surface[:, 3:] @ rotation.T
  facing = ((np.array(CAMERA_POS) - points) * normals).sum(axis=-1) > 0.0
  return points[facing]


def candidate_frames(
  approach: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Wrist rotations about the approach axis and the grasp width along the fingers.

  Hand +X (palm normal) faces the object and +Z (fingers) sweeps the roll circle,
  oriented away from the robot like the reference's half-plane flip.
  """
  ref = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0, 1.0, 0])
  e1 = np.cross(approach, ref)
  e1 /= np.linalg.norm(e1)
  e2 = np.cross(approach, e1)
  thetas = np.linspace(0.0, np.pi, NUM_ROLLS, endpoint=False)
  fingers = np.cos(thetas)[:, None] * e1 + np.sin(thetas)[:, None] * e2
  fingers *= np.where(fingers[:, 1] < 0.0, -1.0, 1.0)[:, None]
  palm = np.broadcast_to(-approach, fingers.shape)
  lateral = np.cross(fingers, palm)
  rotations = np.stack([palm, lateral, fingers], axis=-1)
  centered = points - points.mean(axis=0)
  projection = centered @ fingers.T
  return rotations, projection.max(axis=0) - projection.min(axis=0)


class PregraspSolver:
  """Damped least-squares IK and collision checks on a private native copy."""

  def __init__(self, model: mujoco.MjModel):
    self.model = model = copy.copy(model)
    self.data = mujoco.MjData(model)
    arm = [model.joint(f"robot/{n}").id for n in ARM_JOINTS]
    hand = [model.joint(f"robot/{n}").id for n in HAND_JOINTS]
    self.arm_qpos = model.jnt_qposadr[arm]
    self.arm_dof = model.jnt_dofadr[arm]
    self.hand_qpos = model.jnt_qposadr[hand]
    lo, hi = model.jnt_range[arm].T
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
    self.arm_limits = np.stack([mid - half, mid + half], axis=-1)
    obj = model.joint("object/free_joint")
    self.object_qpos = int(obj.qposadr[0])
    self.wrist = model.body(WRIST_BODY).id
    self.robot_geoms = np.array(
      [
        model.body(model.geom_bodyid[g]).name.startswith("robot/")
        for g in range(model.ngeom)
      ]
    )
    names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
    self.mount_geoms = np.array(
      [n in ("robot/base", "robot/shoulder_link", "props/pedestal") for n in names]
    )
    # Pair margins take the larger geom margin, so only robot-world pairs get it.
    model.geom_margin[~self.robot_geoms] = PREGRASP_CLEARANCE
    self.jacp = np.zeros((3, model.nv))
    self.jacr = np.zeros((3, model.nv))

  def set_state(self, arm: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    self.data.qpos[self.arm_qpos] = arm
    self.data.qpos[self.hand_qpos] = OPEN_HAND
    self.data.qpos[self.object_qpos : self.object_qpos + 3] = pos
    self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = quat

  def ik(
    self, pos: np.ndarray, rotation: np.ndarray, seed: np.ndarray
  ) -> np.ndarray | None:
    q = seed.copy()
    quat, target_quat, err_r = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(target_quat, rotation.flatten())
    for _ in range(100):
      self.data.qpos[self.arm_qpos] = q
      mujoco.mj_kinematics(self.model, self.data)
      mujoco.mj_comPos(self.model, self.data)
      mujoco.mju_mat2Quat(quat, self.data.xmat[self.wrist])
      mujoco.mju_subQuat(err_r, target_quat, quat)
      rot_err = self.data.xmat[self.wrist].reshape(3, 3) @ err_r
      err = np.concatenate([pos - self.data.xpos[self.wrist], rot_err])
      if np.abs(err[:3]).max() < 1e-4 and np.abs(err[3:]).max() < 1e-3:
        lo, hi = self.arm_limits.T
        return q if ((q >= lo) & (q <= hi)).all() else None
      mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.wrist)
      jac = np.vstack([self.jacp[:, self.arm_dof], self.jacr[:, self.arm_dof]])
      step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
      q = q + np.clip(step, -0.3, 0.3)
    return None

  def collides(self, arm: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> bool:
    """Robot contact with the world inside the margin, or robot self-penetration."""
    self.set_state(arm, pos, quat)
    mujoco.mj_forward(self.model, self.data)
    contact = self.data.contact
    robot = self.robot_geoms[contact.geom]
    world = robot.any(axis=-1) & ~robot.all(axis=-1)
    world &= ~self.mount_geoms[contact.geom].all(axis=-1)
    return bool(world.any() or (contact.dist[robot.all(axis=-1)] < 0.0).any())

  def wrist_pose(
    self, center: np.ndarray, rotation: np.ndarray, standoff: float
  ) -> np.ndarray:
    return center + standoff * -rotation[:, 0] - rotation @ np.array(HAND_CENTER)

  def seed(self, pos: np.ndarray) -> np.ndarray:
    """Rotate the calibrated branch's base yaw toward the object."""
    seed = np.array(ARM_IK_SEED)
    seed[0] += np.arctan2(pos[1], pos[0]) - np.arctan2(OBJECT_POS[1], OBJECT_POS[0])
    return seed

  def solve(
    self, pos: np.ndarray, quat: np.ndarray, surface: np.ndarray
  ) -> Pregrasp | None:
    """Best collision-free pre-grasp for one object placement."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    points = visible_points(surface, pos, rotation.reshape(3, 3))
    center = points.mean(axis=0)
    approach = np.array([0.0, 0.0, 1.0]) if TOP_GRASP else np.array(CAMERA_POS) - center
    approach /= np.linalg.norm(approach)
    rotations, widths = candidate_frames(approach, points)
    best, best_score = None, np.inf
    seed = self.seed(pos)
    up = np.array([0.0, 0.0, LIFT_OFFSET])
    for rot, width in zip(rotations, widths, strict=True):
      q = self.ik(self.wrist_pose(center, rot, STANDOFF), rot, seed)
      if q is None or self.collides(q, pos, quat):
        continue
      if widths.min() >= GRASP_WIDTH_LIMIT:
        score = width
      elif width < GRASP_WIDTH_LIMIT:
        score = (
          LENGTH_SCORE_COEFF * width
          + ANGLE_SCORE_COEFF * abs(q[4] - np.pi / 2)
          + 0.5 * ANGLE_SCORE_COEFF * (abs(q[4]) - 3.2)
        )
      else:
        continue
      if score >= best_score:
        continue
      lift = self.ik(self.wrist_pose(center, rot, STANDOFF) + up, rot, q)
      if lift is not None:
        best, best_score = Pregrasp(q, rot, center, lift), score
    return best


def build_pool(
  model: mujoco.MjModel, size: int, seed: int, edge_biased: bool
) -> PregraspPool:
  """Sample placements until ``size`` feasible pre-grasps are found."""
  rng = np.random.default_rng(seed)
  solver = PregraspSolver(model)
  surface = box_surface_points(rng)
  z = OBJECT_POS[2]
  pos, quat, arm, lift = [], [], [], []
  for _ in range(20 * size):
    xy = sample_object_xy(rng, edge_biased and rng.random() < 0.5)
    p = np.array([xy[0], xy[1], z])
    q = yaw_quat(rng.uniform(-np.pi, np.pi))
    solution = solver.solve(p, q, surface)
    if solution is None:
      continue
    pos.append(p)
    quat.append(q)
    arm.append(solution.arm)
    lift.append(solution.lift)
    if len(arm) == size:
      return PregraspPool(*map(np.array, (pos, quat, arm, lift)))
  raise RuntimeError(f"Only {len(arm)} of {size} pre-grasps were feasible.")
