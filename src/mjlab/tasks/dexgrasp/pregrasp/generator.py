"""Pre-grasp generator: object pose -> arm qpos (side grasp toward the camera).

Ties the tested pieces together (visibility raycast, palm-roll sampling,
analytic IK via ArmKinematics, projection-width scoring) the way
RobustDexGrasp's train.py does, adapted to UR5e + RH5-DG2. Deterministic given
its inputs; object-pose randomness is the caller's.

Camera position, the wrist-angle score terms, and the IK-failure fallback pose
are ported from the UR5 setup and are the values most likely to need
viewer/empirical tuning here -- see documents/problems/pregrasp-ur5e-tuning.md.
"""

from __future__ import annotations

from math import atan2, pi

import numpy as np
import trimesh

from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics
from mjlab.tasks.dexgrasp.pregrasp.pose_sampler import sample_rot_mats
from mjlab.tasks.dexgrasp.pregrasp.visibility import visible_points

# Fixed camera in the env frame (RobustDexGrasp cfg_reg.yaml value).
CAMERA_POSITION = np.array([0.035, -0.58, 1.531])
# Grasp-center standoff from the visible-surface centre along the camera ray.
# The reference's 0.25 is for the Allegro palm centre; grasp_center here sits
# ~2 cm behind the fingertips (thumb no longer ahead of them, see
# INIT_FINGER_POSE). Measured over all 35 objects (140 resets): 0.10 puts the
# nearest tip at 3.0 cm min / 7.9 cm mean with no reset contact; 0.08 already
# touches on 1% of resets.
APPROACH_DISTANCE = 0.10
NUM_ROLL_SAMPLES = 10
PROJ_LIMIT = 0.18  # gripper-axis object width below which a grasp is "narrow"
LENGTH_COEFF = 5.0
ANGLE_COEFF = 1.0


def generate_pregrasp(
  obj_pos: np.ndarray,
  obj_quat: np.ndarray,
  aff_mesh: trimesh.Trimesh,
  aff_pcd: np.ndarray,
  kin: ArmKinematics,
  seed_qpos: np.ndarray,
  camera_pos: np.ndarray = CAMERA_POSITION,
  num_samples: int = NUM_ROLL_SAMPLES,
) -> np.ndarray | None:
  """Arm qpos (6,) for the side-grasp pre-grasp, or None when IK fails.

  ``obj_pos``/``obj_quat`` are the object root in the env frame; ``aff_mesh`` and
  ``aff_pcd`` its affordance surface (object frame). Scores feasible palm rolls
  by object width along the gripper axis (narrow preferred), then wrist angle.
  """
  visible, center = visible_points(aff_mesh, aff_pcd, obj_pos, obj_quat, camera_pos)
  approach = camera_pos - center
  approach = approach / np.linalg.norm(approach)
  gc_target = center + APPROACH_DISTANCE * approach

  rot_mats, proj = sample_rot_mats(approach, num_samples, visible)

  ik_res = np.zeros((num_samples, 6))
  feasible = np.zeros(num_samples, dtype=bool)
  for j in range(num_samples):
    q = kin.arm_qpos_for_grasp_center(gc_target, rot_mats[j], seed=seed_qpos)
    if q is None or np.isnan(q).any():
      continue
    ik_res[j] = q
    feasible[j] = True

  feas_idx = np.flatnonzero(feasible)
  if feas_idx.size == 0:
    return None

  narrow = [j for j in feas_idx if proj[j] < PROJ_LIMIT]
  if narrow:
    best = min(narrow, key=lambda j: proj[j] * LENGTH_COEFF + _wrist_penalty(ik_res[j]))
  else:
    best = int(feas_idx[np.argmin(proj[feas_idx])])
  return ik_res[best]


def _wrist_penalty(arm_qpos: np.ndarray) -> float:
  """Prefer |wrist_2| near pi/2, either sign (index 4 = wrist_2_joint).

  Symmetric so it agrees with whichever branch the seed selects; the HOME seed
  has wrist_2 = -pi/2, so a one-sided +pi/2 preference would fight the seed.
  """
  return abs(abs(arm_qpos[4]) - 1.57) * ANGLE_COEFF


def fallback_arm_qpos(obj_pos: np.ndarray) -> np.ndarray:
  """Canonical arm pose facing the object when IK fails.

  Re-derived for this base frame (Rz(180) vs the reference UR5's Rz(90)): the
  grasp-center azimuth trails shoulder_pan by ~180 deg, so aim shoulder_pan at
  the object azimuth + pi (keeping the reference's -0.3 approach offset).
  """
  angle = atan2(obj_pos[1], obj_pos[0])
  return np.array([angle + pi - 0.3, -1.57, 1.57, 0.0, 1.57, -1.57])
