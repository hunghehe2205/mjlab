"""Robot kinematics helper for pre-grasp targeting.

Compiles the UR5e + RH5-DG2 spec once and turns a desired ``rh/grasp_center``
pose (env frame: arm base at the origin xy, z up, table top at its world
height) into arm joint targets via the analytic IK. The constant
flange->grasp_center transform and the base-body pose are measured from the
compiled model, so the frame chain is grounded in MuJoCo, not hand-derived.

Env frame vs spec frame: the spec compiles with the base body at the origin;
in the scene the whole arm is raised by ``mount_pos`` (pedestal top), a pure
translation, so env = spec + mount_pos.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.tasks.dexgrasp.pregrasp.ik_ur5e import InverseKinematicsUR5e, solve_arm_qpos

_GRASP_CENTER_SITE = "rh/grasp_center"
_FLANGE_SITE = "attachment_site"


def _inv_T(T: np.ndarray) -> np.ndarray:
  out = np.eye(4)
  out[:3, :3] = T[:3, :3].T
  out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
  return out


class ArmKinematics:
  """Grasp-center <-> arm-qpos mapping for a mount-raised UR5e."""

  def __init__(self, mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    self._model = rc.get_spec().compile()
    self._data = mujoco.MjData(self._model)
    self._qadr = np.array(
      [
        self._model.jnt_qposadr[
          mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
        ]
        for n in rc.ARM_JOINT_NAMES
      ]
    )
    self._gc = mujoco.mj_name2id(
      self._model, mujoco.mjtObj.mjOBJ_SITE, _GRASP_CENTER_SITE
    )
    self._fl = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, _FLANGE_SITE)
    self._bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base")
    self._mount = np.asarray(mount_pos, dtype=float)
    self._ik = InverseKinematicsUR5e()

    self._forward(np.zeros(6))
    self._T_spec_base = self._body_T(self._bid)
    self._T_flange_gc = _inv_T(self._site_T(self._fl)) @ self._site_T(self._gc)

  def _forward(self, q: np.ndarray) -> None:
    self._data.qpos[:] = 0.0
    self._data.qpos[self._qadr] = q
    mujoco.mj_forward(self._model, self._data)

  def _site_T(self, sid: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = self._data.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = self._data.site_xpos[sid]
    return T

  def _body_T(self, bid: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = self._data.xmat[bid].reshape(3, 3)
    T[:3, 3] = self._data.xpos[bid]
    return T

  def fk_grasp_center_env(self, q: np.ndarray) -> np.ndarray:
    """grasp_center pose (4x4) in the env frame for arm qpos ``q`` (6,)."""
    self._forward(q)
    T = self._site_T(self._gc)
    T[:3, 3] += self._mount
    return T

  def arm_qpos_for_grasp_center(
    self, pos_env: np.ndarray, rot_env: np.ndarray, seed: np.ndarray
  ) -> np.ndarray | None:
    """Arm qpos (6,) landing grasp_center at ``pos_env``/``rot_env``, or None."""
    T_spec_gc = np.eye(4)
    T_spec_gc[:3, :3] = rot_env
    T_spec_gc[:3, 3] = np.asarray(pos_env, dtype=float) - self._mount
    T_spec_flange = T_spec_gc @ _inv_T(self._T_flange_gc)
    T_base_flange = _inv_T(self._T_spec_base) @ T_spec_flange
    return solve_arm_qpos(T_base_flange, seed, ik=self._ik)
