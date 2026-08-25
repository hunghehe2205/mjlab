"""Static arm-hand collision probe for pre-grasp reset candidates."""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc


class ArmHandSelfCollisionProbe:
  """Check arm-to-hand contacts without stepping the simulation."""

  def __init__(self) -> None:
    self._model = rc.get_spec().compile()
    self._data = mujoco.MjData(self._model)
    self._qpos = np.zeros(self._model.nq)
    self._arm_qpos_adr = np.asarray(
      [
        self._model.jnt_qposadr[
          mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in rc.ARM_JOINT_NAMES
      ],
      dtype=np.intp,
    )
    for name, value in rc.INIT_FINGER_POSE.items():
      joint_id = mujoco.mj_name2id(
        self._model, mujoco.mjtObj.mjOBJ_JOINT, f"{rc.HAND_PREFIX}{name}"
      )
      self._qpos[self._model.jnt_qposadr[joint_id]] = value
    self._arm_body_ids = {
      mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
      for name in rc.ARM_LINK_BODIES
    }
    hand_root = mujoco.mj_name2id(
      self._model, mujoco.mjtObj.mjOBJ_BODY, f"{rc.HAND_PREFIX}right_hand"
    )
    self._hand_body_ids = {
      body_id
      for body_id in range(self._model.nbody)
      if self._is_hand_descendant(body_id, hand_root)
    }

  def collides(self, arm_qpos: np.ndarray) -> bool:
    """Return whether ``arm_qpos`` produces an arm-to-hand contact."""
    self._qpos[self._arm_qpos_adr] = arm_qpos
    self._data.qpos[:] = self._qpos
    mujoco.mj_forward(self._model, self._data)
    for contact_index in range(self._data.ncon):
      contact = self._data.contact[contact_index]
      first = int(self._model.geom_bodyid[contact.geom1])
      second = int(self._model.geom_bodyid[contact.geom2])
      if (first in self._arm_body_ids and second in self._hand_body_ids) or (
        second in self._arm_body_ids and first in self._hand_body_ids
      ):
        return True
    return False

  def _is_hand_descendant(self, body_id: int, hand_root: int) -> bool:
    while body_id > 0:
      if body_id == hand_root:
        return True
      body_id = int(self._model.body_parentid[body_id])
    return False
