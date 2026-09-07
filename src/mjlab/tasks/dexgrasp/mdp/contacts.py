"""Native contact sensing for DexGrasp: instantaneous forces in Newton.

Force sensors report the net contact force of the last substep (``data.force``),
matching the reference which reads one sim step's contact. No history buffer, no
impulse scaling, no pad-to-parent folding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.sensor import ContactMatch, ContactSensorCfg

if TYPE_CHECKING:
  from mjlab.sensor import ContactSensor

__all__ = [
  "CONTACT_FORCE_THRESHOLD",
  "FORCE_CLIP",
  "FORCE_CLIP_THUMB",
  "HAND_TABLE_BODIES",
  "contact_force",
  "contact_flags",
  "force_clips",
  "pad_object_sensor",
  "link_object_sensor",
  "hand_table_sensor",
  "arm_any_sensor",
  "arm_table_sensor",
  "arm_object_sensor",
]

# Native force scale (N): reference impulse thresholds / its 0.01 s sim step.
CONTACT_FORCE_THRESHOLD = 1.0  # flag: 0.01 N s / 0.01 s
FORCE_CLIP = 10.0  # clip: 0.1 N s / 0.01 s
FORCE_CLIP_THUMB = 20.0  # thumb clip: 0.2 N s / 0.01 s

# Table-contact primary: every hand body with a collision geom (15 finger links,
# palm, 6 pads), so pressing a joint on the table is penalised, not just a pad.
HAND_TABLE_BODIES = rc.CONTACT_LINK_BODIES + ("R_hand_palm",) + rc.PAD_BODIES

_OBJECT = ContactMatch(mode="subtree", pattern="object", entity="object")
_TABLE = ContactMatch(mode="geom", pattern="table", entity="arena")


def contact_force(sensor: ContactSensor) -> torch.Tensor:
  """Net contact force of the last substep (B, P, 3) in Newton."""
  force = sensor.data.force
  assert force is not None
  return force


def contact_flags(
  sensor: ContactSensor, threshold: float = CONTACT_FORCE_THRESHOLD
) -> torch.Tensor:
  """Per-body contact flag (|F| > threshold) as float (B, P)."""
  return (contact_force(sensor).norm(dim=-1) > threshold).float()


def force_clips(body_names: tuple[str, ...]) -> list[float]:
  """Per-body force clip (N): thumb bodies 20, others 10."""
  return [FORCE_CLIP_THUMB if "thumb" in n else FORCE_CLIP for n in body_names]


def _hand_primary(bodies: tuple[str, ...]) -> ContactMatch:
  return ContactMatch(
    mode="body", pattern=tuple(f"robot/{rc.HAND_PREFIX}{b}" for b in bodies)
  )


def _arm_primary() -> ContactMatch:
  return ContactMatch(
    mode="body", pattern=tuple(f"robot/{b}" for b in rc.ARM_LINK_BODIES)
  )


def pad_object_sensor() -> ContactSensorCfg:
  """6 pad bodies vs the object: net force per pad (grasp affordance)."""
  return ContactSensorCfg(
    name="pad_object",
    primary=_hand_primary(rc.PAD_BODIES),
    secondary=_OBJECT,
    fields=("force",),
    reduce="netforce",
  )


def link_object_sensor() -> ContactSensorCfg:
  """15 finger links vs the object: contact flags for the observation."""
  return ContactSensorCfg(
    name="link_object",
    primary=_hand_primary(rc.CONTACT_LINK_BODIES),
    secondary=_OBJECT,
    fields=("found",),
    reduce="netforce",
  )


def hand_table_sensor() -> ContactSensorCfg:
  """22 hand bodies vs the table: net force per body (table penalty)."""
  return ContactSensorCfg(
    name="hand_table",
    primary=_hand_primary(HAND_TABLE_BODIES),
    secondary=_TABLE,
    fields=("force",),
    reduce="netforce",
  )


def arm_any_sensor() -> ContactSensorCfg:
  """6 arm links vs anything: contact flags (arm_collision)."""
  return ContactSensorCfg(
    name="arm_any",
    primary=_arm_primary(),
    secondary=None,
    fields=("found",),
    reduce="netforce",
  )


def arm_table_sensor() -> ContactSensorCfg:
  """6 arm links vs the table: net force per link."""
  return ContactSensorCfg(
    name="arm_table",
    primary=_arm_primary(),
    secondary=_TABLE,
    fields=("force",),
    reduce="netforce",
  )


def arm_object_sensor() -> ContactSensorCfg:
  """6 arm links vs the object: net force per link."""
  return ContactSensorCfg(
    name="arm_object",
    primary=_arm_primary(),
    secondary=_OBJECT,
    fields=("force",),
    reduce="netforce",
  )
