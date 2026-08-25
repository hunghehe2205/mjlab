"""DexGrasp Phase 1 object assets."""

from mjlab.asset_zoo.objects.dexgrasp.object_constants import (
  OBJECT_NAMES,
  PHASE1_OBJECTS,
  ROBUST_DEXGRASP_BASELINE_NUM_ENVS,
  ROBUST_DEXGRASP_TRAIN_OBJECTS,
  DexGraspObject,
  get_mesh_object_spec,
  get_phase1_variant_cfg,
  get_robustdexgrasp_variant_cfg,
)

__all__ = [
  "OBJECT_NAMES",
  "PHASE1_OBJECTS",
  "ROBUST_DEXGRASP_BASELINE_NUM_ENVS",
  "ROBUST_DEXGRASP_TRAIN_OBJECTS",
  "DexGraspObject",
  "get_mesh_object_spec",
  "get_phase1_variant_cfg",
  "get_robustdexgrasp_variant_cfg",
]
