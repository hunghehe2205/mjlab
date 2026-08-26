from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  PHASE1_OBJECT_NAMES,
  SKELETON_OBJECT,
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from .rl_cfg import dexgrasp_teacher_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-DexGrasp-UR5eRH5DG2",
  env_cfg=dexgrasp_ur5e_rh5dg2_env_cfg(),
  play_env_cfg=dexgrasp_ur5e_rh5dg2_env_cfg(play=True),
  rl_cfg=dexgrasp_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-DexGrasp-UR5eRH5DG2-Single",
  env_cfg=dexgrasp_ur5e_rh5dg2_env_cfg(object_name=SKELETON_OBJECT),
  play_env_cfg=dexgrasp_ur5e_rh5dg2_env_cfg(play=True),
  rl_cfg=dexgrasp_teacher_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-DexGrasp-UR5eRH5DG2-Phase1",
  env_cfg=dexgrasp_ur5e_rh5dg2_env_cfg(object_names=PHASE1_OBJECT_NAMES),
  play_env_cfg=dexgrasp_ur5e_rh5dg2_env_cfg(play=True),
  rl_cfg=dexgrasp_teacher_ppo_runner_cfg(),
)
