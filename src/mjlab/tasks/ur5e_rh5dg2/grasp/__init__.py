from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.ur5e_rh5dg2.grasp.teacher_env_cfg import (
  teacher_env_cfg,
  teacher_ppo_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-Teacher-Ur5e-Rh5dg2",
  env_cfg=teacher_env_cfg(),
  play_env_cfg=teacher_env_cfg(play=True),
  rl_cfg=teacher_ppo_cfg(),
)
