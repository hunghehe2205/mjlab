from mjlab.tasks.registry import register_mjlab_task

from .view_env_cfg import ur5e_rh5dg2_ppo_runner_cfg, ur5e_rh5dg2_view_env_cfg

register_mjlab_task(
  task_id="Mjlab-View-Ur5e-Rh5dg2",
  env_cfg=ur5e_rh5dg2_view_env_cfg(),
  play_env_cfg=ur5e_rh5dg2_view_env_cfg(play=True),
  rl_cfg=ur5e_rh5dg2_ppo_runner_cfg(),
)
