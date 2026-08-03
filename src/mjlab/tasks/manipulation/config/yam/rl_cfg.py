from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

_VISION_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def yam_lift_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_lift_cube",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


def yam_pick_place_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Raised from the state-lift default of 0.005. Pick-and-place chains six
      # stages, so its exploration phase is much longer than lift's two, and the
      # failure mode the lift runs already showed is the action std collapsing
      # before the last stage is ever discovered.
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_pick_place_cube",
    save_interval=100,
    num_steps_per_env=24,
    # The joint_vel_hinge curriculum's final stage lands at iteration 4000 and
    # the difficulty curriculum's at 3000, so the budget has to clear both with
    # room to converge underneath them.
    max_iterations=8_000,
  )


def yam_lift_cube_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = _VISION_CNN_CFG
  class_name = _VISION_MODEL_CLS
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Raised from 0.005 (the state-based default) to keep exploration alive
      # longer. From pixels the policy is slow to discover the grasp-and-lift,
      # and the default entropy lets the action std collapse into the reach-only
      # optimum before it gets there.
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_lift_cube_vision",
    save_interval=100,
    num_steps_per_env=24,
    # Raised from 3000 so the delayed joint_vel_hinge curriculum's final stage
    # (iteration 4000) fires, leaving ~1000 iterations to converge under it.
    max_iterations=5_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def yam_lift_cube_rgbd_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  # RGB+D reuses the vision runner verbatim (same CNN, hyperparameters, and the
  # vision training fix); only the experiment name differs so its logs and
  # checkpoints land in a separate directory.
  cfg = yam_lift_cube_vision_ppo_runner_cfg()
  cfg.experiment_name = "yam_lift_cube_rgbd"
  return cfg


def yam_multi_cube_seg_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN_CFG,
      class_name=_VISION_MODEL_CLS,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN_CFG,
      class_name=_VISION_MODEL_CLS,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="yam_multi_cube_seg",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )
