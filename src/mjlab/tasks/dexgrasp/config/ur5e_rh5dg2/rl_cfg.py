"""PPO runner config for the DexGrasp teacher (RobustDexGrasp settings)."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def dexgrasp_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  # Obs normalization is not in the reference. Raw obs std spans 0.02 (af_vec)
  # to 1.0 (joint_pos) and the critic's value loss sat at 40-700 for 1600
  # iterations without it (documents/08 §6).
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="lrelu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
        "std_range": (0.2, 1.0e6),
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="lrelu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=0.5,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=4,
      # Reference: 4 minibatches (88 env x 70 / 4 = 1540 samples), 16 updates per
      # rollout. 16 minibatches made 64 adaptive-LR adjustments per rollout and the
      # LR swung between its 1e-2 ceiling and 1e-5 floor within ~20 iterations in
      # every 352-env run; fixed 5e-4 removes that confound.
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="fixed",
      gamma=0.996,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=0.5,
    ),
    experiment_name="dexgrasp_teacher_ur5e_rh5dg2",
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    save_interval=100,
    num_steps_per_env=70,  # Full-episode rollout per update (reference).
    max_iterations=10_000,
  )
