"""Evaluate a DexGrasp policy with a fixed post-grasp lifting motion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.objects.dexgrasp import object_constants as oc
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as rc
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.env_cfgs import (
  dexgrasp_ur5e_rh5dg2_env_cfg,
)
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.rl_cfg import (
  dexgrasp_teacher_ppo_runner_cfg,
)
from mjlab.tasks.dexgrasp.dexgrasp_env_cfg import ARM_MOUNT_Z
from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for teacher-policy lift evaluation."""

  checkpoint: Path
  """Local RSL-RL checkpoint to evaluate."""
  objects: tuple[str, ...] = oc.ROBUST_DEXGRASP_TRAIN_OBJECTS
  """Objects to evaluate; defaults to the complete training cohort."""
  num_envs: int = 128
  """Number of episodes per object."""
  grasp_steps: int = 70
  """Policy-control steps before the scripted lift."""
  lift_steps: int = 90
  """Control steps over which to interpolate the arm lift target."""
  lift_height: float = 0.15
  """Vertical grasp-center target displacement in metres."""
  success_height: float = 0.10
  """Minimum object vertical displacement required for success in metres."""
  device: str | None = None
  """Simulation device; defaults to CUDA when available."""
  output_file: Path | None = None
  """Optional JSON path for per-object success rates."""


def lift_targets(
  arm_qpos: np.ndarray,
  kin: ArmKinematics,
  lift_height: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Solve same-orientation arm targets raised by ``lift_height`` metres."""
  targets = arm_qpos.copy()
  reachable = np.zeros(len(arm_qpos), dtype=bool)
  for index, qpos in enumerate(arm_qpos):
    grasp_center = kin.fk_grasp_center_env(qpos)
    target = kin.arm_qpos_for_grasp_center(
      grasp_center[:3, 3] + np.array([0.0, 0.0, lift_height]),
      grasp_center[:3, :3],
      seed=qpos,
    )
    if target is not None and np.isfinite(target).all():
      targets[index] = target + 2.0 * np.pi * np.round((qpos - target) / (2.0 * np.pi))
      reachable[index] = True
  return targets, reachable


def run_object_evaluation(
  object_name: str,
  cfg: EvaluateConfig,
  policy: torch.nn.Module,
  device: str,
) -> dict[str, float]:
  """Run policy grasping followed by a deterministic vertical lift."""
  env_cfg = dexgrasp_ur5e_rh5dg2_env_cfg(
    object_name=object_name, non_uniform_sampling=False
  )
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.terminations = {}
  env_cfg.auto_reset = False
  env = ManagerBasedRlEnv(env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env)
  obs = vec_env.get_observations()

  with torch.no_grad():
    for _ in range(cfg.grasp_steps):
      obs, _, _, _ = vec_env.step(policy(obs))

  robot = env.scene["robot"]
  obj = env.scene["object"]
  n_arm = len(rc.ARM_JOINT_NAMES)
  arm_ids = [robot.joint_names.index(name) for name in rc.ARM_JOINT_NAMES]
  initial_arm = robot.data.joint_pos[:, arm_ids].cpu().numpy()
  initial_object_z = obj.data.root_link_pos_w[:, 2].clone()
  kin = ArmKinematics(mount_pos=(0.0, 0.0, ARM_MOUNT_Z))
  target_arm, reachable_np = lift_targets(initial_arm, kin, cfg.lift_height)
  target_arm_t = torch.as_tensor(target_arm, device=device, dtype=torch.float)
  initial_arm_t = torch.as_tensor(initial_arm, device=device, dtype=torch.float)
  reachable = torch.as_tensor(reachable_np, device=device)

  # Fingers stay under policy control through the lift (the reference keeps
  # squeezing); only the arm is scripted, ramping toward the raised target.
  with torch.no_grad():
    for step in range(1, cfg.lift_steps + 1):
      alpha = step / cfg.lift_steps
      desired = torch.lerp(initial_arm_t, target_arm_t, alpha)
      current = robot.data.joint_pos[:, arm_ids]
      actions = policy(obs).clone()
      actions[:, :n_arm] = ((desired - current) / rc.ACTION_SCALE_ARM).clamp(-1.0, 1.0)
      obs, _, _, _ = vec_env.step(actions)

  # Final height, not peak: a transient bounce must not count as a hold.
  final_gain = obj.data.root_link_pos_w[:, 2] - initial_object_z
  success = reachable & (final_gain > cfg.success_height)
  vec_env.close()
  return {
    "success": float(success.float().mean()),
    "reachable": float(reachable.float().mean()),
    "mean_gain": float(final_gain[reachable].mean()) if reachable.any() else 0.0,
  }


def run_evaluate(cfg: EvaluateConfig) -> dict[str, float]:
  """Load a checkpoint and report lift success for the requested objects."""
  if not cfg.objects:
    raise ValueError("At least one evaluation object is required")
  unknown = set(cfg.objects) - set(oc.ROBUST_DEXGRASP_TRAIN_OBJECTS)
  if unknown:
    raise ValueError(f"Unknown evaluation objects: {sorted(unknown)}")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  agent_cfg = dexgrasp_teacher_ppo_runner_cfg()
  bootstrap_cfg = dexgrasp_ur5e_rh5dg2_env_cfg(
    object_name=cfg.objects[0], non_uniform_sampling=False
  )
  bootstrap_cfg.scene.num_envs = 1
  bootstrap_cfg.terminations = {}
  bootstrap_cfg.auto_reset = False
  bootstrap_env = ManagerBasedRlEnv(bootstrap_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(bootstrap_env)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
  runner.load(str(cfg.checkpoint), map_location=device)
  policy = runner.get_inference_policy(device=device)
  vec_env.close()

  details = {
    name: run_object_evaluation(name, cfg, policy, device) for name in cfg.objects
  }
  metrics = {name: d["success"] for name, d in details.items()}
  for name, d in sorted(details.items(), key=lambda kv: kv[1]["success"]):
    print(
      f"{name:<22} success {d['success']:5.1%}  "
      f"reachable {d['reachable']:5.1%}  mean_gain {d['mean_gain']:+.3f} m"
    )
  overall = sum(metrics.values()) / len(metrics)
  print(f"{'OVERALL':<22} success {overall:5.1%}")
  if cfg.output_file is not None:
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(json.dumps(details, indent=2) + "\n")
  return metrics


def main() -> None:
  run_evaluate(tyro.cli(EvaluateConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
