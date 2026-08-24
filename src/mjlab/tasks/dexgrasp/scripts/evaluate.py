"""Evaluate a DexGrasp policy with a fixed post-grasp lifting motion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

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
) -> float:
  """Run policy grasping followed by a deterministic vertical lift."""
  env_cfg = dexgrasp_ur5e_rh5dg2_env_cfg(object_name=object_name)
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
  arm_ids = [robot.joint_names.index(name) for name in rc.ARM_JOINT_NAMES]
  initial_arm = robot.data.joint_pos[:, arm_ids].cpu().numpy()
  initial_object_z = obj.data.root_link_pos_w[:, 2].clone()
  kin = ArmKinematics(mount_pos=(0.0, 0.0, ARM_MOUNT_Z))
  target_arm, reachable_np = lift_targets(initial_arm, kin, cfg.lift_height)
  target_arm_t = torch.as_tensor(target_arm, device=device, dtype=torch.float)
  reachable = torch.as_tensor(reachable_np, device=device)
  peak_object_z = initial_object_z.clone()

  for step in range(1, cfg.lift_steps + 1):
    alpha = step / cfg.lift_steps
    desired = torch.lerp(
      torch.as_tensor(initial_arm, device=device, dtype=torch.float),
      target_arm_t,
      alpha,
    )
    current = robot.data.joint_pos[:, arm_ids]
    actions = torch.zeros(cfg.num_envs, len(rc.ALL_JOINT_NAMES), device=device)
    actions[:, : len(rc.ARM_JOINT_NAMES)] = (desired - current) / rc.ACTION_SCALE_ARM
    actions[:, : len(rc.ARM_JOINT_NAMES)].clamp_(-1.0, 1.0)
    vec_env.step(actions)
    peak_object_z = torch.maximum(peak_object_z, obj.data.root_link_pos_w[:, 2])

  success = reachable & (peak_object_z - initial_object_z > cfg.success_height)
  vec_env.close()
  return float(success.float().mean())


def run_evaluate(cfg: EvaluateConfig) -> dict[str, float]:
  """Load a checkpoint and report lift success for every Phase-1 object."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  agent_cfg = dexgrasp_teacher_ppo_runner_cfg()
  bootstrap_cfg = dexgrasp_ur5e_rh5dg2_env_cfg()
  bootstrap_cfg.scene.num_envs = 1
  bootstrap_cfg.terminations = {}
  bootstrap_cfg.auto_reset = False
  bootstrap_env = ManagerBasedRlEnv(bootstrap_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(bootstrap_env)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
  runner.load(str(cfg.checkpoint), map_location=device)
  policy = runner.get_inference_policy(device=device)
  vec_env.close()

  metrics = {
    name: run_object_evaluation(name, cfg, policy, device) for name in oc.PHASE1_OBJECTS
  }
  for name, success_rate in metrics.items():
    print(f"{name}: {success_rate:.1%} lift success")
  if cfg.output_file is not None:
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(json.dumps(metrics, indent=2) + "\n")
  return metrics


def main() -> None:
  run_evaluate(tyro.cli(EvaluateConfig))


if __name__ == "__main__":
  main()
