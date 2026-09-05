"""Evaluate a DexGrasp policy: policy grasp, scripted vertical lift, timed hold."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

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
from mjlab.tasks.dexgrasp.mdp.metrics import object_tilt_angle
from mjlab.tasks.dexgrasp.pregrasp.kinematics import ArmKinematics
from mjlab.utils.torch import configure_torch_backends

FingerMode = Literal["live_policy", "frozen_delta"]
TIPPED_ANGLE = math.radians(45.0)
THUMB_YAW_JOINT = "R_thumb_yaw_joint"


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
  hold_steps: int = 25
  """Control steps the arm holds the lifted pose (25 = 5 s, the paper's hold)."""
  lift_height: float = 0.15
  """Vertical grasp-center target displacement in metres."""
  success_height: float = 0.10
  """Minimum object rise above its reset height for lift/hold success."""
  finger_mode: FingerMode = "live_policy"
  """Finger command during lift and hold: keep querying the policy (deployment,
  paper B.3) or repeat the last grasp residual (reference quantitative_eval)."""
  seed: int | None = None
  """Object-pose seed; fix it to compare checkpoints on identical poses."""
  device: str | None = None
  """Simulation device; defaults to CUDA when available."""
  output_file: Path | None = None
  """Optional JSON path for per-object metrics."""


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


def build_eval_env(
  object_name: str, num_envs: int, seed: int | None, device: str
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
  """Single-object eval env: uniform poses, no terminations, manual reset."""
  env_cfg = dexgrasp_ur5e_rh5dg2_env_cfg(
    object_name=object_name, non_uniform_sampling=False
  )
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  env_cfg.terminations = {}
  env_cfg.auto_reset = False
  env = ManagerBasedRlEnv(env_cfg, device=device)
  return env, RslRlVecEnvWrapper(env)


def seed_reset_poses(env: ManagerBasedRlEnv, seed: int) -> None:
  """Reseed the pre-grasp reset event so the next reset draws a fixed pose set."""
  term = env.event_manager.get_term_cfg("reset_grasp_pose").func
  term._rng = np.random.default_rng(seed)


def evaluate_policy(
  env: ManagerBasedRlEnv,
  vec_env: RslRlVecEnvWrapper,
  policy: torch.nn.Module,
  cfg: EvaluateConfig,
) -> dict[str, float]:
  """Policy grasp, scripted vertical lift, then hold; heights are from reset.

  ``success`` keeps the legacy definition (rise over the end-of-grasp height at
  the end of the lift) so old sweeps stay comparable; ``lift_success`` and
  ``hold_success`` measure from the reset height like the reference.
  """
  if cfg.seed is not None:
    seed_reset_poses(env, cfg.seed)
  obs, _ = vec_env.reset()
  robot, obj = env.scene["robot"], env.scene["object"]
  device = env.device
  n_arm = len(rc.ARM_JOINT_NAMES)
  arm_ids = [robot.joint_names.index(name) for name in rc.ARM_JOINT_NAMES]
  thumb_id = robot.joint_names.index(THUMB_YAW_JOINT)
  live = cfg.finger_mode == "live_policy"
  z_reset = obj.data.root_link_pos_w[:, 2].clone()
  pos_reset = obj.data.root_link_pos_w.clone()
  last = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)

  with torch.no_grad():
    for _ in range(cfg.grasp_steps):
      last = policy(obs).clone()
      obs, _, _, _ = vec_env.step(last)
    z_grasp = obj.data.root_link_pos_w[:, 2].clone()
    grasp_disp = (obj.data.root_link_pos_w - pos_reset).norm(dim=-1)
    thumb_yaw = robot.data.joint_pos[:, thumb_id].clone()
    tipped_grasp = object_tilt_angle(obj.data.root_link_quat_w) > TIPPED_ANGLE

    initial_arm = robot.data.joint_pos[:, arm_ids].clone()
    kin = ArmKinematics(mount_pos=(0.0, 0.0, ARM_MOUNT_Z))
    target_np, reachable_np = lift_targets(
      initial_arm.cpu().numpy(), kin, cfg.lift_height
    )
    target = torch.as_tensor(target_np, device=device, dtype=torch.float)
    reachable = torch.as_tensor(reachable_np, device=device)

    # Arm target is the interpolated pose itself, as in the reference; fingers
    # follow ``finger_mode``.
    for step in range(1, cfg.lift_steps + 1):
      desired = torch.lerp(initial_arm, target, step / cfg.lift_steps)
      actions = policy(obs).clone() if live else last.clone()
      actions[:, :n_arm] = (
        desired - robot.data.joint_pos[:, arm_ids]
      ) / rc.ACTION_SCALE_ARM
      obs, _, _, _ = vec_env.step(actions)
    z_lift = obj.data.root_link_pos_w[:, 2].clone()
    lifted = reachable & (z_lift - z_reset > cfg.success_height)
    legacy = reachable & (z_lift - z_grasp > cfg.success_height)

    held = lifted.clone()
    for _ in range(cfg.hold_steps):
      actions = policy(obs).clone() if live else last.clone()
      actions[:, :n_arm] = (
        target - robot.data.joint_pos[:, arm_ids]
      ) / rc.ACTION_SCALE_ARM
      obs, _, _, _ = vec_env.step(actions)
      held &= obj.data.root_link_pos_w[:, 2] - z_reset > cfg.success_height
    final_gain = obj.data.root_link_pos_w[:, 2] - z_reset
    final_tilt = object_tilt_angle(obj.data.root_link_quat_w)
    track_err = (robot.data.joint_pos[:, arm_ids] - target).abs().amax(dim=1)

  has = bool(reachable.any())

  def frac(mask: torch.Tensor) -> float:
    return float(mask.float().mean())

  def over_reachable(values: torch.Tensor) -> float:
    return float(values[reachable].float().mean()) if has else 0.0

  return {
    "success": frac(legacy),
    "lift_success": frac(lifted),
    "hold_success": frac(held),
    "reachable": frac(reachable),
    "mean_gain": over_reachable(final_gain),
    "track_err": over_reachable(track_err),
    "frac_drop": over_reachable(final_gain < -0.01),
    "frac_tipped": over_reachable(final_tilt > TIPPED_ANGLE),
    "tipped_after_grasp": frac(tipped_grasp),
    "grasp_displacement_cm": 100.0 * float(grasp_disp.mean()),
    "thumb_yaw": float(thumb_yaw.mean()),
  }


def load_policy(
  runner: MjlabOnPolicyRunner, checkpoint: Path, device: str
) -> torch.nn.Module:
  runner.load(str(checkpoint), map_location=device)
  return runner.get_inference_policy(device=device)


def run_object_evaluation(
  object_name: str, cfg: EvaluateConfig, device: str
) -> dict[str, float]:
  """Build an env for one object, load the checkpoint and run the protocol."""
  env, vec_env = build_eval_env(object_name, cfg.num_envs, cfg.seed, device)
  runner = MjlabOnPolicyRunner(
    vec_env, asdict(dexgrasp_teacher_ppo_runner_cfg()), device=device
  )
  policy = load_policy(runner, cfg.checkpoint, device)
  metrics = evaluate_policy(env, vec_env, policy, cfg)
  vec_env.close()
  return metrics


def format_metrics(name: str, d: dict[str, float]) -> str:
  return (
    f"{name:<22} lift {d['lift_success']:5.1%}  hold {d['hold_success']:5.1%}  "
    f"legacy {d['success']:5.1%}  tipped {d['frac_tipped']:5.1%}  "
    f"disp {d['grasp_displacement_cm']:4.1f}cm  yaw {d['thumb_yaw']:.2f}  "
    f"track_err {d['track_err']:.3f}"
  )


def run_evaluate(cfg: EvaluateConfig) -> dict[str, float]:
  """Load a checkpoint and report lift/hold success for the requested objects."""
  if not cfg.objects:
    raise ValueError("At least one evaluation object is required")
  unknown = set(cfg.objects) - set(oc.ROBUST_DEXGRASP_TRAIN_OBJECTS)
  if unknown:
    raise ValueError(f"Unknown evaluation objects: {sorted(unknown)}")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  details = {name: run_object_evaluation(name, cfg, device) for name in cfg.objects}
  metrics = {name: d["success"] for name, d in details.items()}
  for name, d in sorted(details.items(), key=lambda kv: kv[1]["hold_success"]):
    print(format_metrics(name, d))
  overall_hold = sum(d["hold_success"] for d in details.values()) / len(details)
  overall = sum(metrics.values()) / len(metrics)
  print(f"{'OVERALL':<22} lift(legacy) {overall:5.1%}  hold {overall_hold:5.1%}")
  if cfg.output_file is not None:
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(json.dumps(details, indent=2) + "\n")
  return metrics


def main() -> None:
  run_evaluate(tyro.cli(EvaluateConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
