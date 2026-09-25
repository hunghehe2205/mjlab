"""RobustDexGrasp lift test: grasp with the policy, then raise the arm by script.

Success is a rise above LIFT_HEIGHT at the end of the episode, over uniformly
sampled placements. Load a checkpoint file or a W&B run:

  uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.evaluate --checkpoint model.pt
  uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.evaluate --wandb-run e/p/id
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import GRASP_TIME
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp import rewards, signals, terminations
from mjlab.tasks.ur5e_rh5dg2.grasp.teacher_env_cfg import (
  teacher_env_cfg,
  teacher_ppo_cfg,
)
from mjlab.utils.os import get_wandb_checkpoint_path


def lift_test_env(num_envs: int, device: str) -> RslRlVecEnvWrapper:
  cfg = teacher_env_cfg(lift_test=True)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=teacher_ppo_cfg().clip_actions)


def run_lift_test(
  vec_env: RslRlVecEnvWrapper, policy: Callable, seed: int = 0
) -> dict[str, float]:
  """One grasp-then-lift episode per env with deterministic actions."""
  env = vec_env.unwrapped
  device = torch.device(env.device)
  devices = [device.index or 0] if device.type == "cuda" else []
  grasp_steps = round(GRASP_TIME / env.step_dt)
  n = env.num_envs
  failed = torch.zeros(n, dtype=torch.bool, device=device)
  fingers = displacement = load = torch.zeros(n, device=device)
  # Same placements and delays for every call, without disturbing training RNG.
  with torch.random.fork_rng(devices=devices), torch.no_grad():
    torch.manual_seed(seed)
    obs, _ = vec_env.reset()
    # Stop one step short of the time limit so auto-reset keeps the final state.
    for step in range(env.max_episode_length - 1):
      obs, _, dones, _ = vec_env.step(policy(obs))
      failed |= dones.bool()
      if step == grasp_steps - 1:
        fingers = signals.fingers_in_contact(env)
        displacement = rewards.object_displacement(env)
        load = signals.table_load(env)
  success = terminations.lifted(env) * ~failed
  return {
    "success_rate": success.mean().item(),
    "mean_lift_m": signals.lift_height(env).mean().item(),
    "early_terminations": failed.float().mean().item(),
    "grasp_fingers_in_contact": fingers.mean().item(),
    "grasp_object_displacement_m": displacement.mean().item(),
    "grasp_table_load_n": load.mean().item(),
  }


def evaluate(
  checkpoint: Path, num_envs: int, device: str, seed: int = 0
) -> dict[str, float]:
  vec_env = lift_test_env(num_envs, device)
  try:
    runner = MjlabOnPolicyRunner(vec_env, asdict(teacher_ppo_cfg()), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    return {"episodes": num_envs, **run_lift_test(vec_env, policy, seed)}
  finally:
    vec_env.unwrapped.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint", type=Path)
  source.add_argument("--wandb-run")
  parser.add_argument("--wandb-checkpoint", help="e.g. model_1000.pt; latest if unset")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()
  checkpoint = args.checkpoint
  if checkpoint is None:
    root = Path("logs/rsl_rl") / teacher_ppo_cfg().experiment_name
    checkpoint, _ = get_wandb_checkpoint_path(
      root.resolve(), Path(args.wandb_run), args.wandb_checkpoint
    )
  result = evaluate(checkpoint, args.num_envs, args.device, args.seed)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
