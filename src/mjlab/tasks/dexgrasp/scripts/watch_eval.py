"""Evaluate a training run's checkpoints as they appear and log them to W&B.

Runs beside ``train`` on the same machine: one eval env is built once, every
new ``model_<iter>.pt`` is scored with the scripted lift + hold protocol on a
fixed pose set, for each finger mode, and the rows go to a JSON file and a
W&B run named ``<run_dir>-eval``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.dexgrasp.config.ur5e_rh5dg2.rl_cfg import (
  dexgrasp_teacher_ppo_runner_cfg,
)
from mjlab.tasks.dexgrasp.scripts.evaluate import (
  EvaluateConfig,
  FingerMode,
  build_eval_env,
  evaluate_policy,
  format_metrics,
  load_policy,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class WatchConfig:
  """Configuration for the checkpoint watcher."""

  run_dir: Path
  """Training log directory containing ``model_*.pt`` files."""
  object_name: str = "potted_meat_can"
  """Object to evaluate on."""
  num_envs: int = 256
  """Episodes per checkpoint and finger mode."""
  seed: int = 0
  """Pose seed shared by every checkpoint so numbers are pose-matched."""
  finger_modes: tuple[FingerMode, ...] = ("live_policy", "frozen_delta")
  """Finger control modes to score."""
  grasp_steps: int = 70
  lift_steps: int = 90
  hold_steps: int = 25
  poll_seconds: float = 60.0
  """Seconds between checks for new checkpoints."""
  max_idle_seconds: float = 3600.0
  """Exit after this long without a new checkpoint."""
  settle_seconds: float = 15.0
  """Skip checkpoints modified more recently than this (still being written)."""
  wandb_project: str | None = "mjlab-dexgrasp"
  """W&B project; None disables logging."""
  wandb_run_name: str | None = None
  """W&B run name; defaults to ``<run_dir name>-eval``."""
  output_file: Path | None = None
  """JSON rows; defaults to ``<run_dir>/eval/watch_eval.json``."""
  device: str | None = None


def ckpt_step(path: Path) -> int:
  return int(path.stem.split("_")[1])


def main() -> None:
  cfg = tyro.cli(WatchConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  out = cfg.output_file or cfg.run_dir / "eval" / "watch_eval.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  rows: list[dict] = json.loads(out.read_text()) if out.exists() else []
  done = {row["checkpoint"] for row in rows}

  wandb_run = None
  if cfg.wandb_project is not None:
    import wandb

    wandb_run = wandb.init(
      project=cfg.wandb_project,
      name=cfg.wandb_run_name or f"{cfg.run_dir.name}-eval",
      config=asdict(cfg) | {"run_dir": str(cfg.run_dir)},
      dir=str(cfg.run_dir),
    )
    wandb.define_metric("Eval/checkpoint")
    wandb.define_metric("Eval/*", step_metric="Eval/checkpoint")

  env, vec_env = build_eval_env(cfg.object_name, cfg.num_envs, cfg.seed, device)
  runner = MjlabOnPolicyRunner(
    vec_env, asdict(dexgrasp_teacher_ppo_runner_cfg()), device=device
  )
  idle_since = time.time()
  print(f"Watching {cfg.run_dir} ({len(done)} checkpoints already scored)", flush=True)
  while True:
    now = time.time()
    pending = sorted(
      (
        p
        for p in cfg.run_dir.glob("model_*.pt")
        if ckpt_step(p) not in done and now - p.stat().st_mtime > cfg.settle_seconds
      ),
      key=ckpt_step,
    )
    if not pending:
      if now - idle_since > cfg.max_idle_seconds:
        print("No new checkpoint; exiting.", flush=True)
        break
      time.sleep(cfg.poll_seconds)
      continue
    for ck in pending:
      step = ckpt_step(ck)
      try:
        policy = load_policy(runner, ck, device)
      except Exception as exc:  # partially written file: retry next poll
        print(f"[WARN] could not load {ck.name}: {exc}", flush=True)
        continue
      row: dict = {"checkpoint": step}
      log: dict[str, float] = {"Eval/checkpoint": float(step)}
      for mode in cfg.finger_modes:
        eval_cfg = EvaluateConfig(
          checkpoint=ck,
          objects=(cfg.object_name,),
          num_envs=cfg.num_envs,
          grasp_steps=cfg.grasp_steps,
          lift_steps=cfg.lift_steps,
          hold_steps=cfg.hold_steps,
          finger_mode=mode,
          seed=cfg.seed,
          device=device,
        )
        metrics = evaluate_policy(env, vec_env, policy, eval_cfg)
        row[mode] = metrics
        log.update({f"Eval/{mode}/{k}": v for k, v in metrics.items()})
        print(format_metrics(f"model_{step} {mode}", metrics), flush=True)
      rows.append(row)
      done.add(step)
      out.write_text(json.dumps(rows, indent=1) + "\n")
      if wandb_run is not None:
        wandb_run.log(log)
      idle_since = time.time()
  vec_env.close()
  if wandb_run is not None:
    wandb_run.finish()


if __name__ == "__main__":
  main()
