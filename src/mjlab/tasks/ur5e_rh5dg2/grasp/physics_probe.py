"""Replay close/lift waypoints through the same actuators used by the teacher.

Run with ``uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe``.
The probe uses no object attachment or state writes after initialization.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.ur5e_rh5dg2.grasp.constants import (
  ARM_ACTION_SCALE,
  CLOSED_HAND,
  FINGERS,
  FORCE_THRESHOLD,
  GRASP_ARM,
  HAND_ACTION_SCALE,
  HOLD_TIME,
  LIFT_ARM,
  LIFT_HEIGHT,
  OBJECT_POS,
  OPEN_HAND,
  PREGRASP_ARM,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspAction
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import (
  lift_height,
  normal_force,
  object_speed,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.terminations import LiftSuccess
from mjlab.tasks.ur5e_rh5dg2.grasp.teacher_env_cfg import teacher_env_cfg


@dataclass
class ProbeResult:
  backend: str
  device: str
  success: bool
  initial_penetration_m: float
  final_lift_m: float
  hold_seconds: float
  object_speed_m_s: float
  contact_force_n: float


def scripted_target(time: float) -> np.ndarray:
  """Approach 0-2 s, close 2-4 s, lift 5-7 s, then hold."""
  approach = np.clip(time / 2.0, 0.0, 1.0)
  close = np.clip((time - 2.0) / 2.0, 0.0, 1.0)
  lift = np.clip((time - 5.0) / 2.0, 0.0, 1.0)
  arm = (
    np.array(PREGRASP_ARM)
    + (np.array(GRASP_ARM) - PREGRASP_ARM) * approach
    + (np.array(LIFT_ARM) - GRASP_ARM) * lift
  )
  hand = np.array(OPEN_HAND) + (np.array(CLOSED_HAND) - OPEN_HAND) * close
  return np.concatenate([arm, hand])


def initial_penetration(model: mujoco.MjModel) -> float:
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return max(0.0, -min((c.dist for c in data.contact), default=0.0))


def run_native_probe() -> ProbeResult:
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, device="cpu").compile()
  cfg.sim.mujoco.apply(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  initial_joints = cfg.scene.entities["robot"].init_state.joint_pos
  assert initial_joints is not None
  joint_names = tuple(initial_joints)
  joint_ids = np.array([model.joint(f"robot/{name}").id for name in joint_names])
  qpos_ids = model.jnt_qposadr[joint_ids]
  ctrl_ids = np.array(
    [int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in joint_ids]
  )
  limits = model.jnt_range[joint_ids]
  initial_q = data.qpos[qpos_ids]
  if np.any(initial_q < limits[:, 0]) or np.any(initial_q > limits[:, 1]):
    raise ValueError("Pre-grasp joint pose exceeds hard joint limits.")
  penetration = initial_penetration(model)
  obj = model.joint("object/free_joint")
  qadr, vadr = int(obj.qposadr[0]), int(obj.dofadr[0])
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  required = math.ceil(HOLD_TIME / step_dt)
  count = 0
  force = np.zeros(6)
  rise = speed = peak_force = 0.0
  for step in range(round(12.0 / step_dt)):
    goal = scripted_target(step * step_dt)
    scale = np.array([ARM_ACTION_SCALE] * 6 + [HAND_ACTION_SCALE] * 18)
    q = data.qpos[qpos_ids]
    target = q + np.clip(goal - q, -scale, scale)
    data.ctrl[ctrl_ids] = np.clip(target, limits[:, 0], limits[:, 1])
    mujoco.mj_step(model, data, nstep=cfg.decimation)
    mujoco.mj_forward(model, data)
    touching: set[str] = set()
    table_force = 0.0
    peak_force = 0.0
    for i in range(data.ncon):
      contact = data.contact[i]
      names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
      if "object/collision" not in names:
        continue
      mujoco.mj_contactForce(model, data, i, force)
      if "props/table_geom" in names:
        table_force = max(table_force, force[0])
      for finger in FINGERS:
        if any(name.startswith(f"robot/R_{finger}_") for name in names):
          peak_force = max(peak_force, force[0])
          if force[0] > FORCE_THRESHOLD:
            touching.add(finger)
    rise = float(data.qpos[qadr + 2] - OBJECT_POS[2])
    speed = float(np.linalg.norm(data.qvel[vadr : vadr + 3]))
    angular_speed = np.linalg.norm(data.qvel[vadr + 3 : vadr + 6])
    stable = (
      rise >= LIFT_HEIGHT
      and speed <= 0.05
      and angular_speed <= 1.0
      and len(touching) >= 2
      and table_force < FORCE_THRESHOLD
    )
    count = count + 1 if stable else 0
    if count >= required:
      break
  return ProbeResult(
    "native",
    "cpu",
    count >= required and penetration < 1e-5,
    penetration,
    rise,
    count * step_dt,
    speed,
    float(peak_force),
  )


def run_warp_probe(device: str = "cpu") -> ProbeResult:
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    ncon = int(wp.to_torch(env.sim.wp_data.nacon)[0].item())
    distances = wp.to_torch(env.sim.wp_data.contact.dist)[:ncon]
    penetration = max(0.0, -distances.min().item()) if ncon else 0.0
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, GraspAction)
    for step in range(env.max_episode_length):
      goal = torch.tensor(
        scripted_target(step * env.step_dt), device=device, dtype=torch.float32
      )[None, :]
      q = env.scene["robot"].data.joint_pos[:, action.target_ids]
      _, _, terminated, truncated, _ = env.step((goal - q) / action.scale)
      if (terminated | truncated).any():
        break
    success = env.termination_manager.get_term("success").item()
    hold = env.termination_manager.get_term_cfg("success").func
    assert isinstance(hold, LiftSuccess)
    return ProbeResult(
      "warp",
      device,
      bool(success) and penetration < 1e-5,
      penetration,
      lift_height(env).item(),
      hold.count.item() * env.step_dt,
      object_speed(env).item(),
      normal_force(env, "hand_object").max().item(),
    )
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("native", "warp"), default="warp")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = (
    run_native_probe() if args.backend == "native" else run_warp_probe(args.device)
  )
  report = json.dumps(asdict(result), indent=2)
  print(report)
  if args.output is not None:
    args.output.write_text(report + "\n")
  if not result.success:
    raise SystemExit("Physics acceptance probe failed; inspect the reported metrics.")


if __name__ == "__main__":
  main()
