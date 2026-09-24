"""Replay a scripted grasp from the sampled nominal pre-grasp.

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
  HAND_ACTION_SCALE,
  HAND_CENTER,
  HOLD_TIME,
  LIFT_HEIGHT,
  OBJECT_POS,
  OPEN_HAND,
  STANDOFF,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.actions import GraspAction
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.events import PregraspReset
from mjlab.tasks.ur5e_rh5dg2.grasp.mdp.signals import (
  lift_height,
  normal_force,
  object_speed,
  stable_hold,
)
from mjlab.tasks.ur5e_rh5dg2.grasp.pregrasp import (
  PregraspSolver,
  box_surface_points,
  yaw_quat,
)
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
  rollout: RolloutAudit


@dataclass
class RolloutAudit:
  """Contact-depth maxima sampled at the 20 Hz control boundaries.

  The 3 mm contact-depth cap is a numerical regression guard for this soft-contact
  primitive, not a claim of rigid contact or a bound on unsampled substeps.
  """

  hand_object_penetration_m: float = 0.0
  object_table_penetration_m: float = 0.0
  undesired_penetration_m: float = 0.0
  approach_object_displacement_m: float = 0.0
  approach_hand_object_penetration_m: float = 0.0
  peak_hand_force_n: float = 0.0

  @property
  def passed(self) -> bool:
    return (
      self.hand_object_penetration_m <= 0.003
      and self.object_table_penetration_m <= 0.003
      and self.undesired_penetration_m <= 0.001
      and self.approach_object_displacement_m <= 0.001
      and self.approach_hand_object_penetration_m <= 1e-5
    )

  def update(
    self,
    model: mujoco.MjModel,
    geom: np.ndarray,
    distance: np.ndarray,
    object_pos: np.ndarray,
    approach: bool,
    hand_force: float,
  ) -> None:
    object_id = model.geom("object/collision").id
    table_id = model.geom("props/table_geom").id
    hand_ids = np.array(
      [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith(
          ("robot/R_", "robot/right_hand")
        )
      ]
    )
    robot_ids = np.array(
      [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith("robot/")
      ]
    )
    obj = (geom == object_id).any(axis=-1)
    table = (geom == table_id).any(axis=-1)
    hand = np.isin(geom, hand_ids).any(axis=-1)
    robot = np.isin(geom, robot_ids)
    # Body ownership also covers the UR5 collision geoms that have no names.
    undesired = robot.any(axis=-1) & ~(obj & hand)

    def depth(mask: np.ndarray) -> float:
      return float(max(0.0, -distance[mask].min(initial=0.0)))

    hand_depth = depth(obj & hand)
    self.hand_object_penetration_m = max(self.hand_object_penetration_m, hand_depth)
    self.object_table_penetration_m = max(
      self.object_table_penetration_m, depth(obj & table)
    )
    self.undesired_penetration_m = max(self.undesired_penetration_m, depth(undesired))
    self.peak_hand_force_n = max(self.peak_hand_force_n, hand_force)
    if approach:
      self.approach_hand_object_penetration_m = max(
        self.approach_hand_object_penetration_m, hand_depth
      )
      self.approach_object_displacement_m = max(
        self.approach_object_displacement_m,
        float(np.linalg.norm(object_pos[:2] - OBJECT_POS[:2])),
      )


APPROACH_END, CLOSE_END, LIFT_START, LIFT_END = 3.0, 5.0, 6.0, 8.0
LIFT_CLEARANCE = 0.06


@dataclass
class ScriptedGrasp:
  """Straight-line approach, close and vertical lift for one placement."""

  approach: np.ndarray  # [K, 6] arm poses every centimetre along the approach.
  lift: np.ndarray  # [6]

  @classmethod
  def solve(
    cls,
    model: mujoco.MjModel,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
  ) -> ScriptedGrasp:
    """Defaults to the nominal placement at OBJECT_POS with zero yaw."""
    pos = np.array(OBJECT_POS) if pos is None else pos
    quat = yaw_quat(0.0) if quat is None else quat
    solver = PregraspSolver(model)
    surface = box_surface_points(np.random.default_rng(0))
    start = solver.solve(pos, quat, surface)
    if start is None:
      raise RuntimeError("No feasible pre-grasp for this placement.")
    rotation = start.rotation
    begin = solver.wrist_pose(start.center, rotation, STANDOFF)
    end = pos - rotation @ np.array(HAND_CENTER)
    count = int(np.ceil(np.linalg.norm(end - begin) / 0.01)) + 1
    arms = [start.arm]
    for point in np.linspace(begin, end, count)[1:]:
      arm = solver.ik(point, rotation, arms[-1])
      if arm is None:
        raise RuntimeError("Approach line leaves the reachable workspace.")
      arms.append(arm)
    top = end + np.array([0.0, 0.0, LIFT_HEIGHT + LIFT_CLEARANCE])
    lift = solver.ik(top, rotation, arms[-1])
    if lift is None:
      raise RuntimeError("Lift pose is unreachable.")
    return cls(np.array(arms), lift)

  def target(self, time: float) -> np.ndarray:
    """Approach, close, pause, lift, then hold."""
    s = np.clip(time / APPROACH_END, 0.0, 1.0) * (len(self.approach) - 1)
    i = min(int(s), len(self.approach) - 2)
    arm = self.approach[i] + (self.approach[i + 1] - self.approach[i]) * (s - i)
    lift = np.clip((time - LIFT_START) / (LIFT_END - LIFT_START), 0.0, 1.0)
    arm = arm + (self.lift - self.approach[-1]) * lift
    close = np.clip((time - APPROACH_END) / (CLOSE_END - APPROACH_END), 0.0, 1.0)
    hand = np.array(OPEN_HAND) + (np.array(CLOSED_HAND) - OPEN_HAND) * close
    return np.concatenate([arm, hand])


def penetration(distances: np.ndarray) -> float:
  return float(max(0.0, -distances.min(initial=0.0)))


def run_native_probe() -> ProbeResult:
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, device="cpu").compile()
  cfg.sim.mujoco.apply(model)
  script = ScriptedGrasp.solve(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  initial_joints = cfg.scene.entities["robot"].init_state.joint_pos
  assert initial_joints is not None
  joint_names = tuple(initial_joints)
  joint_ids = np.array([model.joint(f"robot/{name}").id for name in joint_names])
  qpos_ids = model.jnt_qposadr[joint_ids]
  data.qpos[qpos_ids] = script.target(0.0)
  mujoco.mj_forward(model, data)
  ctrl_ids = np.array(
    [int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in joint_ids]
  )
  limits = model.jnt_range[joint_ids]
  initial_q = data.qpos[qpos_ids]
  if np.any(initial_q < limits[:, 0]) or np.any(initial_q > limits[:, 1]):
    raise ValueError("Pre-grasp joint pose exceeds hard joint limits.")
  initial = penetration(data.contact.dist)
  obj = model.joint("object/free_joint")
  qadr, vadr = int(obj.qposadr[0]), int(obj.dofadr[0])
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  required = math.ceil(HOLD_TIME / step_dt)
  count = 0
  force = np.zeros(6)
  rise = speed = peak_force = 0.0
  audit = RolloutAudit()
  for step in range(round(12.0 / step_dt)):
    goal = script.target(step * step_dt)
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
    audit.update(
      model,
      data.contact.geom,
      data.contact.dist,
      data.qpos[qadr : qadr + 3],
      approach=step * step_dt < APPROACH_END,
      hand_force=float(peak_force),
    )
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
    count >= required and initial < 1e-5 and audit.passed,
    initial,
    rise,
    count * step_dt,
    speed,
    float(peak_force),
    audit,
  )


def run_warp_probe(device: str = "cpu") -> ProbeResult:
  cfg = teacher_env_cfg()
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  cfg.episode_length_s = 12.0
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    script = ScriptedGrasp.solve(env.sim.mj_model)
    reset = env.event_manager.get_term_cfg("reset_pregrasp").func
    assert isinstance(reset, PregraspReset)

    def as_row(x) -> torch.Tensor:
      return torch.tensor(x, device=device, dtype=torch.float32)[None]

    reset.write(
      env,
      torch.arange(1, device=device),
      as_row(OBJECT_POS),
      as_row(yaw_quat(0.0)),
      as_row(script.approach[0]),
      as_row(script.approach[0]),
    )
    env.sim.forward()
    ncon = int(wp.to_torch(env.sim.wp_data.nacon)[0].item())
    initial = penetration(
      wp.to_torch(env.sim.wp_data.contact.dist)[:ncon].cpu().numpy()
    )
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, GraspAction)
    audit = RolloutAudit()
    required = math.ceil(HOLD_TIME / env.step_dt)
    count = 0
    for step in range(env.max_episode_length):
      goal = torch.tensor(
        script.target(step * env.step_dt), device=device, dtype=torch.float32
      )[None, :]
      q = env.scene["robot"].data.joint_pos[:, action.target_ids]
      _, _, terminated, truncated, _ = env.step((goal - q) / action.scale)
      ncon = int(wp.to_torch(env.sim.wp_data.nacon)[0].item())
      audit.update(
        env.sim.mj_model,
        wp.to_torch(env.sim.wp_data.contact.geom)[:ncon].cpu().numpy(),
        wp.to_torch(env.sim.wp_data.contact.dist)[:ncon].cpu().numpy(),
        env.scene["object"].data.root_link_pos_w[0].cpu().numpy(),
        approach=step * env.step_dt < APPROACH_END,
        hand_force=normal_force(env, "hand_object").max().item(),
      )
      count = count + 1 if stable_hold(env, LIFT_HEIGHT).item() else 0
      if count >= required or (terminated | truncated).any():
        break
    return ProbeResult(
      "warp",
      device,
      count >= required and initial < 1e-5 and audit.passed,
      initial,
      lift_height(env).item(),
      count * env.step_dt,
      object_speed(env).item(),
      normal_force(env, "hand_object").max().item(),
      audit,
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
