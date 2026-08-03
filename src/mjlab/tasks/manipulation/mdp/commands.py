from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class LiftingCommand(CommandTerm):
  cfg: LiftingCommandCfg

  def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w
    object_height = object_pos_w[:, 2]
    position_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    at_goal = (position_error < self.cfg.success_threshold).float()

    # Latch episode_success to 1 once goal is reached.
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    self.metrics["object_height"] = object_height
    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    position_error = self.metrics["position_error"]
    return position_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset episode success for resampled envs.
    self.episode_success[env_ids] = 0.0

    # Set target position based on difficulty mode.
    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

    # Reset object to new position.
    if self.cfg.object_pose_range is not None:
      r = self.cfg.object_pose_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]

      # Sample orientation (yaw only, keep upright).
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)

      velocity = torch.zeros(n, 6, device=self.device)

      self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{batch}",
      )


class MultiCubeLiftingCommand(CommandTerm):
  """Selects one of N cubes as the target at each reset."""

  cfg: MultiCubeLiftingCommandCfg

  def __init__(
    self,
    cfg: MultiCubeLiftingCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    self.cubes = [env.scene[name] for name in cfg.entity_names]
    self._num_cubes = len(self.cubes)

    geom_ids = [c.indexing.geom_ids for c in self.cubes]
    max_geoms = max(g.shape[0] for g in geom_ids)
    self._padded_geom_ids = torch.full(
      (self._num_cubes, max_geoms),
      -999,
      device=self.device,
      dtype=geom_ids[0].dtype,
    )
    for i, g in enumerate(geom_ids):
      self._padded_geom_ids[i, : g.shape[0]] = g

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.target_selection = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self._env_arange = torch.arange(self.num_envs, device=self.device)
    self._cached_target_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  @property
  def target_geom_ids(self) -> torch.Tensor:
    """Geom IDs of the target cube per env. Shape: (B, K)."""
    return self._padded_geom_ids[self.target_selection]

  def target_object_pos(self) -> torch.Tensor:
    """Position of the target cube per env.

    Cached per step — updated in _update_metrics which runs before rewards.
    """
    return self._cached_target_obj_pos

  def _update_metrics(self) -> None:
    all_pos = torch.stack([c.data.root_link_pos_w for c in self.cubes])
    self._cached_target_obj_pos = all_pos[self.target_selection, self._env_arange]
    obj_pos = self._cached_target_obj_pos
    error = torch.norm(self.target_pos - obj_pos, dim=-1)
    at_goal = (error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["position_error"] = error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["position_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0

    self.target_selection[env_ids] = torch.randint(
      0, self._num_cubes, (n,), device=self.device
    )

    if self.cfg.difficulty == "fixed":
      target = torch.tensor([0.4, 0.0, 0.3], device=self.device).expand(n, 3)
      self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]
    else:
      r = self.cfg.target_position_range
      lo = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      hi = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target = sample_uniform(lo, hi, (n, 3), device=self.device)
      self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]

    r = self.cfg.object_pose_range
    lo = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    hi = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    for cube in self.cubes:
      pos = sample_uniform(lo, hi, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),
        torch.zeros(n, device=self.device),
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)
      velocity = torch.zeros(n, 6, device=self.device)
      cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      cube.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=(1.0, 0.5, 0.0, 0.3),
        label=f"target_position_{batch}",
      )
      cube_pos = self.target_object_pos()[batch].cpu().numpy()
      marker = cube_pos.copy()
      marker[2] += 0.04
      visualizer.add_sphere(
        center=marker,
        radius=0.01,
        color=(1.0, 0.0, 0.0, 1.0),
        label=f"target_cube_marker_{batch}",
      )


@dataclass(kw_only=True)
class MultiCubeLiftingCommandCfg(CommandTermCfg):
  entity_names: tuple[str, ...] = ()
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.25, 0.40)
    y: tuple[float, float] = (-0.15, 0.15)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiCubeLiftingCommand:
    return MultiCubeLiftingCommand(self, env)


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  entity_name: str
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode."""

    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    """Configuration for object pose sampling when resampling commands."""

    x: tuple[float, float] = (0.3, 0.35)
    y: tuple[float, float] = (-0.1, 0.1)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LiftingCommand:
    return LiftingCommand(self, env)


class PickPlaceCommand(CommandTerm):
  """Floor goal plus the latched stage machine for pick-and-place.

  Two things separate this from :class:`LiftingCommand`. The goal lies on the
  floor rather than in the air, so "bring the cube to the goal" no longer
  implies lifting it -- the same reward is maximized by sliding -- and the lift
  requirement has to be enforced by gating the rewards on a grasp. And the task
  ends with a release, so it needs flags that say when letting go is the right
  move rather than a failure.

  The goal is sampled in an annulus around the object. That makes the minimum
  pick-to-place separation an invariant of the sampler instead of a filter
  bolted on afterwards, and leaves the curriculum only the outer radius to
  widen. Sampling the two independently would put 4.1% of episodes within a
  0.05 m goal radius at spawn -- roughly 84 of 2048 environments handed a free
  success on every reset.

  Ordering note. The environment calls ``reward_manager.compute``
  (``manager_based_rl_env.py:440``) and ``metrics_manager.compute`` (``:441``)
  *before* ``command_manager.compute`` (``:456``), so rewards and metrics read
  the flags of the previous control step while observations (``:464``) read
  fresh ones. The lag is a constant 20 ms and is identical for rewards and
  metrics, which is what lets a metric explain the reward beside it.
  """

  cfg: PickPlaceCommandCfg

  def __init__(self, cfg: PickPlaceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    # Command terms bypass ManagerBase._resolve_common_term_cfg, which only
    # walks `params`; a SceneEntityCfg held as a field must resolve itself.
    cfg.robot_cfg.resolve(env.scene)
    cfg.gripper_cfg.resolve(env.scene)
    self.robot: Entity = env.scene[cfg.robot_cfg.name]
    self._grasp_sensor: ContactSensor = env.scene[cfg.grasp_sensor_name]
    self._floor_sensor: ContactSensor = env.scene[cfg.floor_sensor_name]

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

    # Latched: true until the command is resampled.
    self.ever_grasped = torch.zeros(self.num_envs, device=self.device)
    self.lifted = torch.zeros(self.num_envs, device=self.device)
    self.placed_done = torch.zeros(self.num_envs, device=self.device)
    self.success_fired = torch.zeros(self.num_envs, device=self.device)

    # Instantaneous: recomputed every control step.
    self.grasping = torch.zeros(self.num_envs, device=self.device)
    self.placed_ok = torch.zeros(self.num_envs, device=self.device)
    self.success_now = torch.zeros(self.num_envs, device=self.device)
    self.fired_now = torch.zeros(self.num_envs, device=self.device)

    # Debounce state.
    self.hold_counter = torch.zeros(self.num_envs, device=self.device)
    self._at_rest = torch.zeros(self.num_envs, device=self.device)
    self._clear_run = torch.zeros(self.num_envs, device=self.device)

    for name in (
      "position_error",
      "ever_grasped",
      "lifted",
      "in_radius",
      "on_floor",
      "at_rest",
      "no_contact",
      "placed_ok",
      "episode_success",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    obj_pos = self.object.data.root_link_pos_w
    ee_pos = self.robot.data.site_pos_w[:, self.cfg.robot_cfg.site_ids].squeeze(1)

    # Grasp: both fingers loaded AND the gripper actually spanning something.
    # `found` is a match *count*, not a boolean, and a one-sided touch is a
    # nudge rather than a grasp, hence the `all`.
    #
    # The aperture test is not belt-and-braces, it closes a measured leak. Force
    # on both fingertip bodies does NOT imply the cube is pinched between them:
    # a fully shut gripper used as a blunt blade to shove the cube loads both
    # pads at 6.4 N each -- 20x this threshold -- and satisfied the force-only
    # predicate on 42% of steps, leaking roughly 20 reward per episode to
    # exactly the pushing behaviour the gate exists to withhold. A real pinch
    # holds the finger joint at 0.019-0.026; a shove holds it at ~-0.0001.
    grasp = self._grasp_sensor.data
    assert grasp.found is not None and grasp.force is not None
    touching = grasp.found > 0
    force = torch.norm(grasp.force, dim=-1)
    loaded = (touching & (force > self.cfg.grasp_force_min)).all(dim=-1)
    aperture = self.robot.data.joint_pos[:, self.cfg.gripper_cfg.joint_ids].squeeze(1)
    grasping = loaded & (aperture > self.cfg.grasp_aperture_min)

    # Resting on the floor is asked directly rather than inferred from height: a
    # 40 mm cube sits at z = 0.0200 on a face but 0.0283 on an edge and 0.0346
    # on a corner, so every height threshold either rejects tipped-but-settled
    # cubes or accepts cubes still pinched just above the ground.
    floor = self._floor_sensor.data
    assert floor.found is not None
    on_floor = (floor.found > 0).any(dim=-1)

    position_error = torch.norm(self.target_pos - obj_pos, dim=-1)
    in_radius = position_error < self.cfg.place_tol

    # At rest, with hysteresis: enter below the threshold, leave only at twice
    # it. The hold window is 25 steps; five raw predicates each flickering
    # independently at 1%/step would already drop the chance of completing it
    # to ~0.29, so debouncing is not optional.
    speed = torch.norm(self.object.data.root_link_lin_vel_w, dim=-1)
    spin = torch.norm(self.object.data.root_link_ang_vel_w, dim=-1)
    settled = (speed < self.cfg.settle_lin_vel) & (spin < self.cfg.settle_ang_vel)
    disturbed = (speed > 2.0 * self.cfg.settle_lin_vel) | (
      spin > 2.0 * self.cfg.settle_ang_vel
    )
    zeros = torch.zeros_like(self._at_rest)
    at_rest = torch.where(
      disturbed, zeros, torch.where(settled, torch.ones_like(zeros), self._at_rest)
    )
    self._at_rest = at_rest

    # Released, debounced over N consecutive contact-free steps so a sub-step
    # re-touch cannot be smuggled through.
    no_touch = (grasp.found == 0).all(dim=-1)
    self._clear_run = torch.where(no_touch, self._clear_run + 1.0, zeros)
    no_contact = self._clear_run >= self.cfg.release_clear_steps
    ee_retreated = torch.norm(ee_pos - obj_pos, dim=-1) > self.cfg.release_dist

    # `lifted` demands a grasp as well as height, so flicking the cube into the
    # air cannot unlock the downstream rewards.
    self.ever_grasped = torch.maximum(self.ever_grasped, grasping.float())
    high = obj_pos[:, 2] > self.cfg.lift_height
    self.lifted = torch.maximum(self.lifted, (grasping & high).float())

    placed_ok = (self.lifted > 0) & in_radius & on_floor & (at_rest > 0)
    success_now = placed_ok & no_contact & ee_retreated

    self.hold_counter = torch.where(success_now, self.hold_counter + 1.0, zeros)
    hold_done = self.hold_counter >= self.cfg.hold_steps

    # The one-shot fires on the rising edge only, and `success_fired` latches,
    # so a place / pick-up / place-again loop cannot farm it.
    self.fired_now = (hold_done & (self.success_fired == 0)).float()
    self.success_fired = torch.maximum(self.success_fired, hold_done.float())
    self.placed_done = torch.maximum(self.placed_done, hold_done.float())

    self.grasping = grasping.float()
    self.placed_ok = placed_ok.float()
    self.success_now = success_now.float()

    self.metrics["position_error"] = position_error
    self.metrics["ever_grasped"] = self.ever_grasped
    self.metrics["lifted"] = self.lifted
    self.metrics["in_radius"] = in_radius.float()
    self.metrics["on_floor"] = on_floor.float()
    self.metrics["at_rest"] = at_rest
    self.metrics["no_contact"] = no_contact.float()
    self.metrics["placed_ok"] = self.placed_ok
    self.metrics["episode_success"] = self.success_fired

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    for buf in (
      self.ever_grasped,
      self.lifted,
      self.placed_done,
      self.success_fired,
      self.grasping,
      self.placed_ok,
      self.success_now,
      self.fired_now,
      self.hold_counter,
      self._at_rest,
      self._clear_run,
    ):
      buf[env_ids] = 0.0

    r = self.cfg.object_pose_range
    obj_xy = sample_uniform(
      torch.tensor([r.x[0], r.y[0]], device=self.device),
      torch.tensor([r.x[1], r.y[1]], device=self.device),
      (n, 2),
      device=self.device,
    )
    goal_xy = self._sample_goal_xy(obj_xy)

    origins = self._env.scene.env_origins[env_ids]
    floor = torch.full((n, 1), self.cfg.floor_z, device=self.device)
    self.target_pos[env_ids] = torch.cat([goal_xy, floor], dim=-1) + origins

    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    quat = quat_from_euler_xyz(
      torch.zeros(n, device=self.device),
      torch.zeros(n, device=self.device),
      yaw,
    )
    obj_pos = torch.cat([obj_xy, floor], dim=-1) + origins
    self.object.write_root_link_pose_to_sim(
      torch.cat([obj_pos, quat], dim=-1), env_ids=env_ids
    )
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )

  def _sample_goal_xy(self, obj_xy: torch.Tensor) -> torch.Tensor:
    """Sample a floor goal in an annulus around the object, inside the workspace.

    Rejection over a few attempts gives a well-spread distribution. The fallback
    is the workspace corner farthest from the object, which for the configured
    boxes always clears the inner radius, so the minimum separation holds
    without an unbounded loop.
    """
    n = obj_xy.shape[0]
    box = self.cfg.goal_workspace
    lo = torch.tensor([box.x[0], box.y[0]], device=self.device)
    hi = torch.tensor([box.x[1], box.y[1]], device=self.device)
    r_lo, r_hi = self.cfg.goal_radius_range

    goal = torch.zeros_like(obj_xy)
    valid = torch.zeros(n, dtype=torch.bool, device=self.device)
    for _ in range(self.cfg.goal_sample_attempts):
      theta = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
      radius = sample_uniform(r_lo, r_hi, (n,), device=self.device)
      offset = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
      cand = obj_xy + radius.unsqueeze(-1) * offset
      inside = ((cand >= lo) & (cand <= hi)).all(dim=-1)
      goal = torch.where((inside & ~valid).unsqueeze(-1), cand, goal)
      valid = valid | inside

    corners = torch.tensor(
      [
        [box.x[0], box.y[0]],
        [box.x[0], box.y[1]],
        [box.x[1], box.y[0]],
        [box.x[1], box.y[1]],
      ],
      device=self.device,
    )
    dists = torch.norm(obj_xy.unsqueeze(1) - corners.unsqueeze(0), dim=-1)
    farthest = corners[torch.argmax(dists, dim=-1)]
    return torch.where(valid.unsqueeze(-1), goal, farthest)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      # Drawn at the live success radius, so the curriculum shrinking the
      # tolerance is visible rather than something you have to read off a log.
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=self.cfg.place_tol,
        color=self.cfg.viz.goal_color,
        label=f"pick_place_goal_{batch}",
      )
      marker = self.object.data.root_link_pos_w[batch].cpu().numpy().copy()
      marker[2] += 0.06
      grasped = bool(self.grasping[batch] > 0)
      visualizer.add_sphere(
        center=marker,
        radius=0.012,
        color=(0.1, 1.0, 0.1, 1.0) if grasped else (1.0, 0.2, 0.2, 1.0),
        label=f"pick_place_grasp_{batch}",
      )


@dataclass(kw_only=True)
class PickPlaceCommandCfg(CommandTermCfg):
  """Configuration for :class:`PickPlaceCommand`.

  ``place_tol`` and ``goal_radius_range`` are the two curriculum axes. Both are
  read live every step / every resample, and ``CommandManager`` does not
  deepcopy its config (``command_manager.py:262-273``), so a curriculum term can
  mutate this object in place and the change takes effect immediately.
  """

  entity_name: str
  robot_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  gripper_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", joint_names=())
  )
  """Must select exactly one joint, whose position grows as the gripper opens."""
  grasp_sensor_name: str = "fingertips_cube"
  floor_sensor_name: str = "cube_floor"

  floor_z: float = 0.02
  """Cube centre height at rest on a face; equals the cube half extent."""
  lift_height: float = 0.10
  place_tol: float = 0.10
  settle_lin_vel: float = 0.02
  settle_ang_vel: float = 0.2
  release_dist: float = 0.08
  release_clear_steps: int = 3
  hold_steps: int = 25
  """25 control steps at 50 Hz = 0.5 s."""
  grasp_force_min: float = 0.3
  """Per-finger net force. A firm clamp measures 12-14 N per finger, so this is
  a floor against grazes rather than a tuned value; lowering it does not buy
  robustness, because dropouts show up as ``found`` going to zero on one finger
  rather than as a force dipping under the threshold."""
  grasp_aperture_min: float = 0.012
  """Gripper joint position below which contact is a shove, not a pinch.

  Object-size dependent by construction: it sits roughly halfway between a shut
  gripper (~0.000) and a gripper spanning the 40 mm cube (0.019-0.026). Re-derive
  if the object size changes."""

  goal_radius_range: tuple[float, float] = (0.15, 0.20)
  goal_sample_attempts: int = 8

  @dataclass
  class WorkspaceCfg:
    x: tuple[float, float] = (0.20, 0.50)
    y: tuple[float, float] = (-0.25, 0.25)

  goal_workspace: WorkspaceCfg = field(default_factory=WorkspaceCfg)

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.20, 0.40)
    y: tuple[float, float] = (-0.20, 0.20)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  @dataclass
  class VizCfg:
    goal_color: tuple[float, float, float, float] = (0.2, 0.9, 0.3, 0.35)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PickPlaceCommand:
    return PickPlaceCommand(self, env)


def get_pick_place_command(
  env: ManagerBasedRlEnv, command_name: str
) -> PickPlaceCommand:
  """Fetch a command term, asserting it is a :class:`PickPlaceCommand`."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, PickPlaceCommand):
    raise TypeError(
      f"Command '{command_name}' must be a PickPlaceCommand, got {type(command)}"
    )
  return command
