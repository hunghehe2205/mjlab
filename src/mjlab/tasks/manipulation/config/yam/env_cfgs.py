import colorsys
from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import (
  MultiCubeLiftingCommandCfg,
  PickPlaceCommandCfg,
)
from mjlab.tasks.manipulation.pick_place_env_cfg import (
  COMMAND_NAME as PICK_PLACE_COMMAND,
)
from mjlab.tasks.manipulation.pick_place_env_cfg import (
  GRASP_SENSOR,
  make_pick_place_env_cfg,
)


def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def yam_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Pick a cube off the floor and place it at a floor goal, YAM arm."""
  cfg = make_pick_place_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # The gripper stays inside the single 7-dim joint-position action rather than
  # becoming a separate binary term. At the configured init_std of 1.0 the
  # left_finger joint range spans only 0.457 sigma of the action, so 81.9% of
  # samples already saturate fully open or fully closed -- the discretization a
  # binary term would add is one the physics already performs. rsl_rl also ships
  # no discrete distribution head, so a thresholded term would be a
  # non-differentiable switch inside a policy gradient whose log-prob is still
  # computed on the continuous pre-threshold variable.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  ee_site = ("grasp_site",)
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = ee_site
  cfg.rewards["reach"].params["asset_cfg"].site_names = ee_site
  cfg.rewards["release"].params["asset_cfg"].site_names = ee_site

  command = cfg.commands[PICK_PLACE_COMMAND]
  assert isinstance(command, PickPlaceCommandCfg)
  command.robot_cfg.site_names = ee_site
  # `left_finger` is the only actuated gripper joint; `right_finger` mirrors it
  # through an MJCF equality. Its position grows as the gripper opens, which is
  # the direction `grasp_aperture_min` assumes.
  command.gripper_cfg.joint_names = ("left_finger",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if not isinstance(sensor, ContactSensorCfg):
      continue
    if sensor.name == "ee_ground_collision":
      sensor.primary.pattern = "link_6"
    elif sensor.name == GRASP_SENSOR:
      # The two fingertip bodies. Each carries five pad plates plus six 0.6 mm
      # tip spheres; matching the bodies rather than the spheres is what keeps
      # the signal alive for off-centre grasps carried by the upper pads.
      sensor.primary.pattern = r"[lr]f_down"

  cfg.viewer.body_name = "arm"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    # Play resamples often so a watcher sees repeated attempts; training does
    # not, because resampling teleports the cube.
    cfg.commands[PICK_PLACE_COMMAND].resampling_time_range = (12.0, 12.0)

  return cfg


def yam_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth", "rgbd"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play)

  # Delay the joint-velocity-hinge curriculum for the vision task.
  #
  # The base config ramps this penalty at iterations 500/1000 (12000/24000
  # steps). That schedule is tuned for the state-based policy, which is fed the
  # cube pose directly and breaks out of the reach-only plateau around iteration
  # 250 -- well before the clamp. The vision policy must first learn to localize
  # the cube from a 32x32 image, so at iteration 500 it is still exploring;
  # clamping joint motion there traps it in the reach-only optimum and it never
  # discovers the lift. Push the stages to 2000/4000 to keep the exploration
  # window open until the lift is found. Guarded because play zeroes curriculum.
  #
  # Softened final stage: the original -1.0 clamp (firing at iteration 4000)
  # over-penalized the vigorous joint motion a lift inherently needs, so the
  # policy went timid and task metrics regressed after 4000 (episode_success
  # ~0.80 -> ~0.65, position_error ~0.08 -> ~0.13) while only marginally
  # improving action_rate. Smoothness was NOT free to add on top of a solved
  # policy. Drop the final weight to -0.3 and raise max_vel to 0.8 below so the
  # penalty targets only genuinely excessive velocities and no longer erodes the
  # lift.
  if "joint_vel_hinge_weight" in cfg.curriculum:
    cfg.curriculum["joint_vel_hinge_weight"].params["stages"] = [
      {"step": 0, "weight": -0.01},
      {"step": 2000 * 24, "weight": -0.1},
      {"step": 4000 * 24, "weight": -0.3},
    ]

  # Only penalize joint velocities above 0.8 (base task uses 0.5). Lifting needs
  # brisk motion; a 0.5 threshold flags normal grasp/lift moves as "too fast".
  cfg.rewards["joint_vel_hinge"].params["max_vel"] = 0.8

  # RGB+D stacks rgb (3ch) and depth (1ch) into a single 4-channel image (early
  # fusion); rgb and depth stay single-channel. All modalities render from one
  # camera sensor.
  data_types = ("rgb", "depth") if cam_type == "rgbd" else (cam_type,)

  cam_name = "robot/camera_d405"
  sensor_name = cam_name.split("/")[-1]
  cam_cfg = CameraSensorCfg(
    name=sensor_name,
    camera_name=cam_name,
    height=32,
    width=32,
    data_types=data_types,
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  cam_terms = {}
  for data_type in data_types:
    params: dict[str, Any] = {"sensor_name": sensor_name}
    if data_type == "depth":
      params["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{sensor_name}_{data_type}"] = ObservationTermCfg(
      func=func, params=params
    )

  # concatenate_dim=0 stacks terms on the channel axis -> (B, C, H, W). Only rgbd
  # has multiple terms; rgb/depth keep the single-term default (a no-op).
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0 if len(data_types) > 1 else -1,
  )

  if "rgb" in data_types:
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg


def _cube_color(i: int, n: int) -> tuple[float, float, float, float]:
  """Generate a distinct color for cube i of n using HSV hue rotation."""
  h = i / max(n, 1)
  r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
  return (r, g, b, 1.0)


def yam_multi_cube_seg_env_cfg(
  num_cubes: int = 3,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Multi-cube task: depth + segmentation mask for goal conditioning."""
  cfg = make_lift_cube_env_cfg()

  cube_names = [f"cube_{i}" for i in range(num_cubes)]
  entities: dict[str, EntityCfg] = {"robot": get_yam_robot_cfg()}
  for i, name in enumerate(cube_names):
    color = _cube_color(i, num_cubes)
    entities[name] = EntityCfg(
      spec_fn=lambda c=color: get_cube_spec(rgba=c),
    )
  cfg.scene.entities = entities

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.commands = {
    "lift_height": MultiCubeLiftingCommandCfg(
      entity_names=tuple(cube_names),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
    ),
  }

  cfg.rewards["lift"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "reaching_std": 0.2,
      "bringing_std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["lift_precise"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_bring_object_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "std": 0.05,
    },
  )

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 55 + num_cubes * 120)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=("depth", "segmentation"),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  cam_terms = {
    "depth": ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={
        "sensor_name": "camera_d405",
        "cutoff_distance": 0.5,
      },
    ),
    "target_mask": ObservationTermCfg(
      func=manipulation_mdp.camera_target_cube_mask,
      params={
        "sensor_name": "camera_d405",
        "command_name": "lift_height",
      },
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  for group_name in ("actor", "critic"):
    obs = cfg.observations[group_name]
    obs.terms.pop("ee_to_cube", None)
    obs.terms.pop("cube_to_goal", None)
    obs.terms["goal_position"] = ObservationTermCfg(
      func=manipulation_mdp.target_position,
      params={
        "command_name": "lift_height",
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (
      4.0,
      4.0,
    )

  return cfg
