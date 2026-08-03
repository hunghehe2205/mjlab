"""Pick-and-place: lift a cube off the floor and set it down at a floor goal.

Built on top of :func:`make_lift_cube_env_cfg`, which supplies the scaffolding
that does not change -- action term, fingertip friction randomization, the
end-effector ground sensor, solver settings and viewer -- and then replaces the
parts that do.

Why this is not lift with a lower target. The lift reward is
``exp(-d(cube, goal)^2 / s^2)`` with the goal floating between 0.2 and 0.4 m.
Nothing in it mentions height, but a goal in the air means "bring the cube to
the goal" *implies* picking it up. Put the goal on the floor and that
implication evaporates, so the lift requirement has to live in the reward
instead: every goal-seeking term here is gated on currently holding the cube.

The gate's justification is a trap, not a shortcut. Measured over 80 scripted
bulldoze episodes, shoving is a poor *route*: mean gap closure 62 mm of 173 mm,
0 of 80 episodes ended inside a 0.04 m tolerance, and the cube tipped past 45
degrees in 64% of them -- with a 40 mm cube and floor friction near 1.0,
tipping precedes sliding for any contact above 20 mm, and a closed gripper on a
40 mm face always contacts higher than that. What makes the ungated kernel
dangerous is that it pays anyway: 16.35 per episode for the cube merely sitting
where it spawned, rising to 60 at the goal, and removing the gate roughly
doubles a pusher's take (15.9 -> 35.3). A dense signal the policy cannot
convert into success is exactly the shape of a local optimum worth defending
against.

Second consequence of "place": the arm has to let go. The shipped
``staged_position_reward`` returns ``reaching * (1 + bringing)`` -- everything
multiplied by a kernel that decays as the end effector leaves the cube -- so
under it retreating is always punished and release can never be learned. Here
``reach`` switches off while the cube is held, and a dense ``hold`` term pays
only while the gripper is clear.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import PickPlaceCommandCfg

COMMAND_NAME = "pick_place"
OBJECT_NAME = "cube"
GRASP_SENSOR = "fingertips_cube"
FLOOR_SENSOR = "cube_floor"

# Index of the gripper dimension in the flat action vector. The single
# JointPositionAction sweeps in all 7 actuators, and the gripper is last.
GRIPPER_ACTION_INDEX = 6


def make_pick_place_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the robot-agnostic pick-and-place task configuration."""
  cfg = make_lift_cube_env_cfg()

  # --- sensors ---------------------------------------------------------------
  # Fingertip-to-cube contact. Primary is the two `[lr]f_down` *bodies*, not the
  # twelve 0.6 mm tip spheres: the spheres only register within about +/-2.5 cm
  # of an ideally centred grasp and can read zero for an entire episode while a
  # genuine grasp is being carried by the slanted upper pads. The bodies are a
  # strict superset of tips plus pads and still exclude `lf_rot*` and `link6_*`.
  # `reduce="netforce"` because `maxforce` keeps one contact out of six and
  # under-reports a real grip about fivefold, while `reduce="none"` is not
  # deterministic across runs.
  grasp_sensor_cfg = ContactSensorCfg(
    name=GRASP_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern="",  # Set per-robot.
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="cube_geom", entity=OBJECT_NAME),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=0,
  )

  # "Resting on the floor", asked directly. A 40 mm cube sits at z = 0.0200 on a
  # face but 0.0283 on an edge and 0.0346 on a corner, so any height threshold
  # either denies success to a tipped-but-motionless cube or accepts one still
  # pinched just above the ground.
  floor_sensor_cfg = ContactSensorCfg(
    name=FLOOR_SENSOR,
    primary=ContactMatch(mode="body", pattern="cube", entity=OBJECT_NAME),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
    history_length=0,
  )

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      # `reduce="none"` reports an arbitrary one of the four contacts the pad
      # plates generate and is not reproducible run to run -- the same descent
      # measured 16.96 N and 22.16 N on consecutive runs of one script. That
      # non-determinism sits directly on the termination decision variable.
      sensor.reduce = "maxforce"
  cfg.scene.sensors = cfg.scene.sensors + (grasp_sensor_cfg, floor_sensor_cfg)

  # Measured worst case for this task -- cube on the floor, pad plates also on
  # the floor, arm moving -- is 82 contacts per world (p99 69, mean 18.3), and
  # the compiled model's absolute ceiling is 85. The lift task's 55 would
  # overflow, and overflow is silent: past-buffer contacts are dropped and only
  # a sticky bit is set, which nothing in mjlab reads.
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 120)

  # --- commands --------------------------------------------------------------
  # Resampling is pushed beyond the episode so it never fires mid-episode.
  # `_resample_command` teleports the cube, which with the lift task's
  # (8, 12) s window would yank a placed -- or held -- cube away one to two
  # times per 20 s episode and destroy completed placements.
  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: PickPlaceCommandCfg(
      entity_name=OBJECT_NAME,
      resampling_time_range=(25.0, 30.0),
      debug_vis=True,
      robot_cfg=SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      grasp_sensor_name=GRASP_SENSOR,
      floor_sensor_name=FLOOR_SENSOR,
    )
  }
  cfg.commands = commands

  # --- observations ----------------------------------------------------------
  # NOTE: `critic_terms = {**actor_terms}` in the lift config is a shallow copy,
  # so the two groups share ObservationTermCfg *objects*. Retargeting params
  # through either group updates both; adding or popping keys does not.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["cube_to_goal"].params["command_name"] = COMMAND_NAME
    # Contact belongs in the actor group, not just the critic: the real YAM
    # senses the same thing through gripper current, so it opens no sim-to-real
    # gap the way a privileged pose would.
    terms["grasp_contact"] = ObservationTermCfg(
      func=manipulation_mdp.fingertip_contact,
      params={"sensor_name": GRASP_SENSOR},
    )
    # Both of these are Markov requirements rather than conveniences. Whether to
    # hold or release depends on latched history, and success requires the
    # predicates to hold *continuously*, so the policy has to see how much of
    # the hold window it has banked.
    terms["stage_flags"] = ObservationTermCfg(
      func=manipulation_mdp.pick_place_stage_flags,
      params={"command_name": COMMAND_NAME},
    )
    terms["hold_progress"] = ObservationTermCfg(
      func=manipulation_mdp.pick_place_hold_progress,
      params={"command_name": COMMAND_NAME},
    )

  critic_terms = cfg.observations["critic"].terms
  critic_terms["cube_velocity"] = ObservationTermCfg(
    func=manipulation_mdp.object_velocity,
    params={"object_name": OBJECT_NAME},
  )
  critic_terms["grasp_force"] = ObservationTermCfg(
    func=manipulation_mdp.contact_force_magnitude,
    params={"sensor_name": GRASP_SENSOR},
  )

  # --- rewards ---------------------------------------------------------------
  ee_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.
  release_ee_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

  for name in ("lift", "lift_precise"):
    cfg.rewards.pop(name)

  cfg.rewards["reach"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_reach,
    weight=1.0,
    params={
      "command_name": COMMAND_NAME,
      "object_name": OBJECT_NAME,
      "std": 0.2,
      "asset_cfg": ee_cfg,
    },
  )
  cfg.rewards["grasp"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_grasp,
    weight=1.0,
    params={"command_name": COMMAND_NAME},
  )
  cfg.rewards["lift"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_lift,
    weight=1.0,
    params={"command_name": COMMAND_NAME, "object_name": OBJECT_NAME},
  )
  cfg.rewards["transport"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_goal,
    weight=1.0,
    params={
      "command_name": COMMAND_NAME,
      "object_name": OBJECT_NAME,
      "std": 0.3,
    },
  )
  # `std` is kept equal to the command's `place_tol` by the curriculum.
  cfg.rewards["place"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_goal,
    weight=2.0,
    params={
      "command_name": COMMAND_NAME,
      "object_name": OBJECT_NAME,
      "std": 0.10,
    },
  )
  cfg.rewards["release"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_release,
    weight=2.0,
    params={
      "command_name": COMMAND_NAME,
      "object_name": OBJECT_NAME,
      "asset_cfg": release_ee_cfg,
    },
  )
  # The dense term is the actual success signal. A one-shot bonus is latched, so
  # a policy that finishes and one that finishes then re-grips to farm the dense
  # terms both collect it exactly once and it cancels out of the comparison at
  # any weight. `success` below is only a tiebreaker; at weight * dt with
  # dt = 0.02 it needs a weight near `fraction_active * T` to register at all.
  cfg.rewards["hold"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_hold,
    weight=15.0,
    params={"command_name": COMMAND_NAME},
  )
  cfg.rewards["success"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_success,
    weight=400.0,
    params={"command_name": COMMAND_NAME},
  )
  cfg.rewards["dropped"] = RewardTermCfg(
    func=manipulation_mdp.pick_place_dropped,
    weight=-1.0,
    params={
      "command_name": COMMAND_NAME,
      "object_name": OBJECT_NAME,
      "height": 0.06,
    },
  )
  # Chattering the gripper open/closed every step costs only 0.80 per episode at
  # the global -0.01, while a grasp reward at weight 1.0 held at 50% duty
  # collects 10.0. Break-even is around -0.25, and raising the global term 25x
  # would be a regression for the arm joints, which train fine at -0.01.
  cfg.rewards["gripper_action_rate"] = RewardTermCfg(
    func=manipulation_mdp.action_rate_l2_subset,
    weight=-0.25,
    params={"indices": (GRIPPER_ACTION_INDEX,)},
  )

  # --- terminations ----------------------------------------------------------
  # At the grasp heights a 40 mm floor cube actually requires, the lift task's
  # 10.0 N guard fires on 15-16 of 16 environments -- it is not a guard there,
  # it is a tax on the task. Benign peaks reach 27.96 N, so 30 N leaves only 7%
  # margin and a mild over-press measured 28.23 N without firing.
  #
  # 40 N is chosen for the asymmetry: this term is not a `time_out`, so PPO cuts
  # the value bootstrap and every spurious trip costs the entire remaining
  # return, whereas a missed mild knock costs only motion smoothness. It still
  # catches the damaging cases (42 / 55 / 69 / 267 N presses). Accepted
  # explicitly: slams in the 35-40 N band no longer terminate.
  #
  # Narrowing the primary instead would be equivalent to deleting the guard --
  # GRIPPER_ONLY_COLLISION already zeroes contype/conaffinity everywhere from
  # the base through link_5, so the link_6 subtree *is* the robot's collision set.
  cfg.terminations["ee_ground_collision"].params["force_threshold"] = 40.0

  # --- curriculum ------------------------------------------------------------
  # Pick-and-place has more stages than lift, so its exploration phase runs
  # longer; the stock clamp at iteration 1000 is what froze RGB lift in a
  # reach-only optimum at 0% success. Use the late, soft schedule.
  cfg.curriculum["joint_vel_hinge_weight"].params["stages"] = [
    {"step": 0, "weight": -0.01},
    {"step": 2000 * 24, "weight": -0.1},
    {"step": 4000 * 24, "weight": -0.3},
  ]
  cfg.rewards["joint_vel_hinge"].params["max_vel"] = 0.8

  cfg.curriculum["pick_place_difficulty"] = CurriculumTermCfg(
    func=manipulation_mdp.pick_place_curriculum,
    params={
      "command_name": COMMAND_NAME,
      "place_reward_name": "place",
      "stages": [
        {"step": 0, "radius": (0.15, 0.20), "tol": 0.10},
        {"step": 1500 * 24, "radius": (0.15, 0.32), "tol": 0.06},
        {"step": 3000 * 24, "radius": (0.15, 0.45), "tol": 0.04},
      ],
    },
  )

  # --- metrics ---------------------------------------------------------------
  # `hold_counter` is logged as both mean and max from the first run onward. It
  # is the only signal that separates the two ways this task fails: a max that
  # rarely clears ~5 means the stage flags are flickering and no amount of
  # training will help, whereas a max sitting at `hold_steps` with a low mean
  # just means success is still rare. `CommandTerm` metrics cannot give this --
  # `CommandTerm.reset` reports the value at reset time, which for a counter
  # that zeroes on any violation is usually 0.
  cfg.metrics = {
    "hold_counter_mean": MetricsTermCfg(
      func=manipulation_mdp.pick_place_hold_counter,
      params={"command_name": COMMAND_NAME},
      reduce="mean",
    ),
    "hold_counter_max": MetricsTermCfg(
      func=manipulation_mdp.pick_place_hold_counter,
      params={"command_name": COMMAND_NAME},
      reduce="max",
    ),
    "success": MetricsTermCfg(
      func=manipulation_mdp.pick_place_flag,
      params={"command_name": COMMAND_NAME, "flag": "success_fired"},
      reduce="last",
    ),
    "ever_grasped": MetricsTermCfg(
      func=manipulation_mdp.pick_place_flag,
      params={"command_name": COMMAND_NAME, "flag": "ever_grasped"},
      reduce="last",
    ),
    "lifted": MetricsTermCfg(
      func=manipulation_mdp.pick_place_flag,
      params={"command_name": COMMAND_NAME, "flag": "lifted"},
      reduce="last",
    ),
    "grasp_duty": MetricsTermCfg(
      func=manipulation_mdp.pick_place_flag,
      params={"command_name": COMMAND_NAME, "flag": "grasping"},
      reduce="mean",
    ),
    # Peak, not rate: a termination rate near zero cannot distinguish a guard
    # correctly parked above the benign range from one retuned into a new margin.
    "ee_ground_force_peak": MetricsTermCfg(
      func=manipulation_mdp.contact_force_peak,
      params={"sensor_name": "ee_ground_collision"},
      reduce="max",
    ),
    "grasp_force_peak": MetricsTermCfg(
      func=manipulation_mdp.contact_force_peak,
      params={"sensor_name": GRASP_SENSOR},
      reduce="max",
    ),
  }

  return cfg
