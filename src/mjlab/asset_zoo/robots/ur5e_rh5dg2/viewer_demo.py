"""Interactive MuJoCo viewer for the UR5e + rh5dg2 hand on the workstation.

Run: ``uv run python -m mjlab.asset_zoo.robots.ur5e_rh5dg2.viewer_demo``

Opens the native MuJoCo viewer whose Control panel exposes one slider per
joint (position servos), starting from the home keyframe.
"""

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.rh5dg2_hand import rh5dg2_hand_constants as hand
from mjlab.asset_zoo.robots.universal_robots_ur5e import ur5e_constants as ur5e
from mjlab.asset_zoo.robots.ur5e_rh5dg2 import ur5e_rh5dg2_constants as combo
from mjlab.asset_zoo.scenes.workstation import ARM_MOUNT_Z, get_workstation_spec


def _add_position_actuator(
  spec: mujoco.MjSpec, joint: str, kp: float, kd: float, effort: float
) -> None:
  a = spec.add_actuator()
  a.name = joint
  a.target = joint
  a.trntype = mujoco.mjtTrn.mjTRN_JOINT
  a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  gainprm = np.zeros(10)
  gainprm[0] = kp
  a.gainprm = gainprm
  biasprm = np.zeros(10)
  biasprm[1] = -kp
  biasprm[2] = -kd
  a.biasprm = biasprm
  a.forcelimited = True
  a.forcerange = np.array([-effort, effort])
  # Slider range = joint range, so the control panel spans the full motion.
  a.inheritrange = 1.0


def _add_scene_visuals(spec: mujoco.MjSpec) -> None:
  """Brighten the standalone scene: headlight, lights, skybox and a floor."""
  spec.visual.headlight.ambient = np.array([0.5, 0.5, 0.5])
  spec.visual.headlight.diffuse = np.array([0.7, 0.7, 0.7])
  spec.visual.headlight.specular = np.array([0.1, 0.1, 0.1])

  sky = spec.add_texture()
  sky.name = "skybox"
  sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
  sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  sky.rgb1 = np.array([0.3, 0.5, 0.7])
  sky.rgb2 = np.array([0.0, 0.0, 0.0])
  sky.width = 512
  sky.height = 3072

  for x, y, shadow in ((1.5, 1.5, True), (-1.5, -1.5, False)):
    light = spec.worldbody.add_light()
    light.pos = np.array([x, y, 3.0])
    light.dir = np.array([-x, -y, -3.0])
    light.diffuse = np.array([0.4, 0.4, 0.4])
    light.castshadow = shadow

  floor = spec.worldbody.add_geom()
  floor.name = "floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = np.array([0.0, 0.0, 0.05])
  floor.rgba = np.array([0.3, 0.3, 0.35, 1.0])


def _arm_gains(joint: str) -> tuple[float, float, float]:
  for cfg in ur5e.ARM_ACTUATORS:
    assert isinstance(cfg, BuiltinPositionActuatorCfg)
    if joint in cfg.target_names_expr:
      assert cfg.effort_limit is not None
      return cfg.stiffness, cfg.damping, cfg.effort_limit
  raise KeyError(joint)


def build_model() -> tuple[mujoco.MjModel, dict[str, float]]:
  """Combined robot + scene with position servos and a home keyframe."""
  robot = combo.get_spec()
  for joint in (*ur5e.SIZE3_JOINTS, *ur5e.SIZE1_JOINTS):
    _add_position_actuator(robot, joint, *_arm_gains(joint))
  compiled = robot.compile()
  hand_joints = [
    n
    for i in range(compiled.njnt)
    if (n := mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_JOINT, i))
    and n.startswith("R_")
  ]
  hand_cfg = hand.HAND_ACTUATOR
  assert isinstance(hand_cfg, BuiltinPositionActuatorCfg)
  assert hand_cfg.effort_limit is not None
  for joint in hand_joints:
    _add_position_actuator(
      robot, joint, hand_cfg.stiffness, hand_cfg.damping, hand_cfg.effort_limit
    )

  scene = get_workstation_spec()
  frame = scene.worldbody.add_frame(pos=(0.0, 0.0, ARM_MOUNT_Z), quat=combo.BASE_ROT)
  frame.attach_body(robot.body("base"), "robot_", "")
  _add_scene_visuals(scene)
  opt = scene.option
  opt.timestep = 0.005
  opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  opt.iterations = 10
  opt.ls_iterations = 20
  opt.impratio = 10.0
  opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

  model = scene.compile()
  home = {f"robot_{j}": v for j, v in combo.ARM_HOME_RAD.items()}
  qpos = np.zeros(model.nq)
  ctrl = np.zeros(model.nu)
  for joint, value in home.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qpos[model.jnt_qposadr[jid]] = value
    ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)] = value
  key = scene.add_key(name="home")
  key.qpos = qpos
  key.ctrl = ctrl
  return scene.compile(), home


def main() -> None:
  import mujoco.viewer as viewer

  model, _ = build_model()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  viewer.launch(model, data)


if __name__ == "__main__":
  main()
