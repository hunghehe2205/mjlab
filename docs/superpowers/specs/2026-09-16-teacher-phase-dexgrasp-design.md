# Grasp-only teacher — UR5e + rh5dg2 / mjlab

Updated: 2026-09-24. Reframed as grasp-only after RobustDexGrasp; supersedes the
grasp-and-lift reward with a 3 s hold success.

## Scope and acceptance

Teacher controls all 24 arm/hand joints from a sampled pre-grasp and learns to
grasp one primitive, as the RobustDexGrasp teacher does: no lift or hold reward,
fixed-length grasp episodes. Grasp quality is measured by the paper's lift test:
after the grasp phase the arm is raised by script while the policy keeps the
fingers, and success is a rise above 0.10 m at the end. The object placement and
the matching pre-grasp are randomized per reset. Transport, release, student,
physics domain randomization and real deployment are outside this implementation.

The lift test runs in evaluation and play only; training never overrides policy
actions. A scripted close/lift probe remains the physics acceptance check.

## Scene and physics baseline

- Existing UR5e/rh5dg2 entity and workstation.
- Sampled pre-grasp (`grasp/pregrasp.py`), mirrored from the reference so the
  workspace lies in +Y of the arm base:
  - Object xy: polar angle 0.3π–0.7π, distance 0.45–0.75 m, |x| < 0.25 m.
    Training mixes uniform and Beta(0.5, 0.5) edge-biased draws 50/50; play is
    uniform. Yaw uniform in [-π, π]; the box rests on the table.
  - Visible points: box surface points whose faces point toward a virtual
    camera at (-0.035, 0.58, 1.531) m. Their centroid is the affordance center.
  - Top approach (palm down). The wrist frame `right_hand` has +X = palm normal
    toward the object and +Z = finger axis; 10 finger-axis rolls over the half
    plane facing away from the robot.
  - The grasp reference `HAND_CENTER` = (0.125, 0.005, 0.190) m in the wrist
    frame (box center at pre-close) is placed 0.06 m from the affordance center
    along the approach (the reference uses 0.25 m). The thumb tip then starts
    about 2 cm above the box top, so training focuses on grasping, not reaching.
  - Damped least-squares IK per roll, seeded from a collision-free palm-down
    branch and rejected outside the 0.9 soft limits. Score = 5 x grasp width +
    |wrist_2 - π/2| + 0.5(|wrist_2| - 3.2) among widths below 0.18 m.
  - Reject any robot-world contact within 5 mm and any robot self-penetration
    (the arm mount on the pedestal is excluded).
  - 1024 placements are solved once at startup; each reset draws one uniformly.
- Fixed 18-joint pre-shape: thumb yaw 0.8 rad (47° from the palm plane, 5.6 cm
  thumb-box gap at pre-close) with mcp/pip/dip 0.08/0.06/0.06 rad, index/middle
  yaw 0, other flexion 0.1 rad. All values lie inside the 0.9 soft limits. The
  probe closes the thumb back to yaw 1.2 rad.
- MVP object: box, half extents (0.03, 0.03, 0.06) m, mass 0.08 kg,
  friction 1.0, condim 4. The nominal placement (-0.10, 0.48,
  TABLE_TOP_Z + 0.06) with identity orientation is used by the probe.
- Keep robot actuator gains/armature from the entity configs. Build the probe
  from these configs too, avoiding the standalone viewer's missing hand armature.
- Teacher-local collision variant: replace DIP cylinder pads with inscribed
  capsules/spheres. Their poses, body inertias and visual meshes are preserved;
  the shared robot asset is untouched. Cylinder-box contacts in this Warp version
  support only one contact and the original pads produced backend-sensitive
  grasps. This is an explicit collision approximation, not real-hand calibration.
- implicitfast, elliptic cone, impratio 10; physics timestep 0.005 s.
- Policy interval 0.05 s (20 Hz), decimation 10. This deliberately differs from
  the paper's 5 Hz to allow finer approach/lift corrections. Grasp episode 4 s.
- All object motion must result from gravity, contact and robot actuation:
  no object weld, teleport, gravity cancellation or attachment during rollout.
- Probe must start without penetration and demonstrate >=0.10 m lift, >=3 s hold
  with hand contact and no table support. Record the backend and measured result.
- Probe acceptance also checks contact depths at 20 Hz: hand/object and
  object/table <=3 mm, other robot contacts <=1 mm. During the open-hand approach,
  hand/object depth <=0.01 mm and object horizontal displacement <=1 mm.
  These are numerical regression tolerances for the soft-contact model, not
  rigid-contact guarantees or maxima over all physics substeps. Collision
  ownership must use body IDs, including unnamed UR5 collision geoms.

## Actions and reset

Use one task-local action term with 24 normalized actions, clipped to [-1, 1].
At each policy step, compute once:

    target = clamp(previous_target + action * scale, q - max_offset, q + max_offset)
    target = clamp(target, hard_joint_limits)

Arm scale 0.01 rad, hand scale 0.03 rad per policy step (0.2 and 0.6 rad/s per
unit action). The reference moves 0.005/0.015 rad per 0.2 s step (0.025/0.075
rad/s) around the measured q; ours is faster so the 6 cm approach fits 4 s, and
keeps its 1:3 arm/hand ratio. The target accumulates, so a zero action holds the
pose instead of sagging under gravity, and the hand can build preload; max_offset
(arm 0.10 rad, hand 0.50 rad, the previous per-step scales) bounds the lead over q
and therefore the squeeze. Hold this target through all physics substeps, except
that in half of the steps (random per env) the first substep still uses the
previous target, as the reference's actuation delay. Reset raw action to zero and
target to the reset joint pose, including partial environment resets.

Lift test (`lift_test=True`, default in play): from GRASP_TIME on, the arm target
ramps linearly over 3 s from the arm pose at that instant by a per-placement joint
offset, and the policy's arm actions are ignored; hand targets still follow the
policy. The offset is the IK difference between the pre-grasp and the same wrist
pose raised 0.20 m, solved with the pool. This mirrors the reference's
`switch_root_guidance`, which interpolates the arm joints toward a raised pose over
80 steps while the policy controls the fingers.

Reset robot and props with env origins; the pre-grasp event writes the sampled
object pose and the arm/hand joints with zero velocities and stores each env's
object start position and lift offset.

## Privileged observations and frames

Runtime observations/rewards use Warp-backed EntityData tensors and mjlab
ContactSensor data on the simulation device. Never read live state from
sim.mj_data or loop over CPU mj_contactForce/mj_geomDistance in the training loop.

- d: vectors from fixed hand-link anchors to the oriented box surface, computed
  analytically in batched Torch, including anchors inside the box. Finger DIP
  anchors use force-sensor body origins near the tips. Other anchors use
  joint/body origins. This is point-to-surface,
  not geom-to-geom distance; it does not depend on a distmax sentinel.
- h: clearance in meters relative to the table, clamped to [-0.1, 1.5]. Negative
  values retain the distinction between above and below the tabletop.
- T: palm position relative to env origin and palm quaternion (wxyz).
- delta_q: measured q minus the actual held target, in identical joint order.
- q, qdot: joint state; object pose and velocities relative to the env frame.
- desired contact: per finger link/palm binary contact and bounded normal force.
- undesired contact: robot-table, robot-self and arm-object, separately filtered.

World axes are shared by the translated envs. Distances are translation-invariant;
absolute positions subtract env_origins. No observation noise for this teacher;
actor enable_corruption remains True for the repository convention, without
noise terms. Critic uses the same privileged information.

Observation size: 258 = 57 distance + 19 height + 7 palm pose + 24 tracking
error + 24 q + 24 qdot + 7 object pose + 6 object velocity + 45 contact bits +
45 bounded force features. The policy is not told when the lift test starts, as
in the reference.

Contact sensors use maxforce reduction per body (representative strongest normal
contact, not a sum of every contact). A second hand-object sensor reports the same
contact force in the world frame for the horizontal grip reward. Undesired
penalties use substep force history to catch brief hits. A force threshold
distinguishes actual load from mere proximity.

## Rewards

Reward terms follow `allegro_teacher/Environment.hpp` and `cfg_reg.yaml`; weights
are per second with scale_rewards_by_dt=True. Every penalty function returns a
nonnegative cost and has a negative weight.

| Term | Weight | Reference term (coeff) |
| --- | ---: | --- |
| Weighted fraction of hand links touching the object | 1.5 | affordance_contact (1.5) |
| Weighted horizontal contact force, capped 5 N per link (thumb 10 N) | 1.0 | affordance_impulse, x-y impulse clipped (1.0) |
| Weighted hand-link distance to the box surface, x17 links | -0.5 | affordance_reward (-0.5 x weighted joint distance x16, computed in `train.py`) |
| Hand link below the tabletop: terminate | -1 once | terminal reward -10, overwritten in `train.py` |
| Hand anchor clearance below 2 cm | -0.1 | table_reward, arm_height (-0.03, -0.05) |
| Robot-table, hand-arm and arm-object contacts | -0.2 | table/arm contact and impulse |
| Object displacement norm from its start | -5 | obj_displacement (-5) |
| Object linear speed squared | -15 | obj_vel (-15) |
| Object angular speed squared | -0.2 | obj_qvel (-0.2) |
| Palm linear speed squared, x10 above 0.25 m/s | -1 | wrist_vel (-1) |
| Palm angular speed squared | -0.1 | wrist_qvel (-0.1) |
| Arm joint speed squared, x4 beyond 0.5 rad/s | -1 | arm_joint_vel (-1) |

Contact weights mirror the reference: palm 0, fingertips x3, thumb links x2, thumb
tip x2 more, normalized to sum 1. Distance weights: palm 0, fingertips x4, thumb
tip x2 more. The reference's push penalty has coefficient 0 and is omitted.

The reference adds the -10 terminal reward in `VectorizedEnvironment`, but
`train.py` then overwrites the reward with the recorded sum, so it never reaches
PPO. A full -10 here would also be about 17 times heavier relative to the dense
return (4 s x 1.5 contact = 6, against 70 x 1.5 = 105 there). The penalty is -1:
small, but it keeps early termination from being attractive while the dense
return is still negative.

## Episode, success and diagnostics

Training episodes last GRASP_TIME = 4 s (80 policy steps, close to the
reference's 70 steps at 5 Hz). Terminations: any hand link below the tabletop,
object dropped off the table (not in the reference) and the time limit.

The lift test episode is GRASP_TIME + 4 s. Success: object rise > 0.10 m at the
end, with no earlier termination (the reference checks `obj z - z0 > 0.1` after
its lift phase). `grasp/evaluate.py` runs it over uniformly sampled placements and
reports the success rate, mean rise, fingers in contact and object displacement at
the end of the grasp phase.

Training logs every weighted reward, fingers in contact and object displacement at
the end of the episode, object speed and peak contact force. Tests cover finite
observations, translation invariance, target holding/clipping/reset, reward signs
and weights, the lift-test arm ramp, partial resets and actual sensor response to
contact.

## Training baseline

PPO: actor/critic MLP 128x128, gamma 0.996, lambda 0.95, learning rate 3e-4,
4 epochs x 4 mini-batches, clip 0.2, value coefficient 0.5, max grad norm 0.5,
adaptive KL 0.01, entropy 0.0, observation normalization, rollout 64 steps.
Action std starts at 1.0 and is floored at 0.2, as the reference's
`enforce_minimum_std`. The hand must be able to preload against contact: a 0.10
rad lead failed the initial physical probe, so the hand target may lead q by up to
0.50 rad.

Start with 256 envs on CUDA; play/probe use 1. CPU tests do not establish CUDA
throughput or learning convergence. No training-success claim without a run.

## Implementation sequence

1. Lock these semantics and record the collision-free pre-grasp/probe outcome.
   The probe starts from the sampled pre-grasp for the nominal placement,
   approaches along a straight IK line, closes and lifts vertically.
2. Add the primitive and reproducible scripted physics probe.
3. Add/register teacher, metrics and targeted tests. Run formatting, lint and
   both type checkers; run relevant tests. GPU validation remains explicit if
   this workstation cannot provide it.

## Commands

- Warp physics: `uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe`
  (CPU default; append `--device cuda:0` on a CUDA machine).
- Native reference: append `--backend native`.
- Train: `uv run train Mjlab-Grasp-Teacher-Ur5e-Rh5dg2`
- Lift test: `uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.evaluate --wandb-run <run>`
- View: `uv run play Mjlab-Grasp-Teacher-Ur5e-Rh5dg2 --agent zero`
- Tests: `uv run pytest tests/test_grasp_teacher.py`

Play runs the lift test with the repository's unlimited-time convention: the arm
raises at 4 s and then keeps holding. The physics probe keeps its own
12-second rollout and 3 s hold check.

Measured results and remaining limits are recorded in
[the validation report](../research/2026-09-20-teacher-physics-validation.md).
