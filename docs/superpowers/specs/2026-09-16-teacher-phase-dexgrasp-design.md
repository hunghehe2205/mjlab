# Teacher grasp-and-lift — UR5e + rh5dg2 / mjlab

Updated: 2026-09-20. Supersedes the grasp-only/scripted-lift draft.

## Scope and acceptance

Teacher controls all 24 arm/hand joints from a fixed pre-grasp, grasps one
primitive, lifts it by 0.10 m and holds it for 3 continuous seconds. Success
terminates the episode. Transport, lowering, release, student, domain
randomization and real deployment are outside this implementation.

This is an adaptation of RobustDexGrasp: its original teacher trains grasp-only
and uses a separately commanded lift for evaluation. Our lift is part of the
policy's objective. A scripted close/lift is only a physics acceptance probe;
it must not override policy actions during training.

## Scene and physics baseline

- Existing UR5e/rh5dg2 entity and workstation; fixed pre-grasp and object pose.
- MVP object: box, half extents (0.03, 0.03, 0.06) m, mass 0.08 kg,
  friction 1.0, condim 4.
  Position (-0.10, 0.48, TABLE_TOP_Z + 0.06), identity orientation.
- Keep robot actuator gains/armature from the entity configs. Build the probe
  from these configs too, avoiding the standalone viewer's missing hand armature.
- Teacher-local collision variant: replace DIP cylinder pads with inscribed
  capsules/spheres. Their poses, body inertias and visual meshes are preserved;
  the shared robot asset is untouched. Cylinder-box contacts in this Warp version
  support only one contact and the original pads produced backend-sensitive
  grasps. This is an explicit collision approximation, not real-hand calibration.
- implicitfast, elliptic cone, impratio 10; physics timestep 0.005 s.
- Policy interval 0.05 s (20 Hz), decimation 10. This deliberately differs from
  the paper's 5 Hz to allow finer approach/lift corrections. Episode 12 seconds.
- All object motion must result from gravity, contact and robot actuation:
  no object weld, teleport, gravity cancellation or attachment during rollout.
- Probe must start without penetration and demonstrate >=0.10 m lift, >=3 s hold
  with hand contact and no table support. Record the backend and measured result.

## Actions and reset

Use one task-local action term with 24 normalized actions, clipped to [-1, 1].
At each policy step, compute once:

    target = clamp(q_current + action * scale, hard_joint_limits)

Arm scale 0.10 rad, hand scale 0.50 rad per policy step. Hold this target through
all physics substeps; do not recompute relative to current q every substep.
The target is in radians and stored explicitly. Reset raw action to zero and
target to the reset joint pose, including partial environment resets.

Reset robot, props and object with env origins; restore robot joints and zero
velocities. Reset success counters and action target for exactly the selected
envs. z0 is the nominal resting box center on the table, not a noisy contact
transient. Joint limits and pre-grasp are checked by the physics probe.

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
- hold progress: consecutive successful control steps / required hold steps.

World axes are shared by the translated envs. Distances are translation-invariant;
absolute positions subtract env_origins. No observation noise for this teacher;
actor enable_corruption remains True for the repository convention, without
noise terms. Critic uses the same privileged information.

Observation size: 259 = 57 distance + 19 height + 7 palm pose + 24 tracking
error + 24 q + 24 qdot + 7 object pose + 6 object velocity + 45 contact bits +
45 bounded force features + 1 hold progress.

Contact sensors use maxforce reduction per body (representative strongest normal
contact, not a sum of every contact). Desired contacts are instantaneous for
success; undesired penalties can use substep force history to catch brief hits.
Use a force threshold to distinguish actual load from mere proximity.

## Rewards

Every penalty function returns a nonnegative cost and has a negative manager
weight. Every incentive returns a nonnegative score and has a positive weight.
Use scale_rewards_by_dt=True. Weights are task baselines, not paper reproductions.

- Reach: bounded exponential of mean fingertip-to-surface distance.
- Contact: mean contact score with bounded force contribution; total reward kept
  smaller than the lift/hold incentive. Never reward unlimited squeezing.
- Lift: clip((z-z0)/0.10, 0, 1), gated by hand support.
- Hold: height reached, low linear/angular speed, multiple hand contacts,
  no table support. Bonus for success larger than remaining positive dense return
  so early completion is preferable to delaying termination.
- Height cost: squared shortfall below 2 cm for selected hand anchors, excluding
  fingertips. No log of zero/negative values. No penalty on fixed arm base height.
- Undesired contact: positive contact/force cost with negative weight.
- Horizontal object displacement/velocity and arm/hand motion regularization.
  No penalty on intentional vertical object displacement relative to reset.

Approach/grasp/lift rewards coexist; no hard grasp gate disables arm motion.
Counters advance exactly once per control step, not when observations/logging
are queried. Invalid hold resets the counter. Partial reset cannot affect others.

## Success and diagnostics

Success requires rise >=0.10 m, linear speed <=0.05 m/s, angular speed <=1 rad/s,
contact on at least two distinct fingers, and no object-table support for 3 s.
A dropped object below the tabletop or outside the table footprint fails;
12 s is a timeout. A thrown object or touching the height threshold once is not
success. Success and failure are true terminations, time limit is truncation.

Log each weighted reward (RewardManager), lift height, object speed, contact force,
hold progress and success rate (MetricsManager). Tests cover finite observations,
translation invariance, target holding/clipping/reset, penalty signs, continuous
hold/no-contact rejection, partial resets and actual sensor response to contact.

## Training baseline

PPO: actor/critic MLP 128x128, gamma 0.996, lambda 0.95, learning rate 3e-4,
4 epochs x 4 mini-batches, clip 0.2, value coefficient 0.5, max grad norm 0.5,
adaptive KL 0.01, entropy 0.0, observation normalization, rollout 64 steps.
The hand target scale must allow preload against contact: a 0.10 rad
limit failed the initial physical probe; the final baseline uses 0.50 rad.

Start with 256 envs on CUDA; play/probe use 1. CPU tests do not establish CUDA
throughput or learning convergence. No training-success claim without a run.

## Implementation sequence

1. Lock these semantics and record the collision-free pre-grasp/probe outcome.
2. Add the primitive and reproducible scripted physics probe.
3. Add/register teacher, metrics and targeted tests. Run formatting, lint and
   both type checkers; run relevant tests. GPU validation remains explicit if
   this workstation cannot provide it.

## Commands

- Warp physics: `uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe`
  (CPU default; append `--device cuda:0` on a CUDA machine).
- Native reference: append `--backend native`.
- Train: `uv run train Mjlab-Grasp-Teacher-Ur5e-Rh5dg2`
- View: `uv run play Mjlab-Grasp-Teacher-Ur5e-Rh5dg2 --agent zero`
- Tests: `uv run pytest tests/test_grasp_teacher.py`

Play uses the repository's unlimited-time convention; success/failure still end
an episode. Training and physics acceptance have the 12-second timeout.

Measured results and remaining limits are recorded in
[the validation report](../research/2026-09-20-teacher-physics-validation.md).
