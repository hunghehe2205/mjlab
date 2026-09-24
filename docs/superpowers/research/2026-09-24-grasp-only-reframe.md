# Grasp-only reframe after RobustDexGrasp

2026-09-24. Replaces the grasp-and-lift reward and 3 s hold success with the
reference teacher's grasp-only objective and its lift test. Design:
[spec](../specs/2026-09-16-teacher-phase-dexgrasp-design.md).

## Why

W&B run `dexgrasp-teacher/standoff6cm-v1` (grasp-and-lift, 512 envs), iteration
~400:

| Signal | Value |
| --- | ---: |
| `Episode_Reward/lift` (max 6) | 4.45, lifted with 2+ fingers ~74% of the episode |
| `Episode_Reward/hold` (max 2) | 0.45, all hold conditions true ~22% of the time |
| `Episode_Metrics/lift_height` | 0.087 m |
| `Episode_Metrics/object_speed` | 0.09 m/s |
| `Episode_Metrics/hold_progress`, `success` | 0.006, 0 |

The video at iteration 403 shows the box held up for most of the episode. The
policy had learned to grasp and lift, but the success check (rise >= 0.10 m,
speed <= 0.05 m/s, angular speed <= 1 rad/s, 2 fingers, no table, all for 3 s
without a break) never held for 60 consecutive steps, and the lift reward
saturated exactly at the success height. The reference instead trains grasping
only and checks a scripted lift.

## Reference behaviour

From `allegro_teacher/Environment.hpp`, `cfg_reg.yaml`, `train.py` and
`quantitative_eval.py`:

- Training runs 70 steps at 5 Hz per episode, no lift, no success term. The only
  termination is a hand joint below the table. Its -10 terminal reward is added in
  `VectorizedEnvironment` but overwritten by `train.py`, so PPO never sees it.
- Rewards from `Environment.hpp`: affordance contact (1.5), x-y contact impulse
  clipped per link (1.0), table contact/impulse, arm contact/impulse, object
  displacement (-5), object speed (-15), object angular speed (-0.2), wrist
  linear/angular speed (-1/-0.1), arm joint speed (-1); the push penalty is 0.
  `train.py` adds the weighted joint-to-object distance (-0.5), table and arm height
  log barriers and arm collision (-1). (An earlier version of this note wrongly
  said the distance term was unused.) Its `reward_r.clip(min=-2)` is not assigned
  and has no effect.
- Actions: target = measured q + a x (0.005 rad arm, 0.015 rad hand) per 0.2 s,
  Gaussian actions with std starting at 1.0 and floored at 0.2, and a random
  one-substep actuation delay.
- Evaluation appends 100 steps: `switch_root_guidance` interpolates the arm joints
  over 80 steps toward a raised pose while the policy keeps driving the fingers.
  Success is `obj z - z0 > 0.1`.

## Adaptation

- Action speed: the first grasp-only version kept 0.10/0.50 rad per 0.05 s
  (2/10 rad/s per unit action, 80-130 times the reference) with no std floor, so
  early exploration slammed the hand into the box that starts 2 cm away. Now
  targets accumulate at 0.01/0.03 rad per step (0.2/0.6 rad/s), lead q by at most
  0.10/0.50 rad, std is floored at 0.2, and the actuation delay is reproduced.
- Terminal penalty -1 instead of -10 (see the spec); distance penalty -0.5 replaces
  the positive exponential reach reward.
- Grasp episode 4 s (80 steps at 20 Hz). The reward table is in the spec.
- Lift test: at 4 s the arm ramps over 3 s by a per-placement joint offset that
  raises the pre-grasp wrist 0.20 m (IK solved with the pool), then holds for
  1 s. The policy keeps the hand. `grasp/evaluate.py` reports the success rate over
  uniformly sampled placements; `play` shows the same episode.
- The reference's home-pose lift target is replaced by a vertical raise, because
  the fixed home pose of this arm would also drag the box sideways.

## Checks

With the first (q-relative) action term and zero actions, the arm sagged about
7 cm during the 4 s grasp phase and the lift test raised the palm only 0.18 m.
With accumulated targets the palm settles 4 mm and the lift test raises it
0.20 m. With the new action limits the scripted
probe (approach 3 s, close 2 s, lift over 3 s, 14 s rollout) still lifts 0.140 m
(native) and 0.143 m (Warp CPU) and holds 3 s. `evaluate.py` ran end to end on a
3-iteration CPU checkpoint (success 0, as expected). No policy result for the
grasp-only reward exists yet.
