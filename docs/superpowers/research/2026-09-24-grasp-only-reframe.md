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
  termination is a hand joint below the table (reward -10).
- Recorded rewards: affordance contact (1.5), x-y contact impulse clipped per
  link (1.0), table contact/impulse, arm contact/impulse, object displacement
  (-5), object speed (-15), object angular speed (-0.2), wrist linear/angular
  speed (-1/-0.1), arm joint speed (-1). The config's distance reward (0.5) is not
  recorded; the push penalty is 0.
- Evaluation appends 100 steps: `switch_root_guidance` interpolates the arm joints
  over 80 steps toward a raised pose while the policy keeps driving the fingers.
  Success is `obj z - z0 > 0.1`.

## Adaptation

- Grasp episode 4 s (80 steps at 20 Hz). The reward table is in the spec.
- Lift test: at 4 s the arm ramps over 3 s by a per-placement joint offset that
  raises the pre-grasp wrist 0.20 m (IK solved with the pool), then holds for
  1 s. The policy keeps the hand. `grasp/evaluate.py` reports the success rate over
  uniformly sampled placements; `play` shows the same episode.
- The reference's home-pose lift target is replaced by a vertical raise, because
  the fixed home pose of this arm would also drag the box sideways.

## Checks

With zero actions, the lift test raised the palm 0.18 m over the ramp (the
held-target arm sags about 7 cm during the 4 s grasp phase, which the policy
compensates). `evaluate.py` ran end to end on a 3-iteration CPU checkpoint
(success 0, as expected). The scripted probe still lifts 0.143 m (Warp CPU) and
holds 3 s. No policy result for the grasp-only reward exists yet.
