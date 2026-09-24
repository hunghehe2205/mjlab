# Sampled pre-grasp after RobustDexGrasp

Validated 2026-09-24 on MuJoCo native CPU and MuJoCo Warp CPU. Supersedes the
fixed palm-down waypoints in [palm-down pre-grasp](2026-09-24-palm-down-pregrasp.md).

## What changed

Every reset now draws an object placement and a matching collision-free
pre-grasp, following the reset loop of the reference
[`allegro_teacher/train.py`](https://github.com/zdchan/RobustDexGrasp/blob/main/raisimGymTorch/raisimGymTorch/env/envs/allegro_teacher/train.py)
and `helper/initial_pose_final.py`. Code: `grasp/pregrasp.py` (sampling, rolls,
IK, collision check) and `grasp/mdp/events.py` (`PregraspReset`).

| Reference step | Implementation |
| --- | --- |
| Polar object placement, 50% uniform / 50% Beta(0.5, 0.5) edge-biased | Same ranges, mirrored to +Y of the base: angle 0.3π–0.7π, distance 0.45–0.75 m, \|x\| < 0.25 m. Play/eval is uniform. |
| Random yaw, height from the lowest point | Yaw uniform in [-π, π]; box rests on the table. |
| Ray-traced visible points from a virtual camera | Box surface points on faces facing the camera at (-0.035, 0.58, 1.531) m (exact for a convex primitive). |
| Approach from the camera or from the top | Top approach, palm down (`TOP_GRASP`). |
| Hand center 0.25 m from the affordance center | `HAND_CENTER` (box center at pre-close, wrist frame) placed 0.06 m along the approach (see Standoff). |
| 10 roll candidates, projection width | 10 finger-axis directions over the half plane facing away from the robot. |
| UR5 IK per candidate, drop infeasible | Damped least squares on the native model, seeded from the collision-free palm-down branch, within the 0.9 soft limits. |
| Score 5 x width + wrist_2 terms, argmin | Same formula; widths >= 0.18 m excluded unless all are. |
| Fixed finger pre-shape | `OPEN_HAND` (18 joints). |
| Replace colliding resets | Colliding samples are rejected: robot-world contact within 5 mm or robot self-penetration. |
| Resample every iteration | 1024 placements solved once at startup (~3 s); resets draw uniformly. |

The reference's sign flip leaves only five distinct roll directions out of ten
thetas; here the ten directions are distinct. The object-displacement penalty is
measured from each env's stored start position rather than a fixed pose.

## Standoff

The reference's 0.25 m left the lowest hand point 21 cm above the box top and the
starting `reach` reward at 0.005 (5 cm length scale), so early training would be
spent learning to reach. The standoff is 0.06 m, to focus the policy on grasping:

| Standoff | Lowest hand point above box top | Fingertip-surface mean | Start `reach` |
| ---: | ---: | ---: | ---: |
| 0.25 m | 20.9 cm | 26.7 cm | 0.005 |
| 0.15 m | 10.9 cm | 16.9 cm | 0.034 |
| 0.10 m | 5.9 cm | 12.1 cm | 0.088 |
| **0.06 m** | **1.9 cm (min 1.6)** | **8.6 cm** | **0.180** |
| 0.05 m | 0.9 cm (min 0.6) | 7.8 cm | 0.212 |

The lowest point is the thumb tip, which hangs below the palm; 0.05 m leaves less
than the 5 mm collision margin at some placements.

## Pre-shape

The thumb flexion joints sit near their lower soft limits (mcp/pip/dip
0.08/0.06/0.06 rad), so further opening comes from yaw. Yaw swings the thumb from
opposition toward the palm plane. Measured at pre-close:

| Thumb yaw | Angle to palm plane | Thumb-box gap | Probe |
| ---: | ---: | ---: | --- |
| 1.2 | 58° | 4.1 cm | pass |
| 1.0 | 54° | 4.6 cm | pass |
| **0.8 (chosen)** | **47°** | **5.6 cm** | pass |
| 0.6 | 38° | 6.2 cm | pass |
| 0.4 | 29° | 6.9 cm | pass |

The probe's closed pose keeps yaw 1.2, so closing swings the thumb back into
opposition. Below 0.8 the thumb drifts beside the box rather than in front of it.

![Thumb yaw at pre-close, side and front views](assets/2026-09-24-thumb-opening.png)

## Sampled poses

Eight edge-biased pool entries at reset:

![Sampled pre-grasps](assets/2026-09-24-pregrasp-pool.png)

Scripted grasp from pool entry #0 (approach 0–3 s, close 3–5 s, lift 6–8 s):

![Grasp sequence](assets/2026-09-24-pregrasp-sequence.png)

## Physics measurements

The probe starts from the sampled pre-grasp of the nominal placement, approaches
6 cm along a straight line (IK every centimetre), closes and lifts 0.16 m vertically.
Raw reports: [native](assets/2026-09-24-pregrasp-native.json) and
[Warp CPU](assets/2026-09-24-pregrasp-warp.json).

| Metric | Native CPU | Warp CPU |
| --- | ---: | ---: |
| Final lift | 0.1397 m | 0.1426 m |
| Continuous stable hold | 3.0 s | 3.0 s |
| Final object linear speed | 0.00061 m/s | 0.0034 m/s |
| Peak hand/object contact depth | 1.21 mm | 1.10 mm |
| Peak object/table contact depth | 1.82 mm | 2.91 mm |
| Peak other robot contact depth | 0 | 0 |
| Approach hand/object contact depth | 0 | 0 |
| Maximum approach horizontal displacement | <0.000001 mm | <0.00002 mm |

The same scripted grasp succeeded on 12 of 12 random edge-biased placements
(yaw -124° to +177°, distance 0.45–0.73 m), at both 0.25 m and 0.06 m standoff,
with zero approach displacement.

A 16-env, 3-iteration CPU training run completed with resets from the pool;
object displacement penalty and lift-height metric stayed at zero at the start of
episodes, unlike the fixed pose that pressed the box down.

## Validation

`uv run pytest tests/test_grasp_teacher.py` — 13 passed, plus the slow Warp
probe; fast suite 864 passed; `make check` passed. No CUDA throughput or policy
convergence claim is made.

## Remaining limits

- One box primitive; the visibility test assumes a convex object.
- Top approach only; the side (camera-direction) branch exists but has no
  validated rh5dg2 grasp.
- The pool is sampled once per process. Its size (1024) bounds the number of
  distinct placements seen in training.
- `undesired_contact` dominates early training with random actions; check it
  against `reach` around iteration 200.
