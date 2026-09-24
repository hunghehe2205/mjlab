# Teacher model review against RobustDexGrasp

Reviewed: 2026-09-24. Scope: simulator teacher for one fixed box, pre-grasp,
grasp, policy-controlled lift and hold. This review proposes experiments;
it does not change the training configuration or establish convergence.

## Assessment

Keep the current PPO actor/critic MLPs for the first learning baseline. There
is no learning evidence that network capacity is the bottleneck. Exploration,
temporal parameters and the grasp-to-lift reward transition deserve attention
before a larger or recurrent model.

The paper lists two hidden layers of 128 units. Its LSTM reconstructs missing
contacts for the student; this is not a requirement for our privileged teacher.
See [Appendix A3 and student observation](https://arxiv.org/html/2504.05287v1).

## Current implementation

Resolved configuration from
[teacher_env_cfg.py](../../../src/mjlab/tasks/ur5e_rh5dg2/grasp/teacher_env_cfg.py)
and the inherited view-task runner:

| Component | Current value |
| --- | --- |
| Actor | Observation normalization → 259 → ELU 128 → ELU 128 → 24 means |
| Distribution | Diagonal Gaussian; learnable, state-independent std per action |
| Critic | Observation normalization → 259 → ELU 128 → ELU 128 → scalar value |
| Trainable parameters | Actor 52,912 including std; critic 49,921 |
| Initial std / configured floor | 1.0 / no task-specific floor; library default 1e-6 |
| Action | Clip [-1, 1], residual against current q, joint-limit clamp |
| Residual scale | Arm 0.10 rad, hand 0.50 rad per action unit |
| Control / physics interval | 0.05 s / 0.005 s |
| PPO | LR 3e-4, adaptive KL 0.01, clip 0.2, 4 epochs × 4 mini-batches |
| Discount / GAE / entropy | gamma 0.996, lambda 0.95, entropy coefficient 0 |
| Rollout | 256 envs × 64 steps = 16,384 transitions; 3.2 s per env |
| Episode / hold | 12 s / 3 s |

`std_type="scalar"` means direct std parameterization, not one shared scalar
for all joints. Normalization is retained because geometry, joint velocity and
contact features have different scales. The parameter counts exclude running
normalization buffers.

## Differences verified in the released code

The released teacher uses 153 inputs, 22 outputs, LeakyReLU, initial std 1,
and enforces a minimum std of 0.2 after updates. It trains grasp-only; its
evaluation introduces a commanded arm lift. See
[teacher train.py](https://github.com/zdchan/RobustDexGrasp/blob/main/raisimGymTorch/raisimGymTorch/env/envs/allegro_teacher/train.py).

Its current default config has 128×128 actor/critic, a 0.2 s control interval,
70 grasp steps, and residual multipliers 0.005 for arm and 0.015 for fingers.
See [cfg_reg.yaml](https://github.com/zdchan/RobustDexGrasp/blob/main/raisimGymTorch/raisimGymTorch/env/envs/allegro_teacher/cfgs/cfg_reg.yaml).

The C++ controller scales the action, adds current q and clips joint targets;
it does not apply our normalized [-1, 1] action clamp at that point. See
[Environment.hpp](https://github.com/zdchan/RobustDexGrasp/blob/main/raisimGymTorch/raisimGymTorch/env/envs/allegro_teacher/Environment.hpp).

The training script explicitly selects 4 learning epochs and 4 mini-batches;
the PPO implementation defaults to LR 5e-4, entropy coefficient 0 and lambda
0.95 (the script overrides gamma to 0.996). Thus the appendix table and current
released code should not be treated as an identical executable configuration.
See [PPO implementation](https://github.com/zdchan/RobustDexGrasp/blob/main/raisimGymTorch/raisimGymTorch/algo/ppo/ppo.py).

ELU versus LeakyReLU is a small ablation candidate, not evidence of a defect.
The 259-dimensional observation is a deliberate task adaptation, not a failed
attempt to match 153 dimensions across different robots and objectives.

## Priority 1: exploration and executed actions

The current entropy coefficient is zero and the default std floor is only a
numerical safeguard. Exploration can shrink substantially; whether it actually
collapses has not been measured. The original code's exploration floor is an
important difference from merely copying its zero entropy coefficient.

At zero mean, a Gaussian with std 1 exceeds [-1, 1] in 31.73% of samples per
action dimension. This is a mathematical reference, not a measured clipping
rate from a training run. At std 0.5 the fraction is 4.55%; at 0.3 it is 0.086%.

Log raw action saturation before the wrapper clips it, executed target changes,
and std separately for arm and hand. Consider a bounded std range with log
parameterization. Initial std 0.5 and floor 0.05–0.10 are candidates to compare,
not calibrated recommendations or literal copies of the original floor.
Entropy 0 can remain the first baseline if exploration is explicitly monitored;
a small entropy bonus is a separate experiment if std decays prematurely.

Do not reduce the residual scales to the original values without another
physics probe. Here scale also limits available PD tracking error against
contact. The existing hand gain and successful grasp required appreciable
preload. Separate the available target range from the amount of random noise.

## Priority 2: discount and rollout in seconds

Keeping the same gamma while increasing control frequency changes temporal
discounting. For 12 seconds, current gamma gives 0.996^240 = 0.382, whereas
the same gamma at the original interval gives 0.996^60 = 0.786.

To preserve that discount rate in physical time:

    gamma_new = gamma_old ** (dt_new / dt_old)
              = 0.996 ** (0.05 / 0.2) = 0.9989985

Gamma approximately 0.999 is therefore a reasoned ablation for the added lift
and hold task. Gamma 0.996 is valid, but it is not temporally equivalent to the
original configuration. Neither value is proven optimal for this task.

GAE's gamma×lambda decay time is approximately 0.90 s currently versus 3.62 s
at a 0.2 s interval with the same parameters. Preserving lambda's time scale
would give 0.95^(1/4) = 0.98726; changing it also changes estimator variance.
Test higher lambda separately rather than assuming exact time rescaling is best.

The current rollout spans 3.2 s, while approach, grasp, lift and hold span much
more. PPO bootstrapping allows this; a rollout need not contain a whole episode.
Still, 128 steps (6.4 s) or 256 (12.8 s) are useful comparisons if terminal
success propagates poorly. Compare equal environment transitions, not equal
iteration counts, since rollout length changes samples per update and memory.

## Priority 3: observations and grasp-to-lift learning

The current state already contains joint position/velocity, tracking error,
object pose/velocity, geometric distances, contact features and hold progress.
With q and delta_q, the held target can be recovered as q - delta_q. A raw
previous-action vector or LSTM is not automatically necessary.

Potential small additions are explicit lift-height error and object-table
contact/normal force. Height is already derivable from object pose for the fixed
goal, but table contact is a success input not directly included in the actor's
current vector. These features could make the lift/hold decision easier to learn;
their benefit remains an experiment.

Retain privileged object velocity for the teacher. Do not remove useful state
just to match the original input size. Analytic oriented-box surface distances
are appropriate for this primitive; they will need generalization when the
object geometry changes. World-oriented relative vectors are consistent with
the current shared axes and gravity; a palm-frame rewrite is not required now.

The lift reward is gated by contact on at least two fingers. Until that happens,
it provides no lift incentive. If training gets stuck, inspect reach, contact,
self-collision and actual lift events before concluding that the MLP is too small.
Split the three undesired-contact groups in diagnostics. If exploration remains
the obstacle, warm-starting from recorded successful probe transitions or a
shorter initial hold curriculum is a fallback experiment, not part of the paper
reproduction. Final evaluation must retain the full 10 cm / 3 s criterion.

## Suggested experiment order

1. Keep 128×128 ELU, normalization, 24-dimensional joint residual actions and
   the successful collision/gain setup. Add the missing exploration/reward logs
   and reject unsupported action config options discussed in the earlier review.
2. Compare the current exploration config with one explicit std configuration.
3. With that choice fixed, compare gamma 0.996 versus approximately 0.999.
4. If needed, compare rollout 64 versus 128, then lambda or reward curriculum.
5. Only after these diagnostics, compare 128×128 versus 256×256 at an equal
   transition budget and preferably at least three seeds.

Evaluate deterministic mean-action success separately from noisy training
success. Record success, lift onset, longest hold, drop rate, collision penalties,
action clipping, std, KL and value diagnostics. There is no new convergence
result in this review and no reason yet to require a Transformer, PointNet,
recurrent teacher or separate arm/hand policies.
