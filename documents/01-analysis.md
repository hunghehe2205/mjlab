# Phân tích RobustDexGrasp và mapping sang UR5e + RH5-DG2

> **Snapshot thiết kế ban đầu.** Các thông số `0.01/20`, standoff `0.25` và
> danh sách “chưa có” bên dưới mô tả paper hoặc trạng thái repo tại thời điểm
> khảo sát. Cấu hình runtime hiện tại nằm ở
> [02-phase1-teacher-plan.md](02-phase1-teacher-plan.md).

Phân tích dựa trên code thực tế của repo (không chỉ paper):
`raisimGymTorch/env/envs/allegro_teacher/`, `allegro_student/`,
`env/RaisimGymVecEnvOther.py`, `helper/initial_pose_final.py`.

## 1. Hệ gốc

- UR5 (6 DoF) + Allegro hand (16 DoF, 4 ngón) trên RaiSim. Action 22 chiều.
- Object trên bàn (mặt bàn z ≈ 0.771 m), camera cố định tại
  `[0.035, -0.58, 1.531]`, arm mount trên mặt bàn.
- 35 object train (YCB + tự quét), mỗi object 2 env, các object khó lặp thêm.
- Policy chỉ là **MLP 128×128** — không có PointNet; hình học được "tiêu hóa"
  qua biểu diễn distance-vector.

## 2. Cấu trúc một episode

1. Sample vị trí object: phân cực, khoảng cách r ∈ [0.45, 0.75], góc ∈
   [−0.7π, −0.3π], ràng buộc |x| < 0.25. 50% sample edge-biased bằng
   Beta(0.5, 0.5) (curriculum "mixed sampling"). Xoay z ngẫu nhiên.
2. **Single-view cloud**: ray-cast (trimesh, CPU) từ camera tới 200 điểm
   surface-sample → tập điểm nhìn thấy. Chụp **một lần lúc reset**, không cập
   nhật trong episode.
3. **Pre-grasp bằng IK**:
   - Hướng tiếp cận x̂ = từ tâm vùng nhìn thấy về phía camera (hoặc thẳng đứng
     nếu top-grasp). Vị trí cổ tay = tâm + 0.25 m dọc x̂.
   - Sample 10–20 hướng xoay quanh trục tiếp cận (`sample_rot_mats`), chấm
     điểm: bề rộng object chiếu lên trục kẹp < 0.18 m (coeff 5) + góc wrist
     đẹp (coeff 1), giải **IK giải tích UR5** (`findClosestIK`), chọn nghiệm
     điểm tốt nhất → set thẳng qpos arm. Ngón về `init_finger_pose`.
   - Check self-collision (4 flag contact arm trong global state); nếu kẹt thì
     copy pose từ env cùng object hoặc fallback pose cứng.
4. Rollout **70 bước @ 5 Hz** (control_dt 0.2 s, sim_dt 0.01 s → 20 substeps).
5. Eval (mỗi 100 iter): thêm 100 bước lift — arm target nội suy tuyến tính về
   pose nâng `theta0` trong 80 bước (`switch_root_guidance`). Success = object
   nâng > 0.1 m.

## 3. Action & control

- Action = **delta joint targets** quanh vị trí hiện tại: mỗi bước
  `target = qpos_hiện_tại + action * std`, clip theo joint limit.
- std: finger 0.015 rad, arm 0.005 rad (rất nhỏ → chuyển động chậm, ổn định).
- PD control; random **1-step action delay** (50% mỗi bước giữ target cũ).
- Min action std của policy giữ ở 0.2 (enforce mỗi update).

## 4. Observation

### Teacher (153 = 102 C++ + 51 Python)

Phần C++ (102), theo thứ tự:

| Term | Dim | Ghi chú |
|---|---|---|
| qpos | 22 | arm 6 + ngón 16 |
| sai số PD `target − qpos` | 22 | proxy cho torque |
| contact flags 13 link tay | 13 | impulse norm > 0.01 |
| impulse magnitude 13 link | 13 | với object (phần affordance) |
| chiều cao 17 keypoint so mặt bàn | 17 | z − 0.771 |
| chiều cao 6 arm link | 6 | |
| hand center (world) | 3 | điểm giữa lòng bàn tay |
| euler wrist so với init | 3 | unwrap liên tục |
| euler wrist hiện tại | 3 | |

Phần Python (51) — **biểu diễn hand-centric, đóng góp chính của paper**:
17 keypoint tay (wrist + 16 khớp) → tìm điểm gần nhất trong 200-point cloud
(`torch.cdist`) → vector 3D `điểm − keypoint` (xoay về hệ trục world) → 51 dim.
Teacher dùng **full cloud** bám theo object pose thật (privileged).

### Student

- MLP input 153 giống teacher, nhưng 51 dim af_vec tính từ **cloud single-view
  tĩnh** chụp lúc reset (không bám object khi object xê dịch).
- 26 dim contact+impulse (không đo được trên robot thật) → **LSTM
  reconstruction** từ lịch sử 10 bước × 44 dim (qpos + sai số PD), latent 26.
- Obs có **delay ngẫu nhiên** (lag 1–2 bước).
- Domain randomization: friction {0.5–0.9}, PD gain tay ±10% / arm ±5%, nhiễu
  qpos tay ±0.02 rad / arm ±0.005 rad, nhiễu frame pos ±0.01 m / orient ±0.02
  rad. Tùy chọn `biased`: dịch object ±5 cm một lần khi tay lại gần (< 7 cm) —
  giả lập sai số perception.

## 5. Reward

Phần C++ (coeff trong `cfg_reg.yaml`):

| Term | Coeff | Ghi chú |
|---|---|---|
| affordance_contact | +1.5 | contact có trọng số (đầu ngón ×3, thumb ×2, palm 0) |
| affordance_impulse | +1.0 | impulse **phương ngang xy**, clip [0, 0.1–0.2] |
| table_contact / table_impulse | −1.0 / −0.5 | ngón chạm bàn |
| arm_contact / arm_impulse | −0.1 / −0.1 | arm chạm bàn/object |
| obj_vel / obj_qvel | **−15** / −0.2 | phạt rất nặng làm object di chuyển |
| obj_displacement | −5.0 | dịch so vị trí đầu |
| wrist_vel / wrist_qvel | −1.0 / −0.1 | vel > 0.25 m/s bị ×10 |
| arm_joint_vel | −1.0 | joint vel > 0.5 bị ×4 |
| push (ép object xuống bàn) | 0 (tắt) | |

Phần Python (tính từ obs):

| Term | Coeff | Công thức |
|---|---|---|
| affordance | +0.5 | −Σ wᵢ·dist(keypointᵢ, điểm gần nhất); tip ×4, thumb tip ×8 |
| table | −0.03 | −Σ w·log(50·clip(joint_height, 0.002, 0.02)) — log-barrier |
| arm_height | −0.05 | log-barrier tương tự cho 4 arm link cuối |
| arm_collision | −1.0 | số flag self-collision/chạm bàn của arm |

Reward tổng clip min = −2.

## 6. Training

- **Teacher**: PPO thuần (custom impl), γ=0.996, λ=0.95, 4 epochs, 4
  minibatches, reload checkpoint khi exploding gradient. Không normalize
  obs/reward: train.py tạo VecEnv với `normalize_ob=False` mặc định, cờ
  `normalize_rew` không được dùng (đính chính; xem 08 §6).
- **Student**: DAgger (imitate teacher trên cùng state) + PPO, trộn theo
  curriculum `ppo_ratio = min(iter·0.0005, 1)` (thuần imitation → thuần RL
  trong 2000 iter), cộng loss reconstruction cho LSTM. Value function dùng obs
  privileged. Student khởi tạo từ weights teacher.

## 7. Mapping sang UR5e + RH5-DG2

### Giữ gần như nguyên

- Toàn bộ logic pre-grasp (UR5e chỉ khác UR5 vài mm DH parameter — port
  analytic IK và sửa DH), biểu diễn distance-vector, cấu trúc reward, delta
  action 5 Hz, teacher→student DAgger, LSTM recon, mixed curriculum.

### Phải thiết kế lại

| Thành phần | Allegro (gốc) | RH5-DG2 (mình) |
|---|---|---|
| Action | 6 + 16 = 22 | 6 + 18 = **24** |
| Ngón | 4 ngón × 4 khớp | thumb/index/middle: yaw+mcp+pip+dip; ring/pinky: mcp+pip+dip |
| Keypoints af_vec | 17 (wrist + 16 khớp) | **24** (wrist + 18 khớp + 5 tip) → af_vec 72 dim |
| Contact links | 13 (palm + 3/ngón) | **16** (palm + 3 link/ngón × 5) |
| Tactile thật | không → LSTM recon toàn bộ | **có 5 force sensor đầu ngón + 1 palm** → Phase 3 đo trực tiếp một phần |
| `hand_center`, `init_finger_pose` | đo cho Allegro | đo lại cho RH5-DG2 |
| Simulator | RaiSim, CPU threads | mjlab/MuJoCo-warp, GPU-vectorized |

### Hạ tầng mjlab tương ứng (đã khảo sát)

| Nhu cầu | mjlab có sẵn |
|---|---|
| Task template | `tasks/manipulation` (lift-cube YAM): manager-based, `impratio=10`, `cone="elliptic"` |
| Mỗi env một object khác nhau | `VariantEntityCfg` (per-world mesh variants; cùng cấu trúc kinematic, assignment cố định lúc init) |
| Contact/impulse per link | `ContactSensor` (`ContactMatch` regex theo body, có `force`, `found`, history) |
| Point cloud / raycast | `RayCastSensor` (`hit_pos_w`), `CameraSensor` depth + `unproject_depth` (Phase 3) |
| DR | `mjlab.envs.mdp.dr`: `geom_friction`, `pd_gains`, `body_mass`, … |
| Delta action | `RelativeJointPositionActionCfg` |
| Obs delay/history/noise | `ObservationTermCfg(delay=, history=, noise=)` |
| PPO | rsl_rl 5.4.2 (`RslRlOnPolicyRunnerCfg`), train qua `uv run train <task-id>` |
| Distillation | rsl_rl có `DistillationRunner`; mjlab chưa có cfg wrapper → tự thêm (hook `runner_cls` đã có) |

### Thiếu, phải bổ sung

1. `right_hand.xml`: body `R_*_force_sensor` (pad đầu ngón) **chỉ có visual
   mesh, không có collision geom** → pad không va chạm được. Chưa có site
   `grasp_center` ở lòng bàn tay (tương đương `hand_center`).
2. Chưa có `ACTION_SCALE` / `init_finger_pose` cho robot này.
3. Chưa có pipeline object mesh → MJCF (surface sample 200 điểm, lowest point,
   convex decomposition).
4. Chưa có IK giải tích UR5e trong repo.

## 8. Rủi ro chính

1. **PD gains ngón** hiện uniform (stiffness 1.0, damping 0.1) — chưa chắc đủ
   lực bóp/ổn định khi grasp; sẽ phải tune khi thấy grasp bất ổn.
2. **IK UR5e**: DH khác UR5 vài mm; sai DH → pre-grasp lệch hệ thống.
3. **Contact solver MuJoCo vs RaiSim**: impulse scale khác nhau → ngưỡng
   contact (0.01) và clip impulse (0.1/0.2) cần calibrate lại.
4. `nconmax`/`njmax` phải tăng nhiều cho tay 5 ngón + mesh object.
5. Ring/pinky không có khớp yaw → khả năng "ôm" vật khác Allegro; trọng số
   finger có thể phải điều chỉnh.
