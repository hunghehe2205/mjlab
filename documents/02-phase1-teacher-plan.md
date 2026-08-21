# Phase 1 — Teacher policy (privileged PPO)

Mục tiêu: teacher grasp được 3–5 object trên UR5e + RH5-DG2 trong mjlab,
eval lift success > ~80% trước khi sang Phase 2.

Cấu trúc code đích:

```
src/mjlab/tasks/dexgrasp/
  __init__.py
  dexgrasp_env_cfg.py                  # make_dexgrasp_env_cfg()
  mdp/{__init__,observations,rewards,terminations,events}.py
  pregrasp/{ik_ur5e.py,pose_sampler.py}
  rl/{__init__}.py
  config/ur5e_rh5dg2/{__init__,env_cfgs,rl_cfg}.py   # register_mjlab_task
assets: src/mjlab/asset_zoo/objects/dexgrasp/...
```

## A. Assets

- [ ] Thêm site `grasp_center` vào lòng bàn tay `right_hand.xml` (tương đương
  `hand_center` gốc `[-0.0091, 0, -0.095]` của Allegro — đo lại offset từ mesh
  RH5-DG2, trục x hướng ra khỏi lòng bàn tay theo convention pre-grasp).
- [ ] Thêm collision geom (class `rh5dg2_hand_collision`) cho 6 body
  `R_*_force_sensor` — pad tiếp xúc hiện chỉ có visual mesh.
- [ ] Định nghĩa `INIT_FINGER_POSE` cho 18 khớp (ngón hé mở, thumb đối diện —
  vai trò như `init_finger_pose` 16 dim của Allegro); thêm vào keyframe/InitialStateCfg.
- [ ] Thêm `ACTION_SCALE` cho robot (arm 0.005, finger 0.015 rad).
- [ ] Chọn 3–5 object phase 1: box + cylinder (procedural spec_fn) + 2–3 mesh
  YCB lấy từ `rsc/new_training_set` repo gốc (vd banana, potted_meat_can).
- [ ] Script precompute per-object: 200 surface-sample points (trimesh) +
  lowest point + centroid → lưu `.npz` cạnh MJCF.
- [ ] Mesh → MJCF: convex decomposition (coacd) cho collision, freejoint,
  material `object`; wrap thành `EntityCfg`/`VariantEntityCfg`.
- [ ] Mở rộng `tests/test_ur5e_rh5dg2_constants.py`: site mới, collision pads.

## B. Scene & task skeleton

- [ ] `SceneCfg`: bàn = box tĩnh (mặt bàn ~0.771 m, friction 0.2), UR5e mount
  trên mặt bàn tại gốc, object entity dùng `VariantEntityCfg` (mỗi world một
  mesh, chia đều).
- [ ] `ContactSensor`: palm + 3 link cuối mỗi ngón + 6 pad (match theo body
  regex), lấy `found`/`force` per-link so với object; sensor riêng cho
  ngón↔bàn và arm↔{bàn, object}.
- [ ] Sim cfg: `timestep=0.01`, `decimation=20` (policy 5 Hz), `impratio=10`,
  `cone="elliptic"`, tăng `nconmax`/`njmax`; episode 70 bước (14 s).
- [ ] Đăng ký task `Mjlab-DexGrasp-UR5eRH5DG2` qua `register_mjlab_task`.

## C. Reset & pre-grasp (event terms, CPU per-reset như bản gốc)

- [ ] Event sample pose object: phân cực r ∈ [0.45, 0.75], góc ∈ [−0.7π,
  −0.3π], |x| < 0.25, xoay z ngẫu nhiên, z = mặt bàn − lowest_point.
  (Phase 1 dùng uniform; edge-biased Beta để Phase 2.)
- [ ] Visible points: raycast trimesh từ camera cố định (dùng lại vị trí camera
  gốc, quy đổi hệ tọa độ scene mình) → tâm vùng nhìn thấy + hướng tiếp cận.
- [ ] Port `sample_rot_mats` (sample N hướng quanh trục tiếp cận + projection
  width) — giữ nguyên logic, chạy batch numpy.
- [ ] **IK giải tích UR5e**: port `inverseKinematicsUR5.py`, thay DH parameters
  UR5 → UR5e; unit test forward-check (FK(IK(pose)) ≈ pose) trên lưới pose
  quanh workspace.
- [ ] Scoring nghiệm IK: projection width < 0.18 (coeff 5) + wrist angle
  (coeff 1) → chọn best; set qpos arm + `INIT_FINGER_POSE`.
- [ ] Check collision sau reset (contact sensor arm), fallback: copy pose env
  khác cùng object hoặc pose cứng.
- [ ] Gọi `sim.forward()` sau khi ghi pose (idiom `_pending_forward` của
  lift-cube).

## D. Actions

- [ ] `RelativeJointPositionActionCfg`, scale arm 0.005 / finger 0.015, clip
  theo soft joint limits.
- [ ] (Tùy chọn, bám gốc) random 1-step action delay — có thể bỏ ở teacher
  phase 1, thêm ở student.

## E. Observations (group `actor` = `critic`, đều privileged)

- [ ] Proprio: qpos (24) + sai số PD `target − qpos` (24).
- [ ] Contact flags (16) + impulse magnitude (16) từ ContactSensor
  (ngưỡng calibrate lại cho MuJoCo).
- [ ] Chiều cao 24 keypoint + 6 arm link so mặt bàn.
- [ ] `grasp_center` pos (3), euler wrist (3), euler diff so init (3, unwrap).
- [ ] **af_vec (72)**: term GPU thuần torch — transform 200-point cloud theo
  object pose hiện tại, `torch.cdist` với 24 keypoint (wrist + 18 khớp + 5
  tip), lấy vector tới điểm gần nhất, xoay về trục world.
- [ ] Unit test af_vec trên case nhỏ tính tay.

## F. Rewards (`mdp/rewards.py`, giữ coeff gốc làm baseline)

- [ ] affordance distance: −Σ wᵢ·min_dist (tip ×4, thumb tip ×8, palm 0) — coeff 0.5.
- [ ] affordance contact (weighted, tip ×3, thumb ×2) — coeff 1.5.
- [ ] affordance impulse xy (clip 0.1 link thường / 0.2 tip) — coeff 1.0.
- [ ] table contact/impulse (−1.0 / −0.5) + table log-barrier (−0.03).
- [ ] arm: height log-barrier (−0.05), contact/impulse (−0.1), collision (−1.0).
- [ ] object: vel (−15), qvel (−0.2), displacement (−5).
- [ ] wrist vel/qvel (−1.0/−0.1, ×10 khi >0.25), arm joint vel (−1.0, ×4 khi >0.5).
- [ ] Reward tổng clip min −2 (hook trong env hoặc term riêng).

## G. Termination & eval

- [ ] `time_out` 70 bước; terminate khi keypoint tay dưới mặt bàn; NaN guard.
- [ ] Metric success: script eval riêng — sau 70 bước grasp, arm target nội suy
  về pose nâng trong 80–100 bước (5 Hz), success = object z tăng > 0.1 m;
  log per-object success rate.

## H. RL config & train

- [ ] `RslRlOnPolicyRunnerCfg`: MLP 128×128, γ=0.996, λ=0.95, 4 epochs, 4
  minibatches, obs normalization, min action std 0.2.
- [ ] Smoke train (~vài trăm iter, ít env): reward tăng, không NaN, contact
  xuất hiện.
- [ ] Train đầy đủ + đánh giá: tune PD gains ngón / ngưỡng contact nếu grasp
  bất ổn.

## Thứ tự làm việc gợi ý

A (assets) → B (skeleton chạy được với action zero) → C (reset/pre-grasp,
verify bằng viewer thấy tay đặt đúng pre-grasp quanh object) → D+E → F+G →
H. Mỗi mốc verify bằng viewer/`uv run play` trước khi sang bước sau.
