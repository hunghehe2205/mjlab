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

- [x] Thêm site `grasp_center` vào lòng bàn tay `right_hand.xml` — palm-local
  `[0.05, 0, 0.11]`, x hướng ra lòng bàn tay. Ước lượng đầu, refine ở §C viewer.
- [x] Thêm collision geom (class `rh5dg2_hand_collision`) cho 6 body
  `R_*_force_sensor` — dùng chính mesh pad (convex hull).
- [x] Định nghĩa `INIT_FINGER_POSE` cho 18 khớp (cupped, thumb đối diện) — hằng
  số trong constants, verify self-collision-free. (Chưa nhét vào keyframe: reset
  event ở §C sẽ set; home keyframe giữ ngón mở = 0.)
- [x] Thêm `ACTION_SCALE_ARM=0.005`, `ACTION_SCALE_FINGER=0.015` + nhóm tên khớp.
- [x] Chọn 5 object gần-lồi: box + cylinder (procedural) + potted_meat_can,
  tomato_soup_can, sugar_box (mesh YCB). Banana bỏ (lõm nhẹ → hull sai).
- [x] Script `precompute.py`: 200 surface points + normals + centroid + lowest
  → `.npz` cạnh mesh.
- [x] Mesh → MJCF `spec_fn`: collision = **convex hull** (không coacd, xem
  problems/coacd), freejoint, material `object`; wrap `EntityCfg` + registry
  `PHASE1_OBJECTS`. (`VariantEntityCfg` để §B khi dựng scene multi-object.)
- [x] Test: mở rộng `test_ur5e_rh5dg2_constants.py` + mới `test_dexgrasp_objects.py`.

## B. Scene & task skeleton

- [~] `SceneCfg`: bàn box tĩnh (top 0.771 m, friction 0.2) + **bục riêng** cho
  UR5e (base flush mặt bàn z=0.771, khe ~8cm) + object. Fixed-base entity
  auto-mocap → reset về env_origins. Arm menagerie actuator adopt qua
  `XmlActuatorCfg` (action điều khiển đủ 24 khớp). Cả 5 object đã mesh-hóa →
  `get_phase1_variant_cfg()` (VariantEntityCfg) dựng được. **Còn lại:** wire
  variant vào scene (đặt object theo lowest_point từng world) → làm ở §C.
- [ ] `ContactSensor`: palm + 3 link cuối mỗi ngón + 6 pad (match theo body
  regex), lấy `found`/`force` per-link so với object; sensor riêng cho
  ngón↔bàn và arm↔{bàn, object}. (→ §E cùng observations.)
- [x] Sim cfg: `timestep=0.01`, `decimation=20`, `impratio=10`,
  `cone="elliptic"`, `nconmax=150`/`njmax=1500`; episode 14 s (70 bước).
- [x] Đăng ký task `Mjlab-DexGrasp-UR5eRH5DG2` qua `register_mjlab_task`.

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
