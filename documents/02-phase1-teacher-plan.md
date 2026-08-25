# Phase 1 — Teacher policy (privileged PPO)

Mục tiêu: teacher grasp được cohort 35 object RobustDexGrasp trên UR5e +
RH5-DG2 trong mjlab, eval lift success > ~80% trước khi sang Phase 2.

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
- [x] Nhập cohort 35 object `new_training_set` của RobustDexGrasp. Dùng mesh
  collision `top_watertight_tiny.obj`, tạo 200 affordance points + lowest-point
  metadata; giữ box/cylinder là object debug ngoài cohort.
- [x] Script `precompute.py`: 200 surface points + normals + centroid + lowest
  → `.npz` cạnh mesh.
- [x] Mesh → MJCF `spec_fn`: collision = **convex hull** (không coacd, xem
  problems/coacd), freejoint, material `object`; wrap `EntityCfg` + registry
  `PHASE1_OBJECTS`. (`VariantEntityCfg` để §B khi dựng scene multi-object.)
- [x] Test: mở rộng `test_ur5e_rh5dg2_constants.py` + mới `test_dexgrasp_objects.py`.

## B. Scene & task skeleton

- [x] `SceneCfg`: bàn box tĩnh (top 0.771 m, friction 0.2) + **bục riêng** cho
  UR5e (base flush mặt bàn z=0.771, khe ~8cm) + object. Fixed-base entity
  auto-mocap → reset về env_origins. Arm menagerie actuator adopt qua
  `XmlActuatorCfg` (action điều khiển đủ 24 khớp). Cả 35 object đã mesh-hóa →
  `VariantEntityCfg` gán một mesh theo mỗi world. Cohort mặc định dùng weight
  gốc (44 slots: scissors/off_water_body ×3, 7 object khó ×2, còn lại ×1),
  với 2 lần lặp thành 88 environment train mặc định.
- [x] `ContactSensor` hand↔object: 16 body (palm + 3 link cuối mỗi ngón; pad là
  fixed-child của dip/palm nên subtree-sensor gộp luôn), `reduce="netforce"` +
  `history_length=DECIMATION` → impulse = Σ substeps force × dt. (→ §E; sensor
  ngón↔bàn và arm↔{bàn, object} để §F khi cần reward table/arm.)
- [x] Sim cfg: `timestep=0.01`, `decimation=20`, `impratio=10`,
  `cone="elliptic"`, `nconmax=150`/`njmax=1500`; episode 14 s (70 bước).
- [x] Đăng ký task `Mjlab-DexGrasp-UR5eRH5DG2` qua `register_mjlab_task`.

## C. Reset & pre-grasp (event terms, CPU per-reset như bản gốc)

- [x] Event sample pose object (`pregrasp/pose_sampler.py::sample_object_pose`):
  phân cực r ∈ [0.45, 0.75], góc ∈ [−0.7π, −0.3π], |x| < 0.25, xoay z, z = mặt
  bàn − lowest_point. Phase 1 uniform; nhánh Beta wired (`non_uniform`) cho Phase 2.
- [x] Visible points (`pregrasp/visibility.py`): raycast trimesh từ camera cố
  định (env frame, `CAMERA_POSITION` port từ gốc — verify viewer, xem problems/
  pregrasp-ur5e-tuning) → visible surface + tâm. Fallback ray-miss = điểm pcd.
- [x] `sample_rot_mats` (`pregrasp/pose_sampler.py`): N palm-roll quanh trục tiếp
  cận + projection width; test rot-mat orthonormal, det=+1.
- [x] **IK giải tích UR5e** (`pregrasp/ik_ur5e.py`): analytic 8-nghiệm UR, DH
  fit theo MJCF UR5e (không phải textbook) + offset base/flange đo từ MuJoCo FK
  (`_B=Rz(π)`, `_E≈I`); `solve_arm_qpos(T_base_flange, seed)`. Test round-trip
  FK(IK)≈pose vs MuJoCo: **≤0.5mm** trên 500 pose, 0 miss. `_ARG_TOL=1e-2` để
  không loại nhầm pose reachable do residual DH.
- [x] Scoring + composition (`pregrasp/generator.py::generate_pregrasp`):
  visible → approach → grasp-center target (0.25m) → sample_rot_mats → IK mỗi
  candidate qua `ArmKinematics` (`pregrasp/kinematics.py`, đo frame flange↔gc từ
  model) → chọn best theo projection width < 0.18 (coeff 5) + wrist angle. Test
  integration trên object thật + grasp-center round-trip sub-mm.
- [x] Reset event (`mdp/events.py::ResetGraspPose`) + wire vào robot cfg: ghi
  object root (sampled pose + env_origin) + arm qpos, fingers giữ default
  (`INIT_FINGER_POSE`). Test: grasp-center in-sim khớp IK prediction, object trên
  bàn, no NaN; đọc mesh/point cloud/lowest-point theo variant từng world.
  Probe tĩnh arm↔hand reject candidate va chạm và resample tối đa 8 lần;
  fallback chỉ dùng khi collision-free, nếu không về HOME pose an toàn.
- [x] `sim.forward()` sau ghi pose: env tự forward sau reset events (`_reset_idx`
  → `write_data_to_sim` → `sim.forward()`), reset event không cần tự gọi.

## D. Actions

- [x] `RelativeJointPositionActionCfg`, scale arm 0.005 / finger 0.015 (đã có ở
  §B), + **soft-limit target clip**: thêm cờ opt-in `clip_to_joint_limits` cho
  `RelativeJointPositionAction` (clamp *absolute target* về `soft_joint_pos_limits`;
  delta vẫn không clip). Arm tự bound qua actuator ctrlrange; cờ này cần cho ngón.
  Bật trong `make_dexgrasp_env_cfg`. Test `test_actions.py`: clamp khi bật, giữ
  nguyên khi tắt (default off, không đổi hành vi task khác).
- [ ] (Tùy chọn, bám gốc) random 1-step action delay — bỏ ở teacher phase 1,
  thêm ở student (§ Phase 3).

## E. Observations (group `actor` = `critic`, đều privileged)

- [x] Proprio: qpos (24) + sai số PD `target − qpos` (24).
- [x] Contact flags (16) + impulse magnitude (16) từ ContactSensor
  (ngưỡng 0.001/0.01 N·s giữ baseline RaiSim — calibrate lại ở §H).
- [x] Chiều cao 24 keypoint + 6 arm link so mặt bàn.
- [x] `grasp_center` pos (3), euler wrist (3), euler diff so init (3, unwrap).
- [x] **af_vec (72)**: term GPU thuần torch — transform 200-point cloud theo
  object pose hiện tại, `torch.cdist` với 24 keypoint (wrist + 18 khớp + 5
  tip), lấy vector tới điểm gần nhất, xoay về trục world.
- [x] Unit test af_vec trên case nhỏ tính tay (+ euler/unwrap, + env test bắn
  contact palm khi ép object vào lòng bàn tay).

Tổng **191 dim** (24+24+32+24+6+3+6+72). Chi tiết đặt tên frame/body, euler
convention, và trap khi cài xem `problems/phase1-observations-notes.md`.

## F. Rewards (`mdp/rewards.py`, giữ coeff gốc làm baseline)

- [x] affordance distance: −Σ wᵢ·min_dist (tip ×4, thumb tip ×8, palm 0) — coeff 0.5.
- [x] affordance contact (weighted, tip ×3, thumb ×2) — coeff 1.5.
- [x] affordance impulse xy (clip 0.1 link thường / 0.2 thumb) — coeff 1.0.
- [x] table contact/impulse (−1.0 / −0.5) + table log-barrier (−0.03).
- [x] arm: height log-barrier (−0.05), contact/impulse (−0.1), collision (−1.0).
- [x] object: vel (−15), qvel (−0.2), displacement (−5).
- [x] wrist vel/qvel (−1.0/−0.1, ×10 khi >0.25), arm joint vel (−1.0, ×4 khi >0.5).
- [x] Reward tổng clip min −2 (hook trong env, opt-in `reward_clip_min`) + `scale_rewards_by_dt=False`
  (reference rewards tính per control step, không scale theo dt).

Đã bỏ `action_rate_l2` skeleton (reference không có). Ghi chú: arm_contact/impulse
chỉ filter theo table (arm chạm object bị arm_collision bắt); clip impulse dùng
index thumb tường minh thay vì "3 contact cuối" (xem
problems/phase1-rewards-notes.md).

## G. Termination & eval

- [x] `time_out` 70 bước; terminate khi keypoint tay dưới mặt bàn; NaN guard.
- [x] Metric success: script eval riêng — sau 70 bước grasp, arm target nội suy
  về pose nâng trong 80–100 bước (5 Hz), success = object z tăng > 0.1 m;
  log per-object success rate cho toàn bộ cohort 35 object.

## H. RL config & train

- [x] `RslRlOnPolicyRunnerCfg`: MLP 128×128, γ=0.996, λ=0.95, 4 epochs, 4
  minibatches, obs normalization, min action std 0.2.
- [ ] Smoke train (~vài trăm iter, ít env): reward tăng, không NaN, contact
  xuất hiện.
- [ ] Train đầy đủ + đánh giá: tune PD gains ngón / ngưỡng contact nếu grasp
  bất ổn.

## Thứ tự làm việc gợi ý

A (assets) → B (skeleton chạy được với action zero) → C (reset/pre-grasp,
verify bằng viewer thấy tay đặt đúng pre-grasp quanh object) → D+E → F+G →
H. Mỗi mốc verify bằng viewer/`uv run play` trước khi sang bước sau.
