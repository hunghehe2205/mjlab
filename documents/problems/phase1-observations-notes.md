# Phase 1 §E — observation notes & traps

Ghi chú khi cài teacher observations. Không phải bug chặn train, nhưng phải
biết khi tune §F/§H.

## 1. Tên body trong compiled model giữ prefix `rh/`

`Entity` strip prefix attach (`rh/`) cho entity-local names (nên
`robot.body_names` thấy `R_hand_palm`), nhưng compiled MJCF giữ nguyên:
`robot/rh/R_hand_palm`, `robot/rh/right_hand`. ContactSensor dùng literal
names phải build `robot/{rc.HAND_PREFIX}{local}` (xem
`get_hand_object_contact_sensor`). Body/site của arm không có `rh/`.

## 2. `quat_apply` / `quat_inv` không broadcast

`quat_apply` reshape cả hai về `(-1, ...)` nên không broadcast được
`(B,1,4)` lên `(B,K,3)` (lỗi shape 4 vs 96). Phải expand quat ra `(B,K,4)`
trước: `compute_af_vec` làm vậy.

## 3. Contact impulse = netforce history sum x dt

Sensor `reduce="netforce"` + `history_length=DECIMATION` cho một impulse
vector per body per control step: `Σ substeps force × timestep`. Ngưỡng skip
0.001 / flag 0.01 N·s giữ đúng bản gốc RaiSim (impulse = force×dt). netforce
cộng dồn trước nên không áp skip per-contact được; thay vào đó floor magnitude
dưới `SKIP_IMPULSE` về 0 (xấp xỉ gần đúng). MuJoCo soft contact + effort-limit
ngón khác RaiSim rigid → scale impulse sẽ khác; calibrate lại ngưỡng ở §H khi
đo được grasp contact thật (xem rủi ro #3 trong 01-analysis.md).

## 4. Thứ tự frame/body cố định

- `CONTACT_BODIES` (16): palm, rồi thumb→pinky × (mcp, pip, dip). Index 0 =
  palm (weight 0 ở §F, như wrist_3_link bản gốc). Pad là fixed-child của
  dip/palm nên subtree-sensor gộp contact pad vào link cha.
- `KEYPOINT_BODIES` (24): `right_hand` (wrist) → mỗi ngón (yaw/mcp/pip/dip +
  force_sensor pad), ring/pinky không có yaw.
- `ARM_LINK_BODIES` (6): shoulder_link → wrist_3_link (frame 6 khớp arm).
- Các SceneEntityCfg trong env_cfgs dùng `preserve_order=True` để khoá đúng
  thứ tự tuple trên các obs column (af_vec/heights/contact), tránh §F nhân
  trọng số sai ngón khi đổi thứ tự constants hay model layout.

## 5. Euler convention = RaiSim patched `RotmatToEuler`

Fixed-frame XYZ với dấu và nhánh gimbal của bản patched (xem
`rotations.py::euler_from_rotmat`). Unwrap theo state machine gốc: euler_prev
khởi tạo 0, chỉ unwrap ±2π khi `prev.norm() > 0.01` → frame đầu sau reset giữ
raw. `euler_diff = euler(R_initᵀ R)` không unwrap.

## 6. Stateful term chạy trước `sim.forward()`

`ObservationManager.reset()` chạy trong `_reset_idx` trước khi `sim.forward()`
của state mới → `WristOrientation` capture `wrist_rot_init` lazy ở lần compute
đầu sau reset (pending flag), không đọc FK trong `reset()`.

## 7. PD error cần target đã clip

Thêm property `target` cho `RelativeJointPositionAction` (anchor 1 lần/control
step ở `process_actions`). `pd_error = target − qpos` phản ánh đúng
`pTarget_clipped − gc` của bản gốc (obs đọc sau khi step). `reset()` anchor
`target = qpos` tại reset → pd_error = 0 ở obs reset (khớp C++
`pTarget_clipped = gc_set`), không bị stale target episode cũ.

## Layout 191 dim

| Term | Dim |
|---|---|
| joint_pos | 24 |
| pd_error | 24 |
| contacts (16 flags + 16 impulse) | 32 |
| keypoint_heights | 24 |
| arm_link_heights | 6 |
| hand_center (grasp_center site) | 3 |
| wrist_orientation (euler + euler_diff) | 6 |
| af_vec | 72 |
