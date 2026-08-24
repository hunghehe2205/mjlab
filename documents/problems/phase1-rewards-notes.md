# Phase 1 §F — reward port notes & decisions

Ghi chú khi cài reward stack (mdp/rewards.py). Coeff giữ nguyên cfg_reg.yaml;
`REWARD_COEFFS` map 1:1.

## 1. Reward tính per control step — bật `scale_rewards_by_dt=False`

Reference tính reward mỗi 0.2s control step và không scale theo dt. mjlab mặc
định scale reward theo step dt (×0.2) → sẽ nhân toàn bộ return đi 0.2. Đã tắt
`scale_rewards_by_dt` để giữ magnitude khớp reference.

## 2. Clip tổng min −2: opt-in `reward_clip_min`

Reference `reward_r.clip(min=-2.0)` sau khi cộng C++ + python rewards. mjlab
chưa có hook → thêm field `reward_clip_min: float | None = None` vào
`ManagerBasedRlEnvCfg`, clamp trong `step()` sau `reward_manager.compute()`
(default None = không đổi hành vi task khác). DexGrasp set −2.0.

## 3. Sensor bổ sung cho reward (scene sensors)

- `hand_table_contact`: 16 body hand ↔ table geom (`arena/table`) — table
  contact/impulse.
- `arm_world_contact`: 6 body arm, secondary None — arm_collision (any contact,
  dùng `found` không ngưỡng, khớp `contacts_arm_all`).
- `arm_table_contact`: 6 body arm ↔ table — arm contact/impulse.

**Deviation:** reference gộp arm contact với {table OR object} vào
`contacts_arm_table`. Sensor chỉ secondary 1 element → arm_contact/impulse
(−0.1/−0.1) chỉ bắt table; arm chạm object vẫn bị `arm_collision` (−1.0) bắt.
Đánh giá chấp nhận được (arm hiếm chạm object, penalty chính là collision).

## 4. Impulse clip dùng index thumb tường minh

Reference set `impulse_high` = 0.2 cho 3 contact cuối (giả định thumb ở cuối
danh sách Allegro). RH5-DG2 đặt thumb đầu (`CONTACT_THUMB_INDICES = (1,2,3)`)
→ `get_contact_clip_high()` set 0.2 theo index thumb tường minh, còn lại 0.1.
Giữ nguyên ý đồ (thumb chịu impulse nhiều hơn) bất kể thứ tự.

## 5. Trọng số ngón

- `affordance_weights` (24 keypoint): tips ×4, thumb tip ×8, wrist 0, normalize
  rồi ×16 — dùng cho affordance distance + table log-barrier.
- `contact_weights` (16 body): tips ×3, thumb ×2, thumb tip ×2, palm 0,
  normalize rồi ×16 — dùng cho contact/impulse rewards.

## 6. Không có `action_rate_l2`

Reference không có action-rate penalty (delta action std nhỏ tự đủ mượt). Đã
bỏ term skeleton này.

## 7. Zero-action rollout probe

70 bước zero action: reset contact-free (affordance_contact = 0), arm_collision
= 0 cả episode, reward finite; ngón/object lắng xuống → contact/impulse xuất
hiện từ ~step 30+ (fingers chạm object/table), object_displacement nhỏ (~0.1 m),
một số env chạm clip −2. Đây là baseline để §H đánh giá khi train.