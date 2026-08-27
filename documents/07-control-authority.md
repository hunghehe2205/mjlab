# 07 — Khôi phục quyền điều khiển (sau báo cáo so sánh)

Áp mục 1–6 của thứ tự gộp trong đối thoại. Mọi số đều đo trên máy local,
`potted_meat_can` trừ chỗ ghi rõ khác.

## 0. Tóm tắt

| # | Vấn đề | Trước | Sau |
|---|---|---|---|
| 1 | Ratchet trọng lực (arm) | trôi 0.31 m/ep | **0.0000 m** |
| 1 | Ratchet trọng lực (ngón) | trôi 0.97 rad/ep | **0.0000 rad** |
| 1 | Jitter vật lúc đứng yên | 1.225 rad/s | **0.0074 rad/s** |
| 2 | Arm bám lệnh | 0.638 | **0.974** |
| 2 | Lực kẹp ngón | 0.00 N | **8.9–13.3 N** (cần 3.1) |
| 2 | Bước tới chạm đầu tiên | ~50 | **~30** |
| 3 | Reward thô/bước | −8.5 | **−1.889** |
| 5 | friction hand–bàn | 0.2 | **0.8** (khớp gốc) |
| 6 | mass 5 object | 0.24875 đồng loạt | **theo URDF gốc** |

Chi phí: dt 10→5 ms nhưng `cone="pyramidal"` bù lại — tổng ~0.41× thời gian
bước so với cấu hình cũ, tức **nhanh hơn**.

## 1. Ratchet trọng lực — `gravcomp`

`RelativeJointPositionAction` neo lại target trên qpos thật mỗi bước
(`target = qpos + delta`). PD hữu hạn nên mỗi bước joint nằm dưới target đúng
τ/kp, và độ lún đó bị neo lại thành điểm xuất phát mới → cộng dồn tuyến tính.

elbow: 17.4 N·m / kp 2000 = **0.0087 rad/bước**, trong khi trần action chỉ
**0.005**. Thâm hụt ròng — policy không thể giữ nguyên vị trí, chứ chưa nói tới
điều khiển.

Nâng kp không phải lời giải (đã đo ở dt=10 ms):

| kp/kd | trôi |
|---|---|
| 2000/500 | 0.41 rad |
| 8000/400 | 3.01 rad — mất ổn định |
| 16000/500 | 3.02 rad — mất ổn định |
| 16000/800 | 2.96 rad — mất ổn định |

Lời giải là `gravcomp=1` trên **mọi** body robot — mô hình hoá đúng bộ điều
khiển UR bù tải trọng. Chỉ bù riêng arm còn để lại 0.093 m (tay nặng 0.574 kg).

## 2. Gains — luật ω_n đồng nhất, ζ≈1

### 2a. Arm

Gains menagerie cho **τ = kd/kp = đúng 0.2000 s** trên cả 6 khớp — bằng y hệt
chu kỳ điều khiển 5 Hz. Lý thuyết: một lệnh bão hoà chỉ thực hiện được
1−e^(−1) = **0.6321** phần delta. Đo được **0.638 / 0.637**.

Đặt lại theo `kp = M·ω_n²`, `kd = 2ζ·M·ω_n`. Điều kiện ổn định
`kp·dt²/M = (ω_n·dt)²` — **không phụ thuộc quán tính khớp**, nên một ω_n duy
nhất là an toàn cho cả shoulder lẫn wrist.

Với ω_n = 64, ζ = 1.15:

| khớp | kp cũ | kd cũ | kp mới | kd mới |
|---|---|---|---|---|
| shoulder_pan | 2000 | 400 | **16131** | **580** |
| shoulder_lift | 2000 | 400 | 15044 | 541 |
| elbow | 2000 | 400 | 2992 | 108 |
| wrist_1..3 | 500 | 100 | 414–527 | 15–19 |

System-ID trên UR5 thật của bản gốc: **kp 15775–16202, kd 281.7–577.3**, tức
ω_n = 63.7, ζ = 1.15. Luật này tái tạo lại đúng bộ điều khiển đã nhận dạng —
đây là tăng độ trung thực sim2real, không phải tinh chỉnh cho sim.

τ mới 0.0359 s (gốc 0.036). Bám lệnh **0.638 → 0.974**.

### 2b. Ngón

Lực kẹp bão hoà ở `kp × ACTION_SCALE_FINGER`, nên kp=1 chặn ở **0.015 N·m** —
`effort_limit=1.0` không bao giờ chạm tới. Allegro gốc: 600 × 0.015 = 9 N·m →
clip **0.7 N·m**. Yếu **47×**.

Không nâng kp thẳng được: `implicitfast` xử lý stiffness **tường minh**.
Quét (armature, kp) tại dt=5 ms, cột là `peak_qd` dưới action ngẫu nhiên:

| armature | kp=1 | kp=67 | kp=200 | kp=600 |
|---|---|---|---|---|
| 1e-4 | 0.01 | 48.9 ⚠ | **69.0 nổ** | **66.4 nổ** |
| 1e-3 | 0.05 | 0.05 | 5.24 | 6.47 |
| 1e-2 | 0.15 | 0.01 | 0.01 | 0.17 |
| **3e-2** | 0.09 | 0.01 | 0.01 | **0.01 ✓** |

Armature mới là đòn bẩy. Chốt **armature 3e-2, kp 600, kd 8.49** (ζ=1).
Lực kẹp đo được **8.9–13.3 N**, ngưỡng giữ 0.249 kg với μ=0.8 là 3.05 N.

Armature 3e-2 tương ứng quán tính rotor quy đổi qua hộp số tỉ số ~300 — hợp lý
cho ngón dẫn động bánh răng, và cùng bản chất với damping identified 0→9.8 của
Allegro. **Đây là tham số nên đo trên phần cứng thật** (xem §6).

## 3. Ngân sách tiếp cận

`APPROACH_DISTANCE = 0.25` đặt grasp_center cách tâm vật 25 cm theo hướng về
camera — mà camera `[0.035, −0.58, 1.531]` nên hướng đó gần thẳng đứng. Bản gốc
dùng **cùng công thức** (`pos = aff_center + 0.25 * hand_dir_x_w`) và **cùng
action scale** (`rot_action_std: 0.005`, `finger_action_std: 0.015`). Port trung
thành ở điểm này — 27 cm không phải bug.

Chạy controller scripted hết tốc độ cho phép, số env có contact:

| bước | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|
| trước fix | 0 | 0 | 0 | 0 | 1 | 4 | 6 |
| sau fix | 0 | 0 | 1 | 4 | 6 | 6 | 7 |

Trước: chạm đầu tiên ở bước 50, hết 70 bước vẫn chưa đủ → **không còn bước nào
để kẹp**. Sau: chạm ở bước 30, 6/8 env ở bước 50 → còn 30–40 bước cho kẹp+nâng.

## 4. Reward clip

Sau §1–2, penalty jitter sụp đổ:

| term | reward/bước (action=0) |
|---|---|
| **affordance_distance** | **−1.8720** |
| object_displacement | −0.0138 |
| object_velocity | −0.0029 |
| object_angular_velocity | −0.0006 |
| arm_joint_velocity | −0.0000 |
| wrist_velocity | −0.0000 |

Tổng thô **−8.5 → −1.889**. Tín hiệu còn lại gần như hoàn toàn là "tiến lại gần
vật" — đúng cái cần.

Nhưng sàn −2.0 chỉ còn **0.13** dư địa, và 22.9% số bước rơi dưới ngưỡng dưới
policy ngẫu nhiên → gradient bị làm phẳng. Bản gốc thực ra **không clip**:
`reward_r.clip(min=reward_clip)` ở train.py không gán kết quả, và dòng ngay sau
ghi đè terminal −10 bằng tổng thô. Đặt `reward_clip_min=None`,
`termination_reward=0.0` vừa khớp gốc vừa cần thiết.

## 5. Termination — tự hết

Qua 70 bước × 16 env, cả action=0 lẫn ngẫu nhiên:
`hand_below_table=0`, `object_out_of_workspace=0`. Trước đây chúng nổ vì arm
trôi làm văng vật; §1 xoá nguyên nhân. **Giữ nguyên làm lưới an toàn** — không
cần bỏ như báo cáo đề xuất.

## 6. Friction và mass

Cơ chế gốc (`hardware.hpp:428`, `Environment.hpp:44–47`):

```cpp
setDefaultMaterial(0.8, ...)                            // moi cap mac dinh 0.8
setMaterialPairProp("table", "object", table_friction_) // CHI object-table = 0.2
```

Port đặt `priority=1` trên bàn → ép 0.2 lên **mọi** thứ chạm bàn, kể cả tay.
MuJoCo gộp friction bằng **max** khi priority bằng nhau, nên bỏ priority + đặt
table 0.2 / object 0.2 / robot 0.8 tái tạo chính xác cả ba cặp.

Mass: đúng 5 object trong 35 có mass riêng trong URDF gốc. scissors và
small_block **nhẹ gấp 3×** so với giá trị đồng loạt, và cả hai đều được
oversample khi train.

## 7. Còn lại

- **Chưa chứng minh policy học được.** Controller scripted của tôi kẹp được
  8.9–13.3 N nhưng không nâng nổi: nó ép vật xuống bàn (`obj dz` đứng yên
  −0.0023 m suốt) rồi mất contact ngay bước đầu nâng. Đó là hạn chế của script
  tiếp cận thẳng từ trên xuống, không phải bằng chứng nhiệm vụ bất khả thi —
  kẹp khéo chính là việc RL phải học. Bước tiếp theo là train, không phải
  script thêm.
- **`INIT_FINGER_POSE` vẫn là ước lượng thô** (comment trong code: *"first
  estimate, refine in viewer"*).
- **`FINGER_ARMATURE = 3e-2` nên đo trên phần cứng thật.** Giá trị hiện tại chọn
  theo ràng buộc ổn định + tương tự Allegro, không theo số liệu RH5-DG2.
- 4 câu hỏi mở ở [05](05-audit-fixes.md) §8 vẫn chưa có đáp án.
