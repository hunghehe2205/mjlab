# Audit port RobustDexGrasp → mjlab (UR5e + RH5-DG2) và kế hoạch sửa

> **Báo cáo lịch sử.** Nhiều lỗi và đề xuất trong tài liệu này đã được sửa,
> đo lại hoặc bác bỏ ở [06](06-experiment-report.md),
> [07](07-control-authority.md) và [08](08-log-audit-fixes.md). Không dùng
> bảng thông số tại đây thay cho
> [cấu hình teacher hiện tại](02-phase1-teacher-plan.md).

Đối chiếu `mjlab-DexGrasp` @ `325ec45` với `RobustDexGrasp` gốc (nhánh teacher),
kèm số liệu đo trực tiếp trên env và trên run training đang chạy.

- Bản gốc: `/Users/hunghehe2205/Projects/RobustDexGrasp`
- Run tham chiếu: wandb `mjlab-dexgrasp/advid9la`, iteration 2272/10000
- Mọi số liệu trong tài liệu này là **đo được**, không suy luận.

---

## 0. Tình trạng hiện tại

Phần "phương pháp" port rất sát bản gốc: 16 reward term đúng từng hệ số, PPO khớp
hoàn toàn, observation đủ 8 khối đúng ngữ nghĩa, pre-grasp pipeline đúng cấu trúc,
object cohort đúng 35 vật với oversampling y hệt. 255 test dexgrasp/ur5e pass.

Nhưng **training đã đứng từ ~iteration 150** và policy hội tụ về hành vi thoái hoá:
tránh chạm vật.

| | iter 0 | iter 2272 |
|---|---|---|
| khoảng cách tay↔object (trung bình có trọng số) | 0.133 m | **0.278 m** — lùi ra xa |
| `affordance_contact` / bước (max 1.5) | 0.0147 (~1%) | **0.0016 (~0.1%)** |
| object \|ω\| trung bình | 4.35 rad/s | 4.86 rad/s |
| `object_out_of_workspace` | 2.10 | 2.60 (~38% số episode) |
| learning rate | 2.5e-3 | **1e-5** (sàn, từ ~iter 700) |
| policy std | 1.00 | 0.41 |
| `mean_reward` | −87.5 | −96.2 |

**Reward thô ≈ −9.1/bước, bị clip xuống −2.0 ở ~90% số bước.** Tín hiệu học gần như
là hằng số → gradient ≈ 0 → adaptive-KL đẩy lr xuống sàn → đóng băng.

Phân rã reward thô (quy về mỗi bước, từ `Episode_Reward × 14 / mean_episode_length`):

```
object_angular_velocity  -4.711   <-- 52% tổng phạt, không do policy gây ra
affordance_distance      -2.226
object_displacement      -1.043
object_velocity          -0.934
arm_joint_velocity       -0.159
còn lại                  -0.055
affordance_contact       +0.003
affordance_impulse       +0.004
                        --------
tổng                     -9.13  →  clip  -2.00
```

---

## 1. Bảng ưu tiên

| # | Vấn đề | Mức | Ảnh hưởng | Công sức |
|---|---|---|---|---|
| A1 | Impulse lệch thang ~20× | Bug | Reward + obs | 1 dòng |
| A2 | Sensor bỏ sót pad đầu ngón | Bug | Reward + obs | XML/spec + test |
| A3 | Metric đọc kinematics cũ lúc reset | Bug | Chỉ giám sát | ~10 dòng |
| B1 | Object rung/trượt trên bàn khi robot đứng im | Physics | **Chặn training** | Cần quét |
| C1 | Áp `reward_clip_min` / `termination_reward` mà bản gốc không áp | Khác biệt | Reward | 2 dòng |
| C2 | Friction lệch bản gốc | Khác biệt | sim2real | vài dòng |
| C3 | Soft joint limit 0.9 vs hard limit | Khác biệt | sim2real | 1 dòng |
| C4 | Thứ tự normalize trọng số affordance | Khác biệt | ~3% | 2 dòng |
| D1 | **PD gain ngón tay là số bịa** | sim2real | **Rủi ro lớn nhất** | Cần đo/nhận dạng |
| D2 | Arm actuator không nhận dạng ở 5 Hz | sim2real | Cao | Cần đo |
| E | Test/script/pyproject gãy sau commit `refactor` | Vệ sinh | CI | Dọn |

---

## 2. Nhóm A — Bug thật

### A1. Impulse lệch thang ~20×

**Nguyên nhân.** Bản gốc chạy 20 substep rồi mới gọi `updateObservation()` **một lần**
và đọc `getContacts()` — RaiSim chỉ giữ contact của lần `integrate()` cuối. Giá trị nó
dùng là impulse của **một** substep 10 ms: `F × 0.01`.

Port (`mdp/rewards.py:102-106`, `mdp/observations.py:115`) cộng cả 20 substep:
`history.sum(dim=2) * dt = 20 × F × 0.01 = 0.2F`.

**Hệ quả.** Các ngưỡng được chép nguyên xi từ bản gốc nhưng giờ mang ý nghĩa khác 20 lần:

| | ý nghĩa ở bản gốc | ý nghĩa ở port hiện tại |
|---|---|---|
| flag `> 0.01` | lực > 1 N | lực > **0.05 N** |
| clip `0.1 / 0.2` | bão hoà ở 10 / 20 N | bão hoà ở **0.5 / 1 N** |

Đo thực tế: impulse max **1.6657**, gấp 16 lần mức clip 0.1. `affordance_impulse` mất
hoàn toàn tính "có mức độ", trở thành bản sao có tỉ lệ của `affordance_contact`.

**Sửa.** `mdp/rewards.py`:

```python
def sensor_impulse(sensor: ContactSensor, dt: float) -> torch.Tensor:
  """Mean per-substep impulse vectors (B, P, 3) over the control step."""
  history = sensor.data.force_history
  assert history is not None
  return history.mean(dim=2) * dt
```

và `HandObjectContacts.__call__` (`mdp/observations.py:113-115`) gọi lại
`sensor_impulse(self._sensor, self._dt)` thay vì lặp lại công thức.

**Vì sao `mean` đúng chứ không phải chia 20 cho có.** `mean × dt` = "lực trung bình
trong control step × 10 ms" — đúng thang đo bản gốc, nên mọi ngưỡng 0.01 / 0.1 / 0.2
giữ nguyên hiệu lực. Bản gốc lấy 1 mẫu (substep cuối), port lấy trung bình 20 mẫu →
**ít nhiễu hơn**, không phải sai lệch.

**Với sim2real: đây là cải thiện.** `sum` = "lực trung bình × 200 ms" — đại lượng gắn
chặt vào `decimation`, không có tương ứng phần cứng, và đổi decimation là cả khối obs
lẫn reward âm thầm đổi thang. `mean × dt` tái tạo được trên phần cứng: lấy trung bình
số đọc cảm biến lực trong cửa sổ 200 ms rồi nhân 0.01.

> Lưu ý: 16 chiều obs đổi thang → checkpoint cũ vô hiệu.

---

### A2. Contact sensor bỏ sót pad đầu ngón

**Cấu trúc thật.** Pad (`R_*_force_sensor`) là body riêng, **hàn cứng** vào link cha, và
nó giữ mesh tiếp xúc chính. Link `dip` chỉ có 5 cylinder nhỏ chạy dọc hông ngón.

```
R_thumb/index/middle/ring/pinky_force_sensor  →  cha là *_dip
R_palm_force_sensor                           →  cha là R_hand_palm
```

MuJoCo: `body1=` chỉ khớp geom **thuộc chính body đó**; `subtree1=` khớp cả con cháu.
Kiểm chứng tối giản:

```
quả cầu con chạm sàn (body cha ở cao hơn 20 cm):
  body_mode = [0.]    subtree_mode = [1.]    child_mode = [1.]
```

`config/ur5e_rh5dg2/env_cfgs.py:115` và `:128` dùng `mode="body"` trên `CONTACT_BODIES`
→ **pad vô hình với sensor**. Đo trên env (70 bước × 8 env, ngón siết vào potted_meat_can):

```
body-mode    : 189 lần contact-body kích hoạt
subtree-mode : 429   →  đang mất 56% số contact tay↔object
```

Ảnh hưởng: `affordance_contact` (+1.5), `affordance_impulse` (+1.0), `table_contact`
(−1.0), `table_impulse` (−0.5), và 32/191 chiều observation.

Docstring ở `ur5e_rh5dg2_constants.py:159-161` đã ghi đúng ý định là subtree.

#### Vì sao **không** được flip sang `mode="subtree"`

`CONTACT_BODIES` là chuỗi lồng nhau `palm ⊃ mcp ⊃ pip ⊃ dip ⊃ pad`. Subtree làm một cú
chạm đầu ngón sáng 4 slot. Ba hệ quả:

1. Bản gốc map mỗi contact về **đúng 1** body (`contactMapping_r_[localBodyIndex]`).
2. Trọng số khác nhau theo slot (tip ×3, thumb ×2) — đếm trùng làm méo tỉ lệ giữa các ngón.
3. `divisor=16` giả định tối đa 16 flag độc lập; đếm trùng đẩy reward vượt `[0, 1]`.

Subtree chỉ chính xác cho đúng 5 link `dip`, mà một `ContactMatch` không trộn được hai mode.

#### Cách sửa: đưa collision geom của pad về body cha

Sau khi chuyển, "pad chạm" **chính là** "dip body chạm" → `mode="body"` trở nên chính
xác, 1 slot, không trùng. Body pad vẫn giữ nguyên (nằm trong `KEYPOINT_BODIES` cho reward
khoảng cách, và giữ geom visual).

**Đã chứng minh không đổi động lực học.** Compile hai mô hình rồi so từng mảng:

```
geom_type/contype/conaffinity/condim/group/priority/solmix
solref/solimp/size/friction/margin/gap/matid ........ max|d| = 0.0
body_mass / body_inertia / body_ipos / body_iquat ... max|d| = 0.0
dof_M0 / body_weldid ............................... max|d| = 0.0
xpos (body frames) ................................. max|d| = 0.0
geom_xpos / geom_xmat (tư thế ngẫu nhiên) .......... max|d| ~ 1e-16
tập cặp contact @ margin thật (0 và 0.002) ......... GIỐNG HỆT, 6/6 tư thế
```

Chỉ `geom_bodyid` đổi — đúng mục đích. Lý do sâu hơn:

- Pad không có joint → đã cùng `weldid` với link cha, về động lực học vốn là một khối cứng.
- Cả hai body có `<inertial>` tường minh → MuJoCo dùng con số đó, không suy từ geom.
- MuJoCo vốn loại contact giữa geom cùng weld group → không sinh self-collision mới.
- Geom collision của pad **không có `pos`/`quat` riêng** → đưa lên cha chỉ cần gán
  `pos`/`quat` = `pos`/`quat` của body pad, vị trí trong không gian giống hệt.

**Cảnh báo trung thực:** thứ tự index geom đổi → thứ tự contact trong solver đổi → rollout
dài có tự-va-chạm ngón-với-ngón sẽ phân kỳ hỗn loạn sau ~20 bước. Không phải thay đổi mô
hình, nhưng **không so được run trước/sau theo từng seed**.

#### Hai phương án

**(a) Sửa thẳng `right_hand.xml`** — dời 6 dòng geom vào body cha, thêm `pos`/`quat`
bằng `pos`/`quat` của body pad. Đơn giản, khai báo rõ.

**(b) Re-parent trong `get_spec()` bằng MjSpec** — xoá geom rồi add lại vào body cha:

```python
PAD_BODIES = (
  "R_thumb_force_sensor",
  "R_index_force_sensor",
  "R_middle_force_sensor",
  "R_ring_force_sensor",
  "R_pinky_force_sensor",
  "R_palm_force_sensor",
)
_GEOM_ATTRS = (
  "type",
  "meshname",
  "classname",
  "group",
  "rgba",
  "contype",
  "conaffinity",
  "condim",
  "friction",
  "solref",
  "solimp",
  "margin",
  "gap",
  "priority",
  "density",
  "mass",
)


def _reparent_pad_collisions(hand: mujoco.MjSpec) -> None:
  """Move each pad's collision geom onto its parent link.

  Pads are welded children, so this is dynamically a no-op; it makes the
  body-mode contact sensors on CONTACT_BODIES see pad contacts.
  """
  for pad in PAD_BODIES:
    body = hand.body(pad)
    geom = next(g for g in body.geoms if g.name.endswith("_collision"))
    parent, name = body.parent, geom.name
    attrs = {f: getattr(geom, f) for f in _GEOM_ATTRS}
    pos, quat = body.pos.copy(), body.quat.copy()
    hand.delete(geom)
    new = parent.add_geom()
    new.name, new.pos, new.quat = name, pos, quat
    for field, value in attrs.items():
      setattr(new, field, value)
```

**Khuyến nghị: (b)**, vì mục tiêu là đem ra real. `right_hand.xml` khi đó vẫn là bản sao
trung thực của mô tả phần cứng (URDF), còn việc gộp geom cho sensor là quyết định của
tầng sim. Lịch sử git cho thấy XML này chưa có script sinh trong repo (`d9df4329c` import
231 dòng từ convert bên ngoài, `c472bddea` bạn thêm tay `grasp_center` + 6 pad collision) —
nếu sau này export lại từ URDF thì (b) sống sót, (a) mất.

**Test chặn hồi quy** (chính là phép so ở trên): compile có/không re-parent, assert mọi
mảng per-geom khớp theo tên trừ `geom_bodyid`, và mass/inertia y hệt. Thêm assert: với mỗi
`CONTACT_BODIES`, mọi collision geom trong subtree của nó phải thuộc chính body đó.

Phải sửa **cả hai** sensor `hand_object_contact` và `hand_table_contact` (dùng chung pattern).
Ba sensor arm giữ `mode="body"` — đúng, vì link arm sở hữu geom trực tiếp.

#### Với sim2real: đây là **sửa lỗi lệch phần cứng**

Các pad tên là `*_force_sensor`. Nếu tay thật đọc được chúng thì trên robot thật, khối
contact/impulse được đo **tại pad**; trong sim hiện tại nó đo tại **cylinder ở hông đốt
dip** và không bao giờ tại pad. Tức kênh quan sát quan trọng nhất của RobustDexGrasp đang
đo sai bề mặt vật lý. Fix này đưa slot sensor của sim trùng vị trí cảm biến thật.

> ❓ **Cần xác nhận:** tay RH5-DG2 thật có đọc được 6 force sensor này không? Nếu có,
> thiết kế Phase 3 đổi đáng kể — RobustDexGrasp cần LSTM tái dựng contact vì Allegro
> **không** có cảm biến lực; tay này có thì feed thẳng được, bỏ bớt phần distillation.

---

### A3. Metric đọc kinematics cũ lúc reset

**Nguyên nhân.** Thứ tự trong `manager_based_rl_env.py:594-625`: event ghi **qpos** → các
manager chạy `reset()` → `sim.forward()` mới chạy sau đó (dòng 492). `root_link_pos_w` đọc
`data.xpos`, chỉ được cập nhật bởi `forward()`. Nên mọi hook `reset()` đọc `root_link_pos_w`
đều thấy pose của **episode trước**.

`ObjectDisplacement._root_pos_w` (`mdp/rewards.py:253-260`) đã né đúng bằng cách đọc thẳng
qpos/mocap. `ObjectLiftHeight.reset` (`mdp/metrics.py:65-68`) thì không.

**Đo được:**

```
object z trước reset: 1.323   → _initial_z ghi nhận: 1.323
object z sau  reset : 0.823   → lift height báo: -0.5 m
```

Chiều ngược lại nguy hiểm hơn: episode trước object rơi xuống sàn (z≈0.05) → episode sau
báo "nhấc được 0.77 m" suốt cả episode. Đó là lý do `lift_success = 0.91` ở **iteration 0**
với policy random, và 0.36–0.40 sau đó cũng là artifact.

**Sửa.** Tách `_root_pos_w` thành hàm dùng chung (đề xuất `mdp/utils.py`), gọi ở cả
`ObjectDisplacement` lẫn `ObjectLiftHeight`:

```python
def root_pos_at_reset(entity: Entity) -> torch.Tensor:
  """Root position readable inside a reset hook (before forward())."""
  if not entity.is_fixed_base:
    return entity.data.data.qpos[:, entity.indexing.free_joint_q_adr[:3]]
  mocap_id = entity.indexing.mocap_id
  if mocap_id is not None:
    return entity.data.data.mocap_pos[:, mocap_id]
  return entity.data.root_link_pos_w
```

Chỉ ảnh hưởng metric giám sát, không đụng reward. **Với real: không ảnh hưởng gì.**

---

## 3. Nhóm B — Physics: object không nằm yên

> **Đính chính (xem [06-experiment-report.md](06-experiment-report.md) §4).**
> Giả thuyết "ma sát chưa hội tụ" ở mục này đã được đo và **bác bỏ**:
> `iterations` 10→100 cho 0.99×, `impratio` 10→1 cho 1.00×. Nguyên nhân thật là
> **timestep** (dt 10→5 ms cho 0.42×). Ngoài ra 06 phát hiện một bug lớn hơn
> chưa có ở đây: **ratchet trọng lực** làm arm sụt 0.31 m/episode khi action=0.

Robot đứng im hoàn toàn (action = 0), 70 bước / 14 s:

| object | \|ω\| trung bình | \|ω\| max | trôi (m) |
|---|---|---|---|
| potted_meat_can | 1.04 rad/s | 7.3 | 0.02–0.13 |
| sugar_box | 0.20 | 9.3 | 0.00–0.11 |
| **tuna_fish_can** | **4.16** | 11.6 | 0.05–0.19 |

Góc quay tịnh chỉ ~1 rad trong khi \|ω\| ~4 rad/s → đây là **chattering** (rung ở contact),
không phải quay thật. z dao động 0.7867–0.7881 (1.4 mm). Với tuna_fish_can, riêng
`ω²·0.2 ≈ 3.5/bước` đã vượt sàn clip −2.0 → **cả episode ghim ở sàn, gradient bằng 0**.

Đây là nguyên nhân gốc của hành vi thoái hoá ở §0: chạm vào vật chỉ làm ω tăng thêm →
bị phạt → chiến lược tối ưu là **không chạm**.

### Nghi can chính: solver setting

`SimulationCfg` của task (`dexgrasp_env_cfg.py:196-208`) so với mặc định mjlab:

| | mặc định `MujocoCfg` | DexGrasp override |
|---|---|---|
| `iterations` | 100 | **10** |
| `ls_iterations` | 50 | **20** |
| `impratio` | 1.0 | **10** |
| `cone` | pyramidal | **elliptic** |

Elliptic cone + `impratio=10` là công thức ma sát **khó giải nhất**, mà lại được cấp số
vòng lặp solver **ít hơn mặc định 10 lần**, ở timestep 10 ms. Ma sát chưa hội tụ là
nguyên nhân kinh điển của chattering.

**Loại trừ được một nghi can:** `object_clearance = 0.002` (thả rơi 2 mm) **không** phải
gốc — bản gốc cũng thả 2 mm (`0.773 - lowest_point`, bàn ở 0.771) và vật của nó không rung.

### Nghi can phụ

- Cảnh báo lúc compile: `MULTICCD is enabled, but the scene contains CCD pairs without
  multicontact support: (CAPSULE,CYLINDER), (CAPSULE,MESH), (CYLINDER,CYLINDER),
  (CYLINDER,BOX), (CYLINDER,MESH). At most 1 contact will be generated for these pairs.`
  Các đốt ngón dùng **cylinder** → cylinder↔mesh(object) chỉ 1 điểm tiếp xúc.
  Cân nhắc đổi cylinder ngón sang capsule hoặc box.
- `nconmax=150 / njmax=1500` — chưa đo đỉnh thực tế. mjwarp **âm thầm bỏ contact** khi tràn.
- `solref="0.02 1"` mặc định ở dt=10 ms → hằng số thời gian contact = 2× timestep, đúng mức
  tối thiểu khuyến nghị. Có thể nới `solref` hoặc giảm timestep.
- Bản gốc dùng `world_->setERP(0.0)` + solver riêng của RaiSim — không có tương ứng 1-1,
  MuJoCo phải tự tune.

### Kế hoạch: quét đo, không sửa mò

Metric: `|ω|` trung bình và max của object khi robot đứng im, 70 bước, 3 vật
(potted_meat_can / sugar_box / tuna_fish_can). Mục tiêu `|ω| ≈ 0`.

| # | Thay đổi | Giả thuyết |
|---|---|---|
| 0 | baseline | 1.04 / 0.20 / 4.16 |
| 1 | `iterations` 10 → 100, `ls_iterations` 20 → 50 | ma sát hội tụ |
| 2 | `impratio` 10 → 1 | giảm độ cứng ma sát tương đối |
| 3 | `cone` elliptic → pyramidal | công thức dễ giải hơn |
| 4 | `timestep` 0.01 → 0.005, `decimation` 20 → 40 | giữ 5 Hz, contact ổn định hơn |
| 5 | cylinder ngón → capsule | nhiều điểm tiếp xúc hơn |
| 6 | tổ hợp rẻ nhất đạt mục tiêu | |

Sau khi chọn, **đo lại FPS** (baseline 2099 FPS trên 4060 Ti @ 352 env) để biết trả giá
bao nhiêu, và đo đỉnh `ncon` để chỉnh `nconmax`.

**Với sim2real:** solver hội tụ tốt hơn = contact thực tế hơn = khoảng cách sim2real nhỏ hơn.
Đây là fix thuần cải thiện.

---

## 4. Nhóm C — Khác biệt so với bản gốc

### C1. Clip −2 và terminal −10: bản gốc **không thực sự áp**

- `train.py:628`: `reward_r.clip(min=reward_clip)` — numpy trả array mới, **không gán lại**
  → sàn −2 chưa bao giờ có hiệu lực.
- `train.py:627`: `reward_r[i] = rewards_r[i]['reward_sum']` ghi đè giá trị RaiSim đã cộng
  terminal −10 → PPO cũng không thấy terminal reward.

Port áp cả hai (`manager_based_rl_env.py:467-470`, cấu hình ở `dexgrasp_env_cfg.py:212-213`).

**Hướng xử lý:** *không* bỏ clip trước khi sửa B1. Sàn −2 chỉ gây hại vì reward thô đang
là −9.1/bước; khi jitter hết, reward thô về dải bình thường và clip trở nên vô hại (thậm
chí hữu ích để chặn outlier). Trình tự đúng: sửa B1 → đo lại phân bố reward thô → rồi mới
quyết định giữ/nới/bỏ `reward_clip_min`.

Về `termination_reward = -10`: port áp cho **mọi** termination non-timeout, kể cả
`object_out_of_workspace` và `nan` (hai term port tự thêm). Bản gốc dành −10 riêng cho
hand-below-table. Đề xuất: giữ −10 cho `hand_below_table`, còn `object_out_of_workspace`
nên là truncation (không phạt) hoặc phạt nhẹ hơn — object bay khỏi bàn phần lớn là hệ quả
của B1 chứ không phải lỗi policy.

### C2. Friction

| cặp | bản gốc | port hiện tại |
|---|---|---|
| object ↔ ngón | 0.8 (`setMaterialPairProp`) | **1.0** (mặc định MuJoCo) |
| bàn ↔ object | 0.2 (`hardware.hpp:437`) | 0.2 ✓ |
| bàn ↔ ngón | 0.8 (default material) | **0.2** (bàn có `priority=1`) |

`dexgrasp_env_cfg.py:69` đặt `priority=1` cho geom bàn → mọi contact với bàn dùng 0.2,
kể cả tay. Bản gốc chỉ đặt 0.2 cho **bàn↔object**.

**Sửa:** bỏ `priority` khỏi geom bàn và dùng `<pair>` riêng cho bàn↔object ở 0.2; đặt
`friction` của geom object và geom tay về 0.8. Về lâu dài các số này nên **đo từ bàn/vật
thật** và domain-randomize (bản gốc random friction 0.5–0.9 cho student).

### C3. Soft joint limit 0.9 vs hard limit

`ur5e_rh5dg2_constants.py:234` đặt `soft_joint_pos_limit_factor=0.9`; action clamp theo
soft limit. Bản gốc clamp theo hard limit URDF. Ngón mất ~5–10% tầm gập, và policy không
bao giờ học dùng phần đó.

**Sửa:** đặt 1.0 để khớp bản gốc, **hoặc** giữ 0.9 nhưng bắt phía real clamp y hệt. Điều
tối kỵ là sim và real clamp khác nhau — khi đó cùng một action mang nghĩa khác nhau.

### C4. Thứ tự normalize trọng số affordance

Bản gốc (`train.py:223-224`) normalize **trước** rồi mới zero index wrist → tổng trọng số
là `16 × 32/33 ≈ 15.52`. Port (`mdp/rewards.py:74-79`) zero trước rồi normalize → tổng
đúng 16. Lệch ~3%. Nhỏ, nhưng sửa cho khớp thì rẻ.

### C5. Minibatch shuffle

Bản gốc `shuffle_batch=False` (minibatch tuần tự); rsl-rl mặc định xáo trộn. Ảnh hưởng
không đáng kể — ghi lại cho đầy đủ, không cần sửa.

---

## 5. Nhóm D — Rủi ro sim2real (nghiêm trọng hơn cả 3 bug)

### D1. PD gain ngón tay là số bịa

Bản gốc nạp gain **đã nhận dạng từ phần cứng**, có hẳn một tầng `hardware/` để load theo
từng robot:

| | reference | port |
|---|---|---|
| ngón (kp / kd) | **600 / 20** (`Allegrotemp.txt`) | **1.0 / 0.1** (`constants.py:56-57`) |
| arm (kp / kd) | **15800–16200 / 280–580** (`UR5Identification_id5hz.txt`) | giữ actuator menagerie |

Docstring `ur5e_rh5dg2_constants.py:7-10` cũng thừa nhận gain là đồng nhất chứ không khớp
quán tính.

**Vì sao nguy hiểm.** Với `kp = 1` N·m/rad và `actuatorfrcrange = ±1` N·m, ngón cần
**~1 rad sai số** mới ra đủ mô-men. Delta action là 0.015 rad/bước, nên sau khi chạm vật,
sai số tích lũy rất chậm → lực siết yếu. Bàn tay thật sẽ cứng hơn nhiều bậc: policy học
trên ngón mềm nhũn sẽ ra lệnh mà tay thật thực thi mạnh hơn hẳn → **bóp nát vật hoặc mất
ổn định**. Nó cũng giải thích một phần vì sao siết trong sim yếu.

**Sửa:**
1. Lấy gain thật của tay và arm; nếu không có thì nhận dạng như bản gốc đã làm
   (`UR5Identification_id5hz` — nhận dạng ở đúng tần số điều khiển 5 Hz).
2. Tái lập cấu trúc "gain nạp từ file theo robot" thay vì hằng số trong code.
3. Chỉnh lại `FINGER_ARMATURE` sau khi có gain thật.
4. Domain-randomize gain khi sang student (bản gốc: tay ±10%, arm ±5%).

### D2. Arm actuator không nhận dạng ở tần số điều khiển

Port giữ `<position>` actuator của menagerie cho UR5e. Menagerie tune cho vòng sim nhanh,
không phải vòng ngoài 5 Hz. Bản gốc nhận dạng riêng ở 5 Hz.

### D3. Giới hạn mô-men

`actuatorfrcrange="-1 1"` (±1 N·m) cho mọi khớp ngón — cần khớp với giới hạn thật của tay.

### D4. Standoff pre-grasp lệch thiết kế

Bản gốc đặt **wrist** tại `pos + R·hand_center` (do một lỗi dấu, hand center kết thúc ở
`pos + 2·R·hand_center`, lệch ~9.5 cm). Port đặt `grasp_center` **đúng** tại `pos`. Port
"đúng" hơn nhưng hình học standoff khác bản gốc. `APPROACH_DISTANCE = 0.25` được tune cho
offset `hand_center` của Allegro (0.095 m từ wrist), trong khi RH5-DG2 có `grasp_center`
cách wrist 0.121 m và đầu ngón còn vươn thêm ~0.13 m.

Đo được: lúc reset, keypoint gần nhất cách vật ~0.18 m, trung bình ~0.25 m. Nên **tune lại
`APPROACH_DISTANCE`** theo hình học tay này sau khi B1 xong (xem lại trong viewer).

### D5. Doc tham chiếu chết

`pregrasp/generator.py:10` trỏ tới `documents/problems/pregrasp-ur5e-tuning.md` — file
không tồn tại. Hoặc viết file đó, hoặc bỏ tham chiếu.

---

## 6. Nhóm E — Vệ sinh repo

Commit `325ec45 refactor` xoá các task khác nhưng để lại phần phụ thuộc:

- `uv run pytest` → **16 collection error** (test của velocity / tracking / cartpole /
  manipulation / g1 / go1). `uv run ty check` → **36 diagnostics** cùng nguyên nhân.
  CLAUDE.md yêu cầu `make check` và `make test` phải pass.
- `scripts/csv_to_npz.py` import `mjlab.tasks.tracking...`
- `scripts/visualize_terrain.py`, `scripts/export_scene.py` import `get_go1_robot_cfg`
  (không còn export)
- `scripts/demo.py` chạy task `Mjlab-Tracking-Flat-Unitree-G1` đã xoá
- `pyproject.toml` (chưa commit) thêm `render-hand = "mjlab.scripts.render_hand:main"`
  nhưng `src/mjlab/scripts/render_hand.py` không tồn tại

**Sửa:** xoá test + script mồ côi, gỡ entry point `render-hand` (hoặc thêm file), gỡ
`demo`/`export-scene` khỏi `[project.scripts]` nếu không còn dùng.

---

## 7. Thứ tự thực thi

```
1. A1  impulse mean          — 1 dòng, mở lại tín hiệu impulse
2. A2  re-parent pad geom    — phương án (b) + test chặn hồi quy
3. A3  metric đọc qpos       — để tin được cái đang theo dõi
   ── mốc: chạy 200 iteration, xác nhận affordance_contact tăng ──
4. B1  quét solver setting   — cần đo, không đoán
   ── mốc: |ω| idle ≈ 0, đo lại FPS + đỉnh ncon ──
5. C1  đo lại phân bố reward thô → quyết định clip / termination_reward
6. C2  friction; C3 soft limit; C4 thứ tự normalize
   ── mốc: smoke train 500 iteration, so với baseline advid9la ──
7. D1/D2 PD gain thật        — chặn cửa sim2real, cần dữ liệu phần cứng
8. D4  tune APPROACH_DISTANCE trong viewer
9. E   dọn repo, make check + make test pass
```

Bước 1–3 làm được ngay và độc lập. Bước 4 là thứ **thực sự chặn** training. Bước 7 chặn
việc đem ra real và cần dữ liệu từ bạn.

---

## 8. Câu hỏi cần xác nhận

1. **Tay RH5-DG2 thật có đọc được 6 force sensor không?** Quyết định thiết kế Phase 3
   (student): feed thẳng hay phải LSTM tái dựng như bản gốc.
2. **Có lấy được PD gain thật của tay và arm không**, hay cần nhận dạng như bản gốc?
3. **`right_hand.xml` có nguồn sinh bên ngoài** mà sau này sẽ export lại không? Quyết
   định A2 phương án (a) hay (b).
4. Bàn và vật thật có số liệu ma sát không (cho C2 và domain randomization)?

---

## Phụ lục — nguồn số liệu

| Số liệu | Cách đo |
|---|---|
| 189 vs 429 contact activation | env 8 worlds × 70 bước, sensor body-mode vs subtree-mode song song |
| impulse max 1.6657 | `sensor_impulse` trên `hand_object_contact`, ngón siết tối đa |
| lift height −0.5 m sau reset | teleport object +0.5 m, reset, đọc `ObjectLiftHeight._initial_z` |
| \|ω\| idle 3 vật | action = 0, 70 control step, `root_link_ang_vel_w` |
| tương đương pad move | compile 2 spec, so mọi mảng per-geom khớp theo tên |
| body vs subtree semantics | scene tối giản: quả cầu ở body con chạm sàn |
| phân rã reward/bước | wandb `advid9la`, `Episode_Reward × 14 / mean_episode_length` |
| PD gain bản gốc | `hardware/hand/Allegrotemp.txt`, `hardware/arm/UR5Identification_id5hz.txt` |
